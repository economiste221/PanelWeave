"""Lecture vidéo en flux et prétraitement des frames.

* Décodage frame par frame avec PyAV (timestamps exacts, VFR supporté) ;
  repli sur ``cv2.VideoCapture`` uniquement si PyAV ne peut pas ouvrir le fichier.
* Aucune vidéo n'est chargée entièrement en mémoire : :meth:`VideoReader.frames`
  est un générateur ; :class:`FrameRingBuffer` borne les fenêtres glissantes.
* Prétraitement : version réduite en niveaux de gris pour l'estimation du
  mouvement, masque des zones d'exclusion configurables.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from collections.abc import Iterable, Iterator
from fractions import Fraction
from pathlib import Path
from typing import Protocol

import av
import av.container
import av.error
import cv2
import numpy as np

from panelrecon.core.config import ExclusionZone, PreprocessConfig, VideoIOConfig
from panelrecon.core.models import (
    CancellationToken,
    FrameObs,
    ImageU8,
    MaskU8,
    VideoInfo,
)

logger = logging.getLogger(__name__)

BACKEND_PYAV = "pyav"
# Fraction de période tolérée sur les timestamps pour le plafonnement de cadence.
_BUCKET_TOLERANCE = 0.1
BACKEND_OPENCV = "opencv"


class VideoError(RuntimeError):
    """Erreur générique de lecture vidéo."""


class VideoOpenError(VideoError):
    """Le fichier ne peut être ouvert par aucun backend."""


class VideoDecodeError(VideoError):
    """Trop d'erreurs de décodage pendant la lecture."""


# ---------------------------------------------------------------------------
# Découverte de fichiers
# ---------------------------------------------------------------------------


def discover_videos(
    inputs: Iterable[Path], extensions: Iterable[str], recursive: bool = True
) -> list[Path]:
    """Liste triée et dédupliquée des vidéos trouvées dans ``inputs`` (fichiers ou dossiers)."""
    exts = {e.lower() for e in extensions}
    found: set[Path] = set()
    for item in inputs:
        path = Path(item).expanduser()
        if path.is_file():
            if path.suffix.lower() in exts:
                found.add(path.resolve())
            else:
                logger.warning("Ignoré (extension non reconnue) : %s", path)
        elif path.is_dir():
            candidates = path.rglob("*") if recursive else path.glob("*")
            for candidate in candidates:
                if (
                    candidate.is_file()
                    and candidate.suffix.lower() in exts
                    and not candidate.name.startswith(".")
                ):
                    found.add(candidate.resolve())
        else:
            logger.warning("Entrée introuvable : %s", path)
    return sorted(found)


# ---------------------------------------------------------------------------
# Prétraitement
# ---------------------------------------------------------------------------


def compute_proxy_factor(width: int, height: int, long_side: int) -> float:
    """Facteur ``≤ 1`` ramenant le côté long à ``long_side`` (jamais d'agrandissement)."""
    if width <= 0 or height <= 0:
        raise ValueError(f"Dimensions invalides : {width}x{height}")
    if long_side <= 0:
        raise ValueError(f"long_side invalide : {long_side}")
    return min(1.0, long_side / float(max(width, height)))


def proxy_size(width: int, height: int, factor: float) -> tuple[int, int]:
    """Taille ``(w, h)`` de la version réduite (au moins 1 px)."""
    return max(1, int(round(width * factor))), max(1, int(round(height * factor)))


def to_gray(image: ImageU8) -> ImageU8:
    if image.ndim == 2:
        return image
    return np.asarray(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), dtype=np.uint8)


def make_proxy(image: ImageU8, factor: float) -> ImageU8:
    """Réduit ``image`` d'un facteur ``factor`` (INTER_AREA, anti-repliement)."""
    if factor >= 1.0:
        return image.copy()
    w, h = proxy_size(image.shape[1], image.shape[0], factor)
    return np.asarray(cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA), dtype=np.uint8)


def build_exclusion_mask(
    height: int, width: int, zones: Iterable[ExclusionZone]
) -> MaskU8:
    """Masque uint8 : 255 = pixel utilisable, 0 = pixel dans une zone d'exclusion.

    Les zones sont en coordonnées relatives ``(x0, y0, x1, y1)`` ∈ [0, 1] ; un
    pixel est exclu dès qu'il intersecte la zone (arrondi vers l'extérieur).
    """
    mask: MaskU8 = np.full((height, width), 255, dtype=np.uint8)
    for x0, y0, x1, y1 in zones:
        c0 = max(0, int(math.floor(x0 * width)))
        c1 = min(width, int(math.ceil(x1 * width)))
        r0 = max(0, int(math.floor(y0 * height)))
        r1 = min(height, int(math.ceil(y1 * height)))
        if c1 > c0 and r1 > r0:
            mask[r0:r1, c0:c1] = 0
    return mask


def _rotate_quarter_turns(image: ImageU8, rotation_deg: int) -> ImageU8:
    """Applique une rotation d'affichage multiple de 90° (sens trigonométrique)."""
    k = int(round(rotation_deg / 90.0)) % 4
    if k == 0:
        return image
    return np.ascontiguousarray(np.rot90(image, k=k))


# ---------------------------------------------------------------------------
# Tampon circulaire
# ---------------------------------------------------------------------------


class FrameRingBuffer:
    """Fenêtre glissante bornée de :class:`FrameObs` (mémoire constante)."""

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("La capacité doit être ≥ 1")
        self._frames: deque[FrameObs] = deque(maxlen=capacity)

    @property
    def capacity(self) -> int:
        maxlen = self._frames.maxlen
        assert maxlen is not None
        return maxlen

    def append(self, frame: FrameObs) -> None:
        if self._frames and frame.index <= self._frames[-1].index:
            raise ValueError(
                f"Indices non croissants : {frame.index} après {self._frames[-1].index}"
            )
        self._frames.append(frame)

    def clear(self) -> None:
        self._frames.clear()

    def latest(self) -> FrameObs:
        if not self._frames:
            raise IndexError("Tampon vide")
        return self._frames[-1]

    def get(self, index: int) -> FrameObs | None:
        """Frame d'indice vidéo ``index`` si elle est encore dans le tampon."""
        for frame in reversed(self._frames):
            if frame.index == index:
                return frame
            if frame.index < index:
                break
        return None

    def __len__(self) -> int:
        return len(self._frames)

    def __iter__(self) -> Iterator[FrameObs]:
        return iter(tuple(self._frames))


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


class _RawFrame:
    __slots__ = ("image", "pts", "time_s", "rotation_deg")

    def __init__(self, image: ImageU8, pts: int | None, time_s: float | None, rotation_deg: int):
        self.image = image
        self.pts = pts
        self.time_s = time_s
        self.rotation_deg = rotation_deg


class _Backend(Protocol):
    info: VideoInfo
    decode_errors: int

    def iter_raw(self) -> Iterator[_RawFrame]: ...

    def close(self) -> None: ...


class _PyAVBackend:
    def __init__(self, path: Path, cfg: VideoIOConfig, probe_rotation: bool = False) -> None:
        self._cfg = cfg
        self.decode_errors = 0
        try:
            self._container = av.open(str(path), mode="r")
        except (av.error.FFmpegError, OSError, ValueError) as exc:
            raise VideoOpenError(f"PyAV ne peut pas ouvrir {path} : {exc}") from exc
        try:
            if not self._container.streams.video:
                raise VideoOpenError(f"Aucun flux vidéo dans {path}")
            self._stream = self._container.streams.video[0]
            self._stream.thread_type = "AUTO"
            if cfg.decode_threads > 0:
                self._stream.codec_context.thread_count = cfg.decode_threads
            self.info = self._probe(path, probe_rotation)
        except BaseException:
            self._container.close()
            raise

    def _first_frame_rotation(self) -> int:
        """Rotation d'affichage lue sur la première frame décodable (0 si aucune)."""
        try:
            for frame in self._container.decode(self._stream):
                return int(frame.rotation)
        except av.error.FFmpegError as exc:
            logger.debug("Lecture de la rotation impossible : %s", exc)
        return 0

    def _probe(self, path: Path, probe_rotation: bool) -> VideoInfo:
        stream = self._stream
        ctx = stream.codec_context
        width, height = int(ctx.width), int(ctx.height)
        if width <= 0 or height <= 0:
            raise VideoOpenError(f"Dimensions vidéo invalides dans {path}")
        rate = stream.average_rate or stream.guessed_rate or stream.base_rate
        fps = float(rate) if rate else 0.0
        duration: float | None = None
        if stream.duration is not None and stream.time_base is not None:
            duration = float(stream.duration * stream.time_base)
        elif self._container.duration is not None:
            duration = float(self._container.duration) / av.time_base
        frame_count = int(stream.frames) if stream.frames else None
        if frame_count is None and duration is not None and fps > 0:
            frame_count = int(round(duration * fps))
        rotation = self._first_frame_rotation() if probe_rotation else 0
        if self._cfg.apply_display_rotation and rotation % 180 != 0 and rotation % 90 == 0:
            width, height = height, width
        return VideoInfo(
            path=path,
            width=width,
            height=height,
            fps=fps,
            duration_s=duration,
            frame_count=frame_count,
            codec=str(ctx.name),
            backend=BACKEND_PYAV,
            rotation_deg=rotation,
        )

    def iter_raw(self) -> Iterator[_RawFrame]:
        for packet in self._container.demux(self._stream):
            try:
                decoded = packet.decode()
            except (av.error.InvalidDataError, av.error.UndefinedError) as exc:
                self.decode_errors += 1
                logger.warning(
                    "%s : paquet corrompu ignoré (pts=%s) : %s",
                    self.info.path.name,
                    packet.pts,
                    exc,
                )
                if self.decode_errors > self._cfg.max_decode_errors:
                    raise VideoDecodeError(
                        f"{self.info.path} : plus de {self._cfg.max_decode_errors} "
                        "erreurs de décodage"
                    ) from exc
                continue
            for frame in decoded:
                image = np.asarray(frame.to_ndarray(format="bgr24"), dtype=np.uint8)
                time_s = float(frame.time) if frame.time is not None else None
                yield _RawFrame(image, frame.pts, time_s, int(frame.rotation))

    def close(self) -> None:
        self._container.close()


class _OpenCVBackend:
    def __init__(self, path: Path, cfg: VideoIOConfig) -> None:
        self._cfg = cfg
        self.decode_errors = 0
        self._cap = cv2.VideoCapture(str(path))
        if not self._cap.isOpened():
            self._cap.release()
            raise VideoOpenError(f"OpenCV ne peut pas ouvrir {path}")
        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            self._cap.release()
            raise VideoOpenError(f"Dimensions vidéo invalides dans {path}")
        fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0)
        count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.info = VideoInfo(
            path=path,
            width=width,
            height=height,
            fps=fps,
            duration_s=(count / fps) if fps > 0 and count > 0 else None,
            frame_count=count if count > 0 else None,
            codec="unknown",
            backend=BACKEND_OPENCV,
        )

    def iter_raw(self) -> Iterator[_RawFrame]:
        while True:
            ok, image = self._cap.read()
            if not ok or image is None:
                return
            msec = float(self._cap.get(cv2.CAP_PROP_POS_MSEC))
            time_s = msec / 1000.0 if msec >= 0 else None
            yield _RawFrame(np.ascontiguousarray(image, dtype=np.uint8), None, time_s, 0)

    def close(self) -> None:
        self._cap.release()


def _open_backend(path: Path, cfg: VideoIOConfig, probe: bool = False) -> _Backend:
    """Ouvre PyAV, ou OpenCV en repli. ``probe`` lit en plus la rotation d'affichage
    (consomme la première frame : à n'utiliser que pour un backend jetable)."""
    try:
        return _PyAVBackend(path, cfg, probe_rotation=probe)
    except VideoOpenError as pyav_exc:
        if not cfg.allow_opencv_fallback:
            raise
        logger.warning("%s ; tentative de repli sur cv2.VideoCapture", pyav_exc)
        try:
            return _OpenCVBackend(path, cfg)
        except VideoOpenError as cv_exc:
            raise VideoOpenError(f"{pyav_exc} ; repli OpenCV : {cv_exc}") from cv_exc


def probe_video(path: Path, cfg: VideoIOConfig) -> VideoInfo:
    """Métadonnées d'une vidéo sans la décoder."""
    path = Path(path)
    if not path.is_file():
        raise VideoOpenError(f"Fichier introuvable : {path}")
    backend = _open_backend(path, cfg, probe=True)
    try:
        return backend.info
    finally:
        backend.close()


# ---------------------------------------------------------------------------
# Lecteur
# ---------------------------------------------------------------------------


class VideoReader:
    """Lecteur vidéo en flux.

    Exemple ::

        with VideoReader(path, cfg.video, cfg.preprocess) as reader:
            for obs in reader.frames():
                ...

    Les indices de :class:`FrameObs` sont les rangs des frames **effectivement
    décodées**, dans l'ordre de présentation, **avant** sous-échantillonnage : ils
    restent stables quel que soit ``frame_step``/``max_fps``. Si le flux est
    endommagé, des frames peuvent manquer sans trou dans les indices ; l'instant
    de présentation fiable est alors ``time_s`` (issu des pts).
    """

    def __init__(
        self,
        path: Path,
        video_cfg: VideoIOConfig,
        preprocess_cfg: PreprocessConfig,
        cancel: CancellationToken | None = None,
    ) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise VideoOpenError(f"Fichier introuvable : {self.path}")
        self._video_cfg = video_cfg
        self._pre_cfg = preprocess_cfg
        self._cancel = cancel
        self._backend: _Backend | None = None
        self._rotation_warned = False
        backend = _open_backend(self.path, video_cfg, probe=True)
        try:
            self.info = backend.info
        finally:
            backend.close()
        self.decode_errors = 0
        self.frames_decoded = 0

    def __enter__(self) -> VideoReader:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        if self._backend is not None:
            self._backend.close()
            self._backend = None

    @property
    def proxy_factor(self) -> float:
        return compute_proxy_factor(self.info.width, self.info.height, self._pre_cfg.motion_long_side)

    def frames(
        self,
        start_index: int = 0,
        stop_index: int | None = None,
        compute_proxy: bool = True,
    ) -> Iterator[FrameObs]:
        """Génère les frames d'indice ``start_index ≤ i < stop_index``.

        Chaque appel relit la vidéo depuis le début (décodage séquentiel exact,
        indépendant de la précision du seek du conteneur).
        """
        if start_index < 0:
            raise ValueError("start_index doit être ≥ 0")
        if stop_index is not None and stop_index < start_index:
            raise ValueError("stop_index doit être ≥ start_index")
        self.close()
        backend = _open_backend(self.path, self._video_cfg)
        self._backend = backend
        self.decode_errors = 0
        self.frames_decoded = 0
        step = self._video_cfg.frame_step
        max_fps = self._video_cfg.max_fps
        last_bucket: int | None = None
        fps = backend.info.fps if backend.info.fps > 0 else 0.0
        try:
            for index, raw in enumerate(backend.iter_raw()):
                self.frames_decoded = index + 1
                self.decode_errors = backend.decode_errors
                if self._cancel is not None:
                    self._cancel.raise_if_cancelled()
                if stop_index is not None and index >= stop_index:
                    break
                if index < start_index or (index - start_index) % step != 0:
                    continue
                time_s = raw.time_s
                if time_s is None:
                    time_s = index / fps if fps > 0 else float(index)
                if max_fps > 0.0:
                    # Au plus une frame par créneau de 1/max_fps s. La tolérance absorbe
                    # l'arrondi des pts (ex. intervalles de 16/17 ms à 60 i/s).
                    bucket = math.floor(time_s * max_fps + _BUCKET_TOLERANCE)
                    if last_bucket is not None and bucket <= last_bucket:
                        continue
                    last_bucket = bucket
                image = self._apply_rotation(raw.image, raw.rotation_deg)
                yield self._make_obs(index, time_s, raw.pts, image, compute_proxy)
        finally:
            self.decode_errors = backend.decode_errors
            self.close()

    def _apply_rotation(self, image: ImageU8, rotation_deg: int) -> ImageU8:
        if rotation_deg == 0:
            return image
        if not self._video_cfg.apply_display_rotation:
            if not self._rotation_warned:
                logger.warning(
                    "%s : rotation d'affichage %d° ignorée (apply_display_rotation=False)",
                    self.path.name,
                    rotation_deg,
                )
                self._rotation_warned = True
            return image
        if rotation_deg % 90 != 0:
            if not self._rotation_warned:
                logger.warning(
                    "%s : rotation d'affichage %d° non multiple de 90°, ignorée",
                    self.path.name,
                    rotation_deg,
                )
                self._rotation_warned = True
            return image
        return _rotate_quarter_turns(image, rotation_deg)

    def _make_obs(
        self, index: int, time_s: float, pts: int | None, image: ImageU8, compute_proxy: bool
    ) -> FrameObs:
        image = np.ascontiguousarray(image)
        proxy_gray: ImageU8 | None = None
        factor = 1.0
        if compute_proxy:
            factor = compute_proxy_factor(image.shape[1], image.shape[0], self._pre_cfg.motion_long_side)
            proxy_gray = make_proxy(to_gray(image), factor)
            # Facteur effectif après arrondi des dimensions (horizontal), pour une
            # reconversion exacte proxy → natif.
            factor = proxy_gray.shape[1] / float(image.shape[1])
        return FrameObs(
            index=index,
            time_s=time_s,
            pts=pts,
            image=image,
            proxy_gray=proxy_gray,
            proxy_factor=min(1.0, factor),
        )


# ---------------------------------------------------------------------------
# Encodage
# ---------------------------------------------------------------------------

_ENCODERS: dict[str, tuple[str, dict[str, str]]] = {
    ".mp4": ("libx264", {"preset": "veryfast"}),
    ".mov": ("libx264", {"preset": "veryfast"}),
    ".mkv": ("libx264", {"preset": "veryfast"}),
    ".webm": ("libvpx-vp9", {"b": "0", "deadline": "realtime", "cpu-used": "8"}),
}
# Base de temps des vidéos à fréquence variable : la milliseconde.
VFR_TIME_BASE = Fraction(1, 1000)


class VideoEncoder:
    """Encodeur vidéo en flux (PyAV), frame par frame, à cadence fixe ou variable.

    En mode ``variable_frame_rate``, :meth:`write` exige l'instant de chaque
    frame ; il est arrondi à la milliseconde (base de temps du flux).
    """

    def __init__(
        self,
        path: Path,
        width: int,
        height: int,
        fps: int,
        crf: int = 18,
        variable_frame_rate: bool = False,
    ) -> None:
        suffix = Path(path).suffix.lower()
        if suffix not in _ENCODERS:
            raise ValueError(f"Conteneur non supporté pour l'encodage : {suffix}")
        if width <= 0 or height <= 0 or width % 2 or height % 2:
            raise ValueError(f"Dimensions paires et positives requises (yuv420p) : {width}x{height}")
        if fps <= 0:
            raise ValueError("fps doit être > 0")
        if not 0 <= crf <= 51:
            raise ValueError("crf doit être dans [0, 51]")
        codec, options = _ENCODERS[suffix]
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._width, self._height, self._fps = width, height, fps
        self._vfr = variable_frame_rate
        self._count = 0
        self._last_pts: int | None = None
        self._container: av.container.OutputContainer | None = av.open(str(self.path), mode="w")
        try:
            stream = self._container.add_stream(
                codec, rate=fps, options={**options, "crf": str(crf)}
            )
            assert isinstance(stream, av.VideoStream)
            stream.width = width
            stream.height = height
            stream.pix_fmt = "yuv420p"
            if variable_frame_rate:
                stream.codec_context.time_base = VFR_TIME_BASE
            self._stream = stream
        except BaseException:
            self._container.close()
            raise

    def __enter__(self) -> VideoEncoder:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def frames_written(self) -> int:
        return self._count

    def write(self, image: ImageU8, time_s: float | None = None) -> float:
        """Encode une frame BGR ; renvoie l'instant effectivement enregistré."""
        if image.shape != (self._height, self._width, 3) or image.dtype != np.uint8:
            raise ValueError(
                f"Frame BGR uint8 {self._width}x{self._height} attendue, reçu {image.shape}"
            )
        frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(image), format="bgr24")
        if self._vfr:
            if time_s is None:
                raise ValueError("time_s est requis en mode fréquence variable")
            pts = int(round(time_s * 1000))
            frame.time_base = VFR_TIME_BASE
        else:
            pts = self._count
            frame.time_base = Fraction(1, self._fps)
        if self._last_pts is not None and pts <= self._last_pts:
            raise ValueError(f"Timestamps non strictement croissants ({pts} après {self._last_pts})")
        frame.pts = pts
        if self._container is None:
            raise RuntimeError("Encodeur déjà fermé")
        for packet in self._stream.encode(frame):
            self._container.mux(packet)
        self._last_pts = pts
        self._count += 1
        return float(pts * frame.time_base)

    def close(self) -> None:
        if self._container is None:
            return
        try:
            for packet in self._stream.encode():
                self._container.mux(packet)
        finally:
            self._container.close()
            self._container = None
