"""Générateur de vidéos synthétiques avec vérité terrain exacte.

Une vidéo est une suite de *plans* (un plan = un panel). Pour chaque plan, le
panel haute résolution est animé par une similarité ``panel → écran`` obtenue en
interpolant des poses clés (échelle en log, centre affiché, rotation) avec une
fonction d'easing, puis composé par-dessus un fond flou (agrandissement flouté
du panel, fixe ou suiveur) ou uni. Options couvrant les cas limites : frames
dupliquées, fondu enchaîné entre panels, sous-titres incrustés, panel peu
texturé, fréquence d'images variable.

La vérité terrain enregistre, pour chaque frame, la transformation exacte
``panel → écran`` et la composition (plan pur ou mélange de transition).
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Final

import cv2
import numpy as np
from numpy.typing import NDArray

from panelrecon.core.geometry import coverage_alpha, full_coverage_mask, to_u8, warp_similarity
from panelrecon.core.models import ImageU8, MaskU8, Sequence, SimilarityTransform
from panelrecon.core.video_io import VideoEncoder

logger = logging.getLogger(__name__)

F32 = NDArray[np.float32]

GROUND_TRUTH_VERSION: Final[int] = 1
GROUND_TRUTH_FILENAME: Final[str] = "ground_truth.json"


# ---------------------------------------------------------------------------
# Easing
# ---------------------------------------------------------------------------


def _linear(u: float) -> float:
    return u


def _ease_in(u: float) -> float:
    return u**3


def _ease_out(u: float) -> float:
    return 1.0 - (1.0 - u) ** 3


def _ease_in_out(u: float) -> float:
    return 4.0 * u**3 if u < 0.5 else 1.0 - (-2.0 * u + 2.0) ** 3 / 2.0


def _sine(u: float) -> float:
    return 0.5 - 0.5 * math.cos(math.pi * u)


EASINGS: Final[dict[str, Callable[[float], float]]] = {
    "linear": _linear,
    "ease_in": _ease_in,
    "ease_out": _ease_out,
    "ease_in_out": _ease_in_out,
    "sine": _sine,
}


# ---------------------------------------------------------------------------
# Spécifications
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Pose:
    """Cadrage : le point ``(center_u, center_v)`` du panel (pixels) est affiché
    au centre de l'écran, avec l'échelle ``scale`` et la rotation ``theta``."""

    scale: float
    center_u: float
    center_v: float
    theta: float = 0.0

    def __post_init__(self) -> None:
        if not self.scale > 0:
            raise ValueError(f"Échelle de pose invalide : {self.scale}")

    def to_transform(self, screen_w: int, screen_h: int) -> SimilarityTransform:
        """Similarité ``panel → écran`` : ``p' = s·R(θ)·(p − c) + o``."""
        ox, oy = (screen_w - 1) / 2.0, (screen_h - 1) / 2.0
        c, s = self.scale * math.cos(self.theta), self.scale * math.sin(self.theta)
        tx = ox - (c * self.center_u - s * self.center_v)
        ty = oy - (s * self.center_u + c * self.center_v)
        return SimilarityTransform(self.scale, self.theta, tx, ty)


def interpolate_pose(a: Pose, b: Pose, u: float) -> Pose:
    """Interpolation : échelle géométrique (zoom perçu uniforme), centre et angle linéaires."""
    return Pose(
        scale=math.exp((1 - u) * math.log(a.scale) + u * math.log(b.scale)),
        center_u=(1 - u) * a.center_u + u * b.center_u,
        center_v=(1 - u) * a.center_v + u * b.center_v,
        theta=(1 - u) * a.theta + u * b.theta,
    )


@dataclass(frozen=True)
class PanelSpec:
    """Panel procédural (``image_path`` absent) ou image fournie."""

    width: int = 1200
    height: int = 900
    texture: str = "rich"  # "rich" (manhwa détaillé) ou "flat" (aplats peu texturés)
    seed: int = 0
    image_path: Path | None = None

    def __post_init__(self) -> None:
        if self.texture not in ("rich", "flat"):
            raise ValueError(f"Texture inconnue : {self.texture!r}")
        if self.image_path is None and (self.width < 16 or self.height < 16):
            raise ValueError("Panel trop petit")


@dataclass(frozen=True)
class ShotSpec:
    """Un plan : un panel animé entre des poses clés."""

    panel: PanelSpec
    keyframes: tuple[tuple[float, Pose], ...]
    n_poses: int
    easing: str = "linear"
    hold: int = 1  # chaque pose est répétée `hold` fois (frames dupliquées)

    def __post_init__(self) -> None:
        if not self.keyframes:
            raise ValueError("Au moins une pose clé est requise")
        times = [t for t, _ in self.keyframes]
        if times[0] != 0.0 or (len(times) > 1 and times[-1] != 1.0):
            raise ValueError("Les poses clés doivent couvrir [0, 1]")
        if any(t1 <= t0 for t0, t1 in zip(times, times[1:])):
            raise ValueError("Instants des poses clés non strictement croissants")
        if self.n_poses < 1 or self.hold < 1:
            raise ValueError("n_poses et hold doivent être ≥ 1")
        if self.easing not in EASINGS:
            raise ValueError(f"Easing inconnu : {self.easing!r}")

    @property
    def n_frames(self) -> int:
        return self.n_poses * self.hold

    def pose_at(self, k: int) -> Pose:
        """Pose numéro ``k`` ∈ [0, n_poses) ; l'easing s'applique à chaque segment."""
        if len(self.keyframes) == 1:
            return self.keyframes[0][1]
        u = k / (self.n_poses - 1) if self.n_poses > 1 else 0.0
        for (t0, p0), (t1, p1) in zip(self.keyframes, self.keyframes[1:]):
            if u <= t1:
                local = (u - t0) / (t1 - t0)
                return interpolate_pose(p0, p1, EASINGS[self.easing](local))
        return self.keyframes[-1][1]


@dataclass(frozen=True)
class BackgroundSpec:
    """Fond : ``blur_static`` (agrandissement flou fixe), ``blur_follow`` (qui suit
    le cadrage du panel) ou ``solid`` (couleur unie)."""

    mode: str = "blur_static"
    blur_sigma: float = 12.0
    darken: float = 0.55
    cover_zoom: float = 1.15
    color: tuple[int, int, int] = (24, 24, 24)

    def __post_init__(self) -> None:
        if self.mode not in ("blur_static", "blur_follow", "solid"):
            raise ValueError(f"Mode de fond inconnu : {self.mode!r}")
        if not 0.0 <= self.darken <= 1.0 or self.blur_sigma < 0 or self.cover_zoom < 1.0:
            raise ValueError("Paramètres de fond invalides")


@dataclass(frozen=True)
class SubtitleSpec:
    """Sous-titre fixe incrusté (centré horizontalement, en bas de l'écran)."""

    text: str = "Ceci est un sous-titre incruste"
    font_scale: float = 0.9
    bottom_margin: float = 0.06  # fraction de la hauteur d'écran


@dataclass(frozen=True)
class VideoSpec:
    name: str
    shots: tuple[ShotSpec, ...]
    screen_width: int = 640
    screen_height: int = 360
    fps: int = 25
    crossfade_frames: int = 0
    background: BackgroundSpec = field(default_factory=BackgroundSpec)
    subtitle: SubtitleSpec | None = None
    variable_frame_rate: bool = False
    crf: int = 14
    container: str = ".mp4"
    seed: int = 0

    def __post_init__(self) -> None:
        if not self.shots:
            raise ValueError("Au moins un plan est requis")
        if self.screen_width % 2 or self.screen_height % 2:
            raise ValueError("Dimensions d'écran paires requises")
        if self.crossfade_frames < 0:
            raise ValueError("crossfade_frames doit être ≥ 0")


# ---------------------------------------------------------------------------
# Panels procéduraux
# ---------------------------------------------------------------------------


def _random_polygon(rng: np.random.Generator, w: int, h: int, size: float) -> NDArray[np.int32]:
    cx, cy = rng.uniform(0, w), rng.uniform(0, h)
    n = int(rng.integers(3, 8))
    angles = np.sort(rng.uniform(0, 2 * np.pi, n))
    radii = rng.uniform(0.4, 1.0, n) * size
    pts = np.stack([cx + radii * np.cos(angles), cy + radii * np.sin(angles)], axis=1)
    return np.round(pts).astype(np.int32)


def _random_text(rng: np.random.Generator, n: int) -> str:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ!?"
    return "".join(letters[i] for i in rng.integers(0, len(letters), n))


def make_panel(spec: PanelSpec) -> ImageU8:
    """Panel BGR uint8 : image fournie, ou dessin procédural déterministe."""
    if spec.image_path is not None:
        image = cv2.imread(str(spec.image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Image de panel illisible : {spec.image_path}")
        return np.asarray(image, dtype=np.uint8)
    rng = np.random.default_rng(spec.seed)
    w, h = spec.width, spec.height
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    top = rng.uniform(150, 255, 3).astype(np.float32)
    bottom = rng.uniform(80, 230, 3).astype(np.float32)
    ramp = (yy / max(1, h - 1))[..., None]
    img = (top * (1 - ramp) + bottom * ramp).astype(np.float32)
    unit = min(w, h)
    if spec.texture == "flat":
        canvas = to_u8(img)
        for _ in range(4):
            poly = _random_polygon(rng, w, h, unit * 0.45)
            color = tuple(int(c) for c in rng.integers(40, 230, 3))
            cv2.fillPoly(canvas, [poly], color, lineType=cv2.LINE_AA)
        return np.asarray(cv2.GaussianBlur(canvas, (0, 0), unit * 0.004 + 0.5), dtype=np.uint8)

    # Trame de points (screentone) sur une bande diagonale.
    period = max(4, unit // 90)
    dots = ((xx % period - period / 2) ** 2 + (yy % period - period / 2) ** 2) < (period * 0.3) ** 2
    band = np.abs((xx - yy * 0.7) - w * 0.3) < unit * 0.25
    img[dots & band] *= 0.55
    canvas = to_u8(img)
    line = max(2, unit // 250)
    for _ in range(40):
        poly = _random_polygon(rng, w, h, unit * rng.uniform(0.04, 0.2))
        color = tuple(int(c) for c in rng.integers(0, 256, 3))
        cv2.fillPoly(canvas, [poly], color, lineType=cv2.LINE_AA)
        cv2.polylines(canvas, [poly], True, (15, 15, 15), line, lineType=cv2.LINE_AA)
    # Hachures.
    for _ in range(6):
        x0, y0 = int(rng.uniform(0, w)), int(rng.uniform(0, h))
        length = int(unit * rng.uniform(0.08, 0.2))
        for k in range(0, length, max(3, line * 3)):
            cv2.line(canvas, (x0 + k, y0), (x0 + k + length // 3, y0 + length // 3), (30, 30, 30),
                     max(1, line // 2), lineType=cv2.LINE_AA)
    # Bulles de dialogue avec texte.
    for _ in range(5):
        center = (int(rng.uniform(0.1, 0.9) * w), int(rng.uniform(0.1, 0.9) * h))
        axes = (int(unit * rng.uniform(0.08, 0.14)), int(unit * rng.uniform(0.05, 0.08)))
        cv2.ellipse(canvas, center, axes, 0, 0, 360, (250, 250, 250), -1, lineType=cv2.LINE_AA)
        cv2.ellipse(canvas, center, axes, 0, 0, 360, (10, 10, 10), line, lineType=cv2.LINE_AA)
        font_scale = unit / 1400.0
        for row in (-1, 1):
            text = _random_text(rng, 6)
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, line)
            org = (center[0] - tw // 2, center[1] + row * int(th * 0.9) + th // 2)
            cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (10, 10, 10),
                        line, lineType=cv2.LINE_AA)
    # Cadre noir du panel.
    cv2.rectangle(canvas, (0, 0), (w - 1, h - 1), (0, 0, 0), line * 2)
    return canvas


# ---------------------------------------------------------------------------
# Vérité terrain
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FrameTruth:
    """Vérité terrain d'une frame.

    ``shot_id`` vaut ``None`` pour une frame de transition (mélange de plans) ;
    ``transform`` est alors ``None`` et ``blend`` liste les ``(plan, poids)``.
    """

    index: int
    time_s: float
    shot_id: int | None
    transform: SimilarityTransform | None
    pose_index: int | None
    is_duplicate: bool
    blend: tuple[tuple[int, float], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "time_s": self.time_s,
            "shot_id": self.shot_id,
            "transform": None if self.transform is None else self.transform.to_dict(),
            "pose_index": self.pose_index,
            "is_duplicate": self.is_duplicate,
            "blend": [[s, w] for s, w in self.blend],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FrameTruth:
        return cls(
            index=int(d["index"]),
            time_s=float(d["time_s"]),
            shot_id=None if d["shot_id"] is None else int(d["shot_id"]),
            transform=None if d["transform"] is None else SimilarityTransform.from_dict(d["transform"]),
            pose_index=None if d["pose_index"] is None else int(d["pose_index"]),
            is_duplicate=bool(d["is_duplicate"]),
            blend=tuple((int(s), float(w)) for s, w in d["blend"]),
        )


@dataclass(frozen=True)
class ShotTruth:
    shot_id: int
    panel_file: str
    panel_width: int
    panel_height: int
    start_idx: int
    end_idx: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "shot_id": self.shot_id,
            "panel_file": self.panel_file,
            "panel_width": self.panel_width,
            "panel_height": self.panel_height,
            "start_idx": self.start_idx,
            "end_idx": self.end_idx,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ShotTruth:
        return cls(**{k: d[k] for k in ("shot_id", "panel_file", "panel_width", "panel_height",
                                         "start_idx", "end_idx")})


@dataclass(frozen=True)
class GroundTruth:
    """Vérité terrain complète d'une vidéo synthétique."""

    name: str
    video_file: str
    screen_width: int
    screen_height: int
    fps: int
    variable_frame_rate: bool
    subtitle_box: tuple[int, int, int, int] | None  # x0, y0, x1, y1 (exclus), pixels écran
    shots: tuple[ShotTruth, ...]
    frames: tuple[FrameTruth, ...]
    root: Path = Path(".")

    @property
    def screen_size(self) -> tuple[int, int]:
        return self.screen_width, self.screen_height

    @property
    def video_path(self) -> Path:
        return self.root / self.video_file

    def panel_path(self, shot_id: int) -> Path:
        return self.root / self.shots[shot_id].panel_file

    def load_panel(self, shot_id: int) -> ImageU8:
        image = cv2.imread(str(self.panel_path(shot_id)), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(self.panel_path(shot_id))
        return np.asarray(image, dtype=np.uint8)

    def expected_sequences(self) -> list[Sequence]:
        """Séquences attendues : frames pures de chaque plan (transitions exclues)."""
        return [Sequence(s.start_idx, s.end_idx) for s in self.shots]

    def shot_frames(self, shot_id: int) -> list[FrameTruth]:
        return [f for f in self.frames if f.shot_id == shot_id]

    def transforms(self, shot_id: int) -> dict[int, SimilarityTransform]:
        """``index → (panel → écran)`` pour les frames pures du plan."""
        return {f.index: f.transform for f in self.shot_frames(shot_id) if f.transform is not None}

    def max_scale(self, shot_id: int) -> float:
        return max(t.scale for t in self.transforms(shot_id).values())

    def visibility_mask(self, frame: FrameTruth, erode_px: int = 0) -> MaskU8:
        """Pixels écran montrant **uniquement** le panel (hors bords anti-aliasés,
        hors sous-titre). Vide pour une frame de transition."""
        if frame.transform is None or frame.shot_id is None:
            return np.zeros((self.screen_height, self.screen_width), dtype=np.uint8)
        shot = self.shots[frame.shot_id]
        mask = full_coverage_mask(
            (shot.panel_width, shot.panel_height), frame.transform, self.screen_size, erode_px
        )
        if self.subtitle_box is not None:
            x0, y0, x1, y1 = self.subtitle_box
            mask[max(0, y0 - erode_px) : y1 + erode_px, max(0, x0 - erode_px) : x1 + erode_px] = 0
        return mask

    # ------------------------------------------------------------ sérialisation
    def to_dict(self) -> dict[str, Any]:
        return {
            "version": GROUND_TRUTH_VERSION,
            "name": self.name,
            "video_file": self.video_file,
            "screen_width": self.screen_width,
            "screen_height": self.screen_height,
            "fps": self.fps,
            "variable_frame_rate": self.variable_frame_rate,
            "subtitle_box": None if self.subtitle_box is None else list(self.subtitle_box),
            "shots": [s.to_dict() for s in self.shots],
            "frames": [f.to_dict() for f in self.frames],
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=1) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> GroundTruth:
        path = Path(path)
        d = json.loads(path.read_text(encoding="utf-8"))
        if d.get("version") != GROUND_TRUTH_VERSION:
            raise ValueError(f"Version de vérité terrain non supportée : {d.get('version')}")
        box = d["subtitle_box"]
        return cls(
            name=d["name"],
            video_file=d["video_file"],
            screen_width=int(d["screen_width"]),
            screen_height=int(d["screen_height"]),
            fps=int(d["fps"]),
            variable_frame_rate=bool(d["variable_frame_rate"]),
            subtitle_box=None if box is None else (int(box[0]), int(box[1]), int(box[2]), int(box[3])),
            shots=tuple(ShotTruth.from_dict(s) for s in d["shots"]),
            frames=tuple(FrameTruth.from_dict(f) for f in d["frames"]),
            root=path.parent,
        )


# ---------------------------------------------------------------------------
# Rendu
# ---------------------------------------------------------------------------


class _ShotRenderer:
    """Rend les frames d'un plan (panel + fond), sans état temporel."""

    def __init__(self, shot: ShotSpec, panel: ImageU8, spec: VideoSpec) -> None:
        self.shot = shot
        self.panel = panel
        self.spec = spec
        sw, sh = spec.screen_width, spec.screen_height
        ph, pw = panel.shape[:2]
        bg = spec.background
        self._cover_scale = max(sw / pw, sh / ph) * bg.cover_zoom
        self._bg_base: ImageU8 | None = None
        self._bg_static: ImageU8 | None = None
        if bg.mode != "solid":
            # Agrandissement flou précalculé à l'échelle de couverture.
            base = cv2.resize(panel, (max(1, round(pw * self._cover_scale)),
                                      max(1, round(ph * self._cover_scale))),
                              interpolation=cv2.INTER_AREA)
            if bg.blur_sigma > 0:
                base = cv2.GaussianBlur(base, (0, 0), bg.blur_sigma)
            self._bg_base = to_u8(base.astype(np.float32) * (1.0 - bg.darken))
            if bg.mode == "blur_static":
                center = Pose(self._cover_scale, (pw - 1) / 2.0, (ph - 1) / 2.0)
                self._bg_static = self._render_background(center)

    def _render_background(self, pose: Pose) -> ImageU8:
        assert self._bg_base is not None
        sw, sh = self.spec.screen_width, self.spec.screen_height
        # Le fond précalculé est à l'échelle cover_scale : pose relative en conséquence.
        rel = Pose(1.0, pose.center_u * self._cover_scale, pose.center_v * self._cover_scale)
        out = cv2.warpAffine(self._bg_base, rel.to_transform(sw, sh).matrix(), (sw, sh),
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return np.asarray(out, dtype=np.uint8)

    def background(self, pose: Pose) -> ImageU8:
        sw, sh = self.spec.screen_width, self.spec.screen_height
        bg = self.spec.background
        if bg.mode == "solid":
            return np.full((sh, sw, 3), bg.color, dtype=np.uint8)
        if bg.mode == "blur_static":
            assert self._bg_static is not None
            return self._bg_static
        return self._render_background(pose)

    def render(self, pose: Pose) -> tuple[F32, SimilarityTransform]:
        """Frame flottante (H, W, 3) et transformation exacte ``panel → écran``."""
        sw, sh = self.spec.screen_width, self.spec.screen_height
        ph, pw = self.panel.shape[:2]
        transform = pose.to_transform(sw, sh)
        fg = warp_similarity(self.panel, transform, (sw, sh)).astype(np.float32)
        alpha = coverage_alpha((pw, ph), transform, (sw, sh))[..., None]
        bg = self.background(pose).astype(np.float32)
        return fg * alpha + bg * (1.0 - alpha), transform


def subtitle_overlay(
    spec: SubtitleSpec, sw: int, sh: int
) -> tuple[F32, F32, tuple[int, int, int, int]]:
    """Calque RGB, opacité et boîte englobante d'un sous-titre (contour noir, texte blanc)."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = max(1, int(round(spec.font_scale * 2)))
    outline = thickness + 3
    (tw, th), base = cv2.getTextSize(spec.text, font, spec.font_scale, outline)
    x = (sw - tw) // 2
    y = int(sh * (1.0 - spec.bottom_margin)) - base
    layer = np.zeros((sh, sw, 3), dtype=np.uint8)
    alpha = np.zeros((sh, sw), dtype=np.uint8)
    cv2.putText(alpha, spec.text, (x, y), font, spec.font_scale, 255, outline, cv2.LINE_AA)
    cv2.putText(layer, spec.text, (x, y), font, spec.font_scale, (255, 255, 255), thickness,
                cv2.LINE_AA)
    ys, xs = np.nonzero(alpha)
    margin = 2
    box = (max(0, int(xs.min()) - margin), max(0, int(ys.min()) - margin),
           min(sw, int(xs.max()) + 1 + margin), min(sh, int(ys.max()) + 1 + margin))
    return layer.astype(np.float32), (alpha.astype(np.float32) / 255.0)[..., None], box


def _frame_times(spec: VideoSpec, n_frames: int) -> list[float]:
    """Instants de présentation ; en VFR, intervalles irréguliers de 1 à 3 périodes,
    arrondis à la milliseconde (base de temps de l'encodeur)."""
    if not spec.variable_frame_rate:
        return [i / spec.fps for i in range(n_frames)]
    rng = np.random.default_rng(spec.seed + 7919)
    steps = rng.choice([1.0, 1.0, 1.5, 2.0, 3.0], size=n_frames) / spec.fps
    steps[0] = 0.0
    times = np.cumsum(steps)
    out: list[float] = []
    for t in times:
        ms = round(float(t) * 1000)
        if out and ms <= round(out[-1] * 1000):
            ms = round(out[-1] * 1000) + 1
        out.append(ms / 1000.0)
    return out


def render_video(spec: VideoSpec) -> Iterator[tuple[ImageU8, FrameTruth]]:
    """Génère les frames BGR uint8 et leur vérité terrain, en flux (mémoire bornée).

    Les instants ``time_s`` sont ceux de :func:`_frame_times` (identiques à ceux
    qu'enregistre l'encodeur)."""
    panels = [make_panel(s.panel) for s in spec.shots]
    renderers = [_ShotRenderer(s, p, spec) for s, p in zip(spec.shots, panels)]
    sw, sh = spec.screen_width, spec.screen_height
    overlay = subtitle_overlay(spec.subtitle, sw, sh) if spec.subtitle is not None else None

    total = sum(s.n_frames for s in spec.shots) + spec.crossfade_frames * (len(spec.shots) - 1)
    times = _frame_times(spec, total)
    index = 0

    def finish(img: F32) -> ImageU8:
        if overlay is not None:
            layer, alpha, _ = overlay
            img = layer * alpha + img * (1.0 - alpha)
        return to_u8(img)

    for shot_id, (shot, renderer) in enumerate(zip(spec.shots, renderers)):
        if shot_id > 0 and spec.crossfade_frames > 0:
            prev_shot = spec.shots[shot_id - 1]
            img_a, _ = renderers[shot_id - 1].render(prev_shot.pose_at(prev_shot.n_poses - 1))
            img_b, _ = renderer.render(shot.pose_at(0))
            for j in range(spec.crossfade_frames):
                w_b = (j + 1) / (spec.crossfade_frames + 1)
                truth = FrameTruth(index, times[index], None, None, None, False,
                                   ((shot_id - 1, 1.0 - w_b), (shot_id, w_b)))
                yield finish(img_a * (1.0 - w_b) + img_b * w_b), truth
                index += 1
        for k in range(shot.n_poses):
            img, transform = renderer.render(shot.pose_at(k))
            frame = finish(img)
            for rep in range(shot.hold):
                truth = FrameTruth(index, times[index], shot_id, transform, k, rep > 0,
                                   ((shot_id, 1.0),))
                yield frame, truth
                index += 1


def generate_video(spec: VideoSpec, out_dir: Path) -> GroundTruth:
    """Encode la vidéo, enregistre les panels originaux (PNG) et la vérité terrain."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    video_file = f"{spec.name}{spec.container}"
    shots: list[ShotTruth] = []
    for shot_id, shot in enumerate(spec.shots):
        panel = make_panel(shot.panel)
        panel_file = f"{spec.name}_panel{shot_id}.png"
        if not cv2.imwrite(str(out_dir / panel_file), panel):
            raise OSError(f"Écriture impossible : {out_dir / panel_file}")
        shots.append(ShotTruth(shot_id, panel_file, panel.shape[1], panel.shape[0], -1, -1))

    frames: list[FrameTruth] = []
    with VideoEncoder(out_dir / video_file, spec.screen_width, spec.screen_height, spec.fps,
                      crf=spec.crf, variable_frame_rate=spec.variable_frame_rate) as encoder:
        for image, truth in render_video(spec):
            recorded = encoder.write(image, truth.time_s)
            if abs(recorded - truth.time_s) > 1e-6:
                truth = replace(truth, time_s=recorded)
            frames.append(truth)

    for i, shot_truth in enumerate(shots):
        indices = [f.index for f in frames if f.shot_id == i]
        shots[i] = replace(shot_truth, start_idx=min(indices), end_idx=max(indices))
    box = None
    if spec.subtitle is not None:
        box = subtitle_overlay(spec.subtitle, spec.screen_width, spec.screen_height)[2]
    gt = GroundTruth(
        name=spec.name,
        video_file=video_file,
        screen_width=spec.screen_width,
        screen_height=spec.screen_height,
        fps=spec.fps,
        variable_frame_rate=spec.variable_frame_rate,
        subtitle_box=box,
        shots=tuple(shots),
        frames=tuple(frames),
        root=out_dir,
    )
    gt.save(out_dir / f"{spec.name}_{GROUND_TRUTH_FILENAME}")
    logger.info("Vidéo synthétique %s : %d frames, %d plan(s)", video_file, len(frames), len(shots))
    return gt


# ---------------------------------------------------------------------------
# Scénarios de référence (cas limites de la spécification)
# ---------------------------------------------------------------------------

_PANEL_W, _PANEL_H = 1200, 900
_CU, _CV = (_PANEL_W - 1) / 2.0, (_PANEL_H - 1) / 2.0


def _shot(keys: list[tuple[float, Pose]], n: int = 30, easing: str = "linear", hold: int = 1,
          texture: str = "rich", seed: int = 0) -> ShotSpec:
    return ShotSpec(PanelSpec(_PANEL_W, _PANEL_H, texture, seed), tuple(keys), n, easing, hold)


def _scenario_static(seed: int) -> VideoSpec:
    # Panel entièrement visible : 1200x900 × 0.35 = 420x315 < 640x360.
    return VideoSpec("static", (_shot([(0.0, Pose(0.35, _CU, _CV))], n=12, seed=seed),), seed=seed)


def _scenario_pan_horizontal(seed: int) -> VideoSpec:
    keys = [(0.0, Pose(0.9, 380.0, _CV)), (1.0, Pose(0.9, 820.0, _CV))]
    return VideoSpec("pan_horizontal", (_shot(keys, seed=seed),), seed=seed)


def _scenario_pan_vertical(seed: int) -> VideoSpec:
    # Panel plus étroit que l'écran (fond visible sur les côtés), coupé en haut/bas.
    keys = [(0.0, Pose(0.5, _CU, 300.0)), (1.0, Pose(0.5, _CU, 600.0))]
    return VideoSpec("pan_vertical", (_shot(keys, seed=seed),), seed=seed)


def _scenario_zoom_in(seed: int) -> VideoSpec:
    keys = [(0.0, Pose(0.45, _CU, _CV)), (1.0, Pose(0.9, _CU + 60, _CV - 40))]
    return VideoSpec("zoom_in", (_shot(keys, seed=seed),), seed=seed)


def _scenario_zoom_out(seed: int) -> VideoSpec:
    keys = [(0.0, Pose(0.95, 420.0, 330.0)), (1.0, Pose(0.5, _CU, _CV))]
    return VideoSpec("zoom_out", (_shot(keys, seed=seed),), seed=seed)


def _scenario_pan_zoom_eased(seed: int) -> VideoSpec:
    keys = [(0.0, Pose(0.55, 400.0, 300.0)), (0.5, Pose(0.8, 650.0, 450.0)),
            (1.0, Pose(0.7, 800.0, 600.0))]
    return VideoSpec("pan_zoom_eased", (_shot(keys, n=40, easing="ease_in_out", seed=seed),),
                     background=BackgroundSpec(mode="blur_follow"), seed=seed)


def _scenario_duplicates(seed: int) -> VideoSpec:
    keys = [(0.0, Pose(0.8, 400.0, _CV)), (1.0, Pose(0.8, 800.0, _CV))]
    return VideoSpec("duplicates", (_shot(keys, n=15, hold=2, seed=seed),), seed=seed)


def _scenario_crossfade(seed: int) -> VideoSpec:
    a = _shot([(0.0, Pose(0.8, 400.0, _CV)), (1.0, Pose(0.8, 800.0, _CV))], n=20, seed=seed)
    b = _shot([(0.0, Pose(0.5, _CU, _CV)), (1.0, Pose(0.85, _CU, 350.0))], n=20,
              easing="sine", seed=seed + 1)
    return VideoSpec("crossfade", (a, b), crossfade_frames=6, seed=seed)


def _scenario_subtitles(seed: int) -> VideoSpec:
    keys = [(0.0, Pose(0.8, _CU, 250.0)), (1.0, Pose(0.8, _CU, 650.0))]
    return VideoSpec("subtitles", (_shot(keys, seed=seed),), subtitle=SubtitleSpec(), seed=seed)


def _scenario_flat(seed: int) -> VideoSpec:
    keys = [(0.0, Pose(0.8, 400.0, _CV)), (1.0, Pose(0.8, 800.0, _CV))]
    return VideoSpec("flat_texture", (_shot(keys, texture="flat", seed=seed),), seed=seed)


def _scenario_short(seed: int) -> VideoSpec:
    keys = [(0.0, Pose(0.7, 500.0, _CV)), (1.0, Pose(0.75, 700.0, _CV))]
    return VideoSpec("short", (_shot(keys, n=4, seed=seed),), seed=seed)


def _scenario_vfr(seed: int) -> VideoSpec:
    keys = [(0.0, Pose(0.6, 450.0, 350.0)), (1.0, Pose(0.85, 750.0, 550.0))]
    return VideoSpec("vfr", (_shot(keys, easing="ease_out", seed=seed),),
                     variable_frame_rate=True, container=".mkv", seed=seed)


SCENARIOS: Final[dict[str, Callable[[int], VideoSpec]]] = {
    "static": _scenario_static,
    "pan_horizontal": _scenario_pan_horizontal,
    "pan_vertical": _scenario_pan_vertical,
    "zoom_in": _scenario_zoom_in,
    "zoom_out": _scenario_zoom_out,
    "pan_zoom_eased": _scenario_pan_zoom_eased,
    "duplicates": _scenario_duplicates,
    "crossfade": _scenario_crossfade,
    "subtitles": _scenario_subtitles,
    "flat_texture": _scenario_flat,
    "short": _scenario_short,
    "vfr": _scenario_vfr,
}


def scenario(name: str, seed: int = 0) -> VideoSpec:
    try:
        return SCENARIOS[name](seed)
    except KeyError:
        raise ValueError(f"Scénario inconnu {name!r} ; disponibles : {sorted(SCENARIOS)}") from None
