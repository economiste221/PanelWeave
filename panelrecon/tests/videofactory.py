"""Écriture de vidéos de test avec PyAV (cadence fixe ou variable)."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from fractions import Fraction
from pathlib import Path

import av
import numpy as np

from panelrecon.core.models import ImageU8

_CODECS: dict[str, tuple[str, dict[str, str]]] = {
    ".mp4": ("libx264", {"crf": "12", "preset": "veryfast"}),
    ".mov": ("libx264", {"crf": "12", "preset": "veryfast"}),
    ".mkv": ("libx264", {"crf": "12", "preset": "veryfast"}),
    ".webm": ("libvpx-vp9", {"crf": "12", "b": "0", "deadline": "realtime", "cpu-used": "8"}),
}


def write_video(
    path: Path,
    frames: Iterable[ImageU8],
    fps: int = 25,
    times_s: Sequence[float] | None = None,
) -> Path:
    """Encode ``frames`` (BGR uint8, dimensions paires) dans ``path``.

    Si ``times_s`` est fourni, chaque frame reçoit ce timestamp (à la milliseconde) :
    cela produit une vidéo à fréquence variable.
    """
    codec, options = _CODECS[path.suffix.lower()]
    frame_list = list(frames)
    if not frame_list:
        raise ValueError("Aucune frame à écrire")
    h, w = frame_list[0].shape[:2]
    if h % 2 or w % 2:
        raise ValueError("Les dimensions doivent être paires (yuv420p)")
    if times_s is not None and len(times_s) != len(frame_list):
        raise ValueError("times_s doit avoir autant d'éléments que frames")
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream(codec, rate=fps, options=options)
        assert isinstance(stream, av.VideoStream)
        stream.width = w
        stream.height = h
        stream.pix_fmt = "yuv420p"
        if times_s is not None:
            stream.codec_context.time_base = Fraction(1, 1000)
        for i, image in enumerate(frame_list):
            frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(image), format="bgr24")
            if times_s is not None:
                frame.pts = int(round(times_s[i] * 1000))
                frame.time_base = Fraction(1, 1000)
            else:
                frame.pts = i
                frame.time_base = Fraction(1, fps)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


def index_frame(index: int, width: int = 320, height: int = 240) -> ImageU8:
    """Frame dont la luminance moyenne encode ``index`` (pas de 9 niveaux) et
    dont le coin haut-gauche porte un repère jaune (pour tester l'orientation)."""
    level = 20 + 9 * index
    if level > 235:
        raise ValueError("index trop grand pour être encodé dans la luminance")
    image = np.full((height, width, 3), level, dtype=np.uint8)
    image[: height // 4, : width // 4] = 255
    image[: height // 4, : width // 4, 0] = 0
    return image


def decoded_index(image: ImageU8) -> int:
    """Inverse de :func:`index_frame` (robuste à la compression)."""
    h, w = image.shape[:2]
    region = image[h // 2 :, w // 2 :]
    return int(round((float(region.mean()) - 20.0) / 9.0))
