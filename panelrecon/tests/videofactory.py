"""Écriture de petites vidéos de test (cadence fixe ou variable)."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

from panelrecon.core.models import ImageU8
from panelrecon.core.video_io import VideoEncoder


def write_video(
    path: Path,
    frames: Iterable[ImageU8],
    fps: int = 25,
    times_s: Sequence[float] | None = None,
    crf: int = 12,
) -> Path:
    """Encode ``frames`` (BGR uint8, dimensions paires) dans ``path``.

    Si ``times_s`` est fourni, chaque frame reçoit ce timestamp (à la milliseconde) :
    cela produit une vidéo à fréquence variable.
    """
    frame_list = list(frames)
    if not frame_list:
        raise ValueError("Aucune frame à écrire")
    if times_s is not None and len(times_s) != len(frame_list):
        raise ValueError("times_s doit avoir autant d'éléments que frames")
    h, w = frame_list[0].shape[:2]
    with VideoEncoder(path, w, h, fps, crf=crf, variable_frame_rate=times_s is not None) as enc:
        for i, image in enumerate(frame_list):
            enc.write(image, None if times_s is None else times_s[i])
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
