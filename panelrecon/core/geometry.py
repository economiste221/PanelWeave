"""Opérations géométriques partagées : warp anti-repliement, masques de validité, coins.

Les warps respectent la convention de :mod:`panelrecon.core.models` : une
:class:`SimilarityTransform` ``T`` envoie les coordonnées de l'image source vers
celles de l'image de sortie (``dst(T(p)) = src(p)``).
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from panelrecon.core.models import FloatArray, ImageU8, MaskU8, SimilarityTransform

# En dessous de ce facteur de réduction, l'image est d'abord réduite par INTER_AREA
# (cv2.warpAffine n'intègre pas de filtre anti-repliement).
_PRESCALE_THRESHOLD = 0.999
# Seuil d'opacité au-delà duquel un pixel est considéré entièrement couvert.
FULL_COVERAGE_ALPHA = 0.999


def corners(width: int, height: int) -> FloatArray:
    """Coins des centres de pixels extrêmes, sens horaire depuis le haut-gauche."""
    return np.array(
        [[0.0, 0.0], [width - 1.0, 0.0], [width - 1.0, height - 1.0], [0.0, height - 1.0]],
        dtype=np.float64,
    )


def _resize_affine(src_w: int, src_h: int, dst_w: int, dst_h: int) -> FloatArray:
    """Matrice 3x3 de l'application ``src → dst`` réalisée par ``cv2.resize``.

    ``cv2.resize`` aligne les centres de pixels : ``x_s = (x_d + 0.5)/sx − 0.5``.
    """
    sx = dst_w / float(src_w)
    sy = dst_h / float(src_h)
    return np.array(
        [[sx, 0.0, 0.5 * sx - 0.5], [0.0, sy, 0.5 * sy - 0.5], [0.0, 0.0, 1.0]], dtype=np.float64
    )


def warp_similarity(
    image: NDArray[Any],
    transform: SimilarityTransform,
    out_size: tuple[int, int],
    interpolation: int = cv2.INTER_LANCZOS4,
    border_value: float = 0.0,
) -> NDArray[Any]:
    """Warp de ``image`` par ``transform`` dans une sortie ``(largeur, hauteur)``.

    Si la transformation réduit l'image, une pré-réduction INTER_AREA est
    appliquée puis composée **exactement** avec le warp résiduel, ce qui évite le
    repliement de spectre sans biaiser la géométrie.
    """
    out_w, out_h = out_size
    if out_w <= 0 or out_h <= 0:
        raise ValueError(f"Taille de sortie invalide : {out_size}")
    src = image
    matrix = transform.homogeneous()
    if transform.scale < _PRESCALE_THRESHOLD:
        h, w = image.shape[:2]
        new_w = max(1, int(round(w * transform.scale)))
        new_h = max(1, int(round(h * transform.scale)))
        src = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        matrix = matrix @ np.linalg.inv(_resize_affine(w, h, new_w, new_h))
    warped = cv2.warpAffine(
        src,
        matrix[:2],
        (out_w, out_h),
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )
    return np.asarray(warped)


def coverage_alpha(
    src_size: tuple[int, int], transform: SimilarityTransform, out_size: tuple[int, int]
) -> NDArray[np.float32]:
    """Opacité ∈ [0, 1] de l'image source ``(largeur, hauteur)`` une fois warpée."""
    w, h = src_size
    ones = np.ones((h, w), dtype=np.float32)
    # Bord explicite à 0 autour de la source : l'interpolation produit alors une
    # rampe anti-aliasée sur la frontière au lieu d'un bord dur.
    padded = cv2.copyMakeBorder(ones, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0.0)
    shifted = transform @ SimilarityTransform.from_translation(-1.0, -1.0)
    alpha = warp_similarity(padded, shifted, out_size, interpolation=cv2.INTER_LINEAR)
    return np.clip(alpha.astype(np.float32), 0.0, 1.0)


def full_coverage_mask(
    src_size: tuple[int, int],
    transform: SimilarityTransform,
    out_size: tuple[int, int],
    erode_px: int = 0,
) -> MaskU8:
    """Masque 255 là où la sortie est **entièrement** couverte par la source warpée."""
    alpha = coverage_alpha(src_size, transform, out_size)
    mask = np.where(alpha >= FULL_COVERAGE_ALPHA, 255, 0).astype(np.uint8)
    if erode_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * erode_px + 1, 2 * erode_px + 1))
        mask = np.asarray(
            cv2.erode(mask, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0), dtype=np.uint8
        )
    return mask


def warp_mask(
    mask: MaskU8, transform: SimilarityTransform, out_size: tuple[int, int]
) -> MaskU8:
    """Warp d'un masque binaire : un pixel de sortie n'est valide que s'il provient
    entièrement de pixels valides (aucun mélange avec l'extérieur)."""
    as_float = (mask > 0).astype(np.float32)
    alpha = warp_similarity(as_float, transform, out_size, interpolation=cv2.INTER_LINEAR)
    return np.where(alpha >= FULL_COVERAGE_ALPHA, 255, 0).astype(np.uint8)


def to_u8(image: NDArray[Any]) -> ImageU8:
    """Arrondi saturé vers uint8."""
    return np.clip(np.rint(np.asarray(image, dtype=np.float64)), 0, 255).astype(np.uint8)
