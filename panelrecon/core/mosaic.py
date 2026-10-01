"""Mosaïque : warp des frames dans le repère canonique et fusion médiane pondérée.

Étapes, pour une séquence recalée :

1. **Canevas** : boîte englobante des zones valides de toutes les frames dans le
   repère canonique ; erreur explicite si elle dépasse la borne configurée.
2. **Observations** : chaque frame est warpée (Lanczos) sur sa seule emprise
   dans le canevas, avec son masque de validité (panel ∩ hors bords d'écran ∩
   hors zones d'exclusion, érodé du support du noyau) et un poids de bord
   (rampe sur la distance au bord du masque). Les observations sont empilées en
   uint8 (BGR + poids de bord quantifié, 0 = invalide), en mémoire ou sur disque
   (memmap) selon leur taille.
3. **Fusion par tuiles** : médiane pondérée vectorisée par canal ; poids =
   poids de bord × (1 / échelle frame→canevas)^p, qui favorise les frames les
   plus zoomées.
4. **Couverture et recadrage** : nombre d'observations valides par pixel ;
   alpha nul sous le seuil de couverture (aucun contenu inventé) ; recadrage sur
   la zone couverte.
"""

from __future__ import annotations

import logging
import math
import tempfile
from collections.abc import Callable, Iterable
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import cv2
import numpy as np
from numpy.typing import NDArray

from panelrecon.core.config import PipelineConfig
from panelrecon.core.geometry import FULL_COVERAGE_ALPHA, corners, to_u8
from panelrecon.core.models import (
    CancellationToken,
    CropBox,
    FrameObs,
    ImageU8,
    MaskU8,
    MosaicResult,
    Sequence,
    SimilarityTransform,
)
from panelrecon.core.registration import PanelMaskProvider, RegistrationResult
from panelrecon.core.video_io import build_exclusion_mask

logger = logging.getLogger(__name__)

_INTERPOLATIONS: Final[dict[str, int]] = {"lanczos4": cv2.INTER_LANCZOS4, "cubic": cv2.INTER_CUBIC}
_EDGE_LEVELS: Final[float] = 254.0  # poids de bord quantifié sur 1..255
ProgressCallback = Callable[[float, str], None]


class CanvasTooLargeError(MemoryError):
    """Le canevas demandé dépasse la borne ``mosaic.max_canvas_megapixels``."""


@dataclass(frozen=True)
class Canvas:
    """Canevas : ``offset`` envoie le repère canonique vers les pixels du canevas."""

    width: int
    height: int
    offset: SimilarityTransform

    @property
    def size(self) -> tuple[int, int]:
        return self.width, self.height


# ---------------------------------------------------------------------------
# Masques de validité et poids
# ---------------------------------------------------------------------------


def validity_mask(
    frame: FrameObs, config: PipelineConfig, panel_mask: MaskU8 | None = None
) -> MaskU8:
    """Pixels natifs exploitables : panel ∩ hors bords d'écran ∩ hors zones d'exclusion."""
    h, w = frame.height, frame.width
    cfg = config.mosaic
    mask = build_exclusion_mask(h, w, config.preprocess.exclusion_zones)
    b = cfg.frame_border_px
    if b > 0:
        mask[:b] = 0
        mask[h - b :] = 0
        mask[:, :b] = 0
        mask[:, w - b :] = 0
    if panel_mask is not None:
        if panel_mask.shape != (h, w):
            raise ValueError("Le masque du panel doit avoir la taille native de la frame")
        mask = np.where(panel_mask > 0, mask, 0).astype(np.uint8)
    if cfg.mask_erode_px > 0:
        k = 2 * cfg.mask_erode_px + 1
        mask = np.asarray(
            cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)),
                      borderType=cv2.BORDER_CONSTANT, borderValue=0),
            dtype=np.uint8,
        )
    return mask


def edge_weight(mask: MaskU8, feather_px: float, min_weight: float) -> NDArray[np.float32]:
    """Poids ∈ [min_weight, 1] croissant avec la distance au bord du masque ; 0 hors masque."""
    if feather_px <= 0:
        return (mask > 0).astype(np.float32)
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    dist = cv2.distanceTransform((padded > 0).astype(np.uint8), cv2.DIST_L2, 5)[1:-1, 1:-1]
    weight = np.clip(dist / feather_px, min_weight, 1.0).astype(np.float32)
    weight[mask == 0] = 0.0
    return weight


def scale_weight(frame_to_canvas: SimilarityTransform, power: float) -> float:
    """Poids d'échelle : un pixel de frame couvrant ``s`` pixels de canevas vaut ``s^-p``."""
    return float(frame_to_canvas.scale ** (-power))


# ---------------------------------------------------------------------------
# Canevas
# ---------------------------------------------------------------------------


def _mask_corners(mask: MaskU8) -> NDArray[np.float64]:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return np.zeros((0, 2))
    x0, x1, y0, y1 = float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max())
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])


def plan_canvas(
    transforms: dict[int, SimilarityTransform],
    regions: dict[int, NDArray[np.float64]],
    max_megapixels: float,
    clip: tuple[float, float, float, float] | None = None,
) -> Canvas:
    """Canevas englobant les régions (coins en coordonnées natives) de chaque frame,
    éventuellement restreint à ``clip`` = ``(x0, y0, x1, y1)`` dans le repère canonique
    (emprise du panel)."""
    pts = [transforms[i].apply(r) for i, r in regions.items() if len(r)]
    if not pts:
        raise ValueError("Aucune région valide à placer dans le canevas")
    allpts = np.vstack(pts)
    lo = allpts.min(axis=0)
    hi = allpts.max(axis=0)
    if clip is not None:
        lo = np.maximum(lo, np.array(clip[:2]))
        hi = np.minimum(hi, np.array(clip[2:]))
        if np.any(hi <= lo):
            raise ValueError("L'emprise du panel ne recoupe aucune frame")
    x_min, y_min = np.floor(lo) - 1.0
    x_max, y_max = np.ceil(hi) + 1.0
    width, height = int(x_max - x_min) + 1, int(y_max - y_min) + 1
    megapixels = width * height / 1e6
    if megapixels > max_megapixels:
        raise CanvasTooLargeError(
            f"Canevas de {width}x{height} ({megapixels:.1f} Mpx) > limite "
            f"mosaic.max_canvas_megapixels = {max_megapixels} Mpx. Réduire le zoom maximal "
            "pris en compte ou augmenter la limite."
        )
    return Canvas(width, height, SimilarityTransform.from_translation(-x_min, -y_min))


# ---------------------------------------------------------------------------
# Médiane pondérée
# ---------------------------------------------------------------------------


def weighted_median(
    values: NDArray[np.uint8], weights: NDArray[np.float32]
) -> NDArray[np.float32]:
    """Médiane pondérée selon l'axe 0, par canal.

    ``values`` : ``(n, h, w, c)`` ; ``weights`` : ``(n, h, w)`` (0 = observation
    invalide). Quand la demi-masse tombe exactement entre deux observations, la
    moyenne des deux est renvoyée (à poids égaux, c'est la médiane usuelle).
    Renvoie ``NaN`` là où le poids total est nul.
    """
    n, h, w, c = values.shape
    v = values.astype(np.float32)
    invalid = weights <= 0
    v[np.broadcast_to(invalid[..., None], v.shape)] = np.inf
    order = np.argsort(v, axis=0, kind="stable")
    sorted_v = np.take_along_axis(v, order, axis=0)
    sorted_w = np.take_along_axis(np.broadcast_to(weights[..., None], v.shape), order, axis=0)
    cumulative = np.cumsum(sorted_w, axis=0)
    total = cumulative[-1]
    half = 0.5 * total
    tol = 1e-6 * total
    lo = np.argmax(cumulative >= half - tol, axis=0)[None]
    hi = np.argmax(cumulative > half + tol, axis=0)[None]
    v_lo = np.take_along_axis(sorted_v, lo, axis=0)[0]
    v_hi = np.take_along_axis(sorted_v, hi, axis=0)[0]
    out = 0.5 * (v_lo + v_hi)
    return np.where(total > 0, out, np.nan).astype(np.float32)


# ---------------------------------------------------------------------------
# Recadrage
# ---------------------------------------------------------------------------


def crop_box(covered: NDArray[np.bool_], mode: str) -> CropBox:
    """``bbox`` : rectangle englobant ; ``covered`` : rognage glouton des côtés jusqu'à
    couverture totale (opérations vectorisées par ligne/colonne)."""
    ys, xs = np.nonzero(covered)
    if xs.size == 0:
        raise ValueError("Aucun pixel couvert")
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    if mode == "bbox":
        return CropBox(x0, y0, x1, y1)
    if mode != "covered":
        raise ValueError(f"Mode de recadrage inconnu : {mode!r}")
    while x1 > x0 and y1 > y0:
        region = covered[y0:y1, x0:x1]
        if region.all():
            break
        # Fraction de pixels non couverts de chaque bord ; on retire le pire.
        holes = {
            "top": 1.0 - region[0].mean(),
            "bottom": 1.0 - region[-1].mean(),
            "left": 1.0 - region[:, 0].mean(),
            "right": 1.0 - region[:, -1].mean(),
        }
        side = max(holes, key=lambda k: holes[k])
        if side == "top":
            y0 += 1
        elif side == "bottom":
            y1 -= 1
        elif side == "left":
            x0 += 1
        else:
            x1 -= 1
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Aucun rectangle entièrement couvert")
    return CropBox(x0, y0, x1, y1)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class _ObservationStack:
    """Pile ``(n, H, W, 4)`` uint8 : BGR + poids de bord (0 = invalide), en RAM ou memmap."""

    def __init__(self, n: int, canvas: Canvas, config: PipelineConfig, stack: ExitStack) -> None:
        shape = (n, canvas.height, canvas.width, 4)
        nbytes = math.prod(shape)
        if nbytes <= config.mosaic.in_memory_stack_mb * 2**20:
            self.data: NDArray[np.uint8] = np.zeros(shape, dtype=np.uint8)
            self.on_disk = False
        else:
            root = config.mosaic.temp_dir or None
            tmp = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="panelrecon_",
                                                                       dir=root)))
            self.data = np.memmap(tmp / "observations.u8", dtype=np.uint8, mode="w+", shape=shape)
            self.on_disk = True
            logger.info("Pile d'observations sur disque : %.0f Mio dans %s", nbytes / 2**20, tmp)


def build_mosaic(
    frames: Iterable[FrameObs],
    registration: RegistrationResult,
    config: PipelineConfig,
    panel_masks: PanelMaskProvider | None = None,
    cancel: CancellationToken | None = None,
    progress: ProgressCallback | None = None,
    clip: tuple[float, float, float, float] | None = None,
) -> MosaicResult:
    """Fusionne les frames recalées d'une séquence.

    ``frames`` est parcouru une fois (flux) ; seules les frames présentes dans
    ``registration.transforms`` sont utilisées. ``clip`` (repère canonique)
    restreint le canevas à l'emprise connue du panel.
    """
    cfg = config.mosaic
    transforms = registration.transforms
    if not transforms:
        raise ValueError("Aucune frame recalée")
    interpolation = _INTERPOLATIONS[cfg.interpolation]

    # Canevas : emprise des frames, restreinte à celle du panel quand elle est connue.
    regions = {i: corners(*registration.frame_sizes[i]) for i in transforms}
    canonical = plan_canvas(transforms, regions, cfg.max_canvas_megapixels, clip)
    to_canvas = {i: canonical.offset @ t for i, t in transforms.items()}
    order = sorted(transforms)
    slot = {idx: k for k, idx in enumerate(order)}
    scale_w = np.array([scale_weight(to_canvas[i], cfg.scale_weight_power) for i in order],
                       dtype=np.float32)
    scale_w /= scale_w.max()

    with ExitStack() as stack:
        obs = _ObservationStack(len(order), canonical, config, stack)
        seen: set[int] = set()
        for frame in frames:
            if cancel is not None:
                cancel.raise_if_cancelled()
            if frame.index not in slot:
                continue
            mask = validity_mask(frame, config, panel_masks(frame) if panel_masks else None)
            _warp_observation(frame, mask, to_canvas[frame.index], canonical, obs.data[slot[frame.index]],
                              interpolation, cfg.edge_feather_px, cfg.min_edge_weight)
            seen.add(frame.index)
            if progress is not None:
                progress(0.7 * len(seen) / len(order), f"warp frame {frame.index}")
        missing = set(order) - seen
        if missing:
            raise ValueError(f"Frames recalées absentes du flux : {sorted(missing)[:10]}")

        fused, coverage = _fuse_tiles(obs.data, scale_w, cfg.tile_size, cancel, progress)

    covered = coverage >= cfg.min_coverage
    if not covered.any():
        raise ValueError("Aucun pixel suffisamment couvert")
    crop = crop_box(covered, cfg.crop_mode)
    sl = (slice(crop.y0, crop.y1), slice(crop.x0, crop.x1))
    bgra = np.zeros((crop.height, crop.width, 4), dtype=np.uint8)
    region_cov = covered[sl]
    bgra[..., :3] = np.where(region_cov[..., None], to_u8(np.nan_to_num(fused[sl], nan=0.0)), 0)
    bgra[..., 3] = np.where(region_cov, 255, 0).astype(np.uint8)
    result = MosaicResult(
        sequence=Sequence(order[0], order[-1],
                          excluded=tuple(i for i in registration.excluded if order[0] <= i <= order[-1])),
        image_bgra=bgra,
        coverage=np.ascontiguousarray(coverage[sl]),
        transforms=to_canvas,
        crop=crop,
        canvas_scale=registration.canonical_scale,
    )
    logger.info(
        "Mosaïque %dx%d (canevas %dx%d), %d frames, couverture moyenne %.1f",
        crop.width, crop.height, canonical.width, canonical.height, len(order),
        float(coverage[sl][region_cov].mean()),
    )
    return result


def _warp_observation(
    frame: FrameObs,
    mask: MaskU8,
    frame_to_canvas: SimilarityTransform,
    canvas: Canvas,
    out: NDArray[np.uint8],
    interpolation: int,
    feather_px: float,
    min_weight: float,
) -> None:
    """Warp de la frame et de son poids de bord sur sa seule emprise dans le canevas."""
    region = _mask_corners(mask)
    if len(region) == 0:
        logger.warning("Frame %d : masque de validité vide", frame.index)
        return
    pts = frame_to_canvas.apply(region)
    x0 = max(0, int(math.floor(pts[:, 0].min())) - 2)
    y0 = max(0, int(math.floor(pts[:, 1].min())) - 2)
    x1 = min(canvas.width, int(math.ceil(pts[:, 0].max())) + 3)
    y1 = min(canvas.height, int(math.ceil(pts[:, 1].max())) + 3)
    if x1 <= x0 or y1 <= y0:
        return
    local = SimilarityTransform.from_translation(-x0, -y0) @ frame_to_canvas
    size = (x1 - x0, y1 - y0)
    matrix = local.matrix()
    image = cv2.warpAffine(frame.image, matrix, size, flags=interpolation,
                           borderMode=cv2.BORDER_REPLICATE)
    weight = edge_weight(mask, feather_px, min_weight)
    warped_w = cv2.warpAffine(weight, matrix, size, flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    support = cv2.warpAffine((mask > 0).astype(np.float32), matrix, size, flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    valid = support >= FULL_COVERAGE_ALPHA
    quantized = np.where(valid, np.clip(np.rint(warped_w * _EDGE_LEVELS), 1, 255), 0)
    target = out[y0:y1, x0:x1]
    target[..., :3] = np.where(valid[..., None], image, 0)
    target[..., 3] = quantized.astype(np.uint8)


def _fuse_tiles(
    stack: NDArray[np.uint8],
    scale_w: NDArray[np.float32],
    tile: int,
    cancel: CancellationToken | None,
    progress: ProgressCallback | None,
) -> tuple[NDArray[np.float32], NDArray[np.uint16]]:
    n, height, width, _ = stack.shape
    fused = np.full((height, width, 3), np.nan, dtype=np.float32)
    coverage = np.zeros((height, width), dtype=np.uint16)
    tiles = [(y, x) for y in range(0, height, tile) for x in range(0, width, tile)]
    for k, (y, x) in enumerate(tiles):
        if cancel is not None:
            cancel.raise_if_cancelled()
        block = np.asarray(stack[:, y : y + tile, x : x + tile])
        edge = block[..., 3]
        present = edge.reshape(n, -1).any(axis=1)
        if not present.any():
            continue
        block, edge = block[present], edge[present]
        weights = edge.astype(np.float32) / 255.0 * scale_w[present][:, None, None]
        fused[y : y + tile, x : x + tile] = weighted_median(block[..., :3], weights)
        coverage[y : y + tile, x : x + tile] = (edge > 0).sum(axis=0).astype(np.uint16)
        if progress is not None:
            progress(0.7 + 0.3 * (k + 1) / len(tiles), "fusion")
    return fused, coverage
