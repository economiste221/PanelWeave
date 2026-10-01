"""Évaluation par rapport à une vérité terrain synthétique.

* :func:`pose_errors` compare des transformations estimées ``frame → canevas``
  aux transformations exactes ``panel → écran``. Le repère du canevas étant
  arbitraire (jauge), la similarité ``panel → canevas`` qui explique au mieux les
  estimations est d'abord ajustée par moindres carrés ; les erreurs résiduelles
  sont exprimées en **pixels écran**.
* :func:`masked_ssim` / :func:`evaluate_mosaic` mesurent la fidélité d'une
  reconstruction et la justesse de la couverture annoncée.
* :func:`oracle_mosaic` reconstruit un panel avec les poses exactes : c'est la
  borne supérieure de référence des phases suivantes (et la validation de la
  chaîne génération → encodage → décodage → vérité terrain).
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Final

import cv2
import numpy as np
from numpy.typing import NDArray
from skimage.metrics import structural_similarity

from panelrecon.core.geometry import corners, to_u8, warp_mask, warp_similarity
from panelrecon.core.models import (
    CropBox,
    ImageU8,
    MaskU8,
    MosaicResult,
    Sequence,
    SimilarityTransform,
)
from panelrecon.core.synthetic import FrameTruth, GroundTruth

# Rayon de la fenêtre gaussienne SSIM (sigma 1.5, troncature skimage à 3.5 sigma).
SSIM_WINDOW_RADIUS: Final[int] = 5


@dataclass(frozen=True)
class ReferenceThresholds:
    """Seuils d'acceptation des tests de référence (spécification)."""

    max_translation_px: float = 1.0
    max_scale_rel: float = 0.005
    min_ssim: float = 0.95
    min_true_coverage: float = 0.98
    max_coverage_overreach: float = 0.01


# ---------------------------------------------------------------------------
# Erreurs de pose
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FramePoseError:
    translation_px: float  # déplacement du centre de l'écran
    corner_px: float  # déplacement maximal des coins de l'écran
    scale_rel: float  # |s_est / s_vrai − 1|
    rotation_rad: float  # |θ_est − θ_vrai|


@dataclass(frozen=True)
class PoseErrors:
    per_frame: dict[int, FramePoseError]
    gauge: SimilarityTransform  # panel → canevas ajustée

    @property
    def max_translation_px(self) -> float:
        return max(e.translation_px for e in self.per_frame.values())

    @property
    def mean_translation_px(self) -> float:
        return float(np.mean([e.translation_px for e in self.per_frame.values()]))

    @property
    def max_corner_px(self) -> float:
        return max(e.corner_px for e in self.per_frame.values())

    @property
    def max_scale_rel(self) -> float:
        return max(e.scale_rel for e in self.per_frame.values())

    @property
    def max_rotation_rad(self) -> float:
        return max(e.rotation_rad for e in self.per_frame.values())


def _sample_points(width: int, height: int) -> NDArray[np.float64]:
    center = np.array([[(width - 1) / 2.0, (height - 1) / 2.0]])
    return np.vstack([corners(width, height), center])


def _residual_error(
    residual: SimilarityTransform, width: int, height: int
) -> FramePoseError:
    """Erreur d'une transformation ``écran → écran`` censée valoir l'identité."""
    pts = _sample_points(width, height)
    disp = np.linalg.norm(residual.apply(pts) - pts, axis=1)
    return FramePoseError(
        translation_px=float(disp[-1]),
        corner_px=float(disp[:-1].max()),
        scale_rel=abs(residual.scale - 1.0),
        rotation_rad=abs(math.remainder(residual.theta, 2 * math.pi)),
    )


def fit_gauge(
    estimated: Mapping[int, SimilarityTransform],
    truth: Mapping[int, SimilarityTransform],
    screen_size: tuple[int, int],
) -> SimilarityTransform:
    """Similarité ``panel → canevas`` expliquant au mieux les estimations."""
    common = sorted(set(estimated) & set(truth))
    if not common:
        raise ValueError("Aucune frame commune entre estimations et vérité terrain")
    pts = _sample_points(*screen_size)
    panel_pts = np.vstack([truth[i].inverse().apply(pts) for i in common])
    canvas_pts = np.vstack([estimated[i].apply(pts) for i in common])
    return SimilarityTransform.fit(panel_pts, canvas_pts)


def pose_errors(
    estimated: Mapping[int, SimilarityTransform],
    truth: Mapping[int, SimilarityTransform],
    screen_size: tuple[int, int],
) -> PoseErrors:
    """Erreurs de poses ``frame → canevas`` estimées, après ajustement de la jauge.

    ``truth[i]`` est la transformation exacte ``panel → écran`` de la frame ``i``.
    """
    gauge = fit_gauge(estimated, truth, screen_size)
    inv_gauge = gauge.inverse()
    per_frame: dict[int, FramePoseError] = {}
    for i in sorted(set(estimated) & set(truth)):
        residual = truth[i] @ inv_gauge @ estimated[i]  # écran → écran
        per_frame[i] = _residual_error(residual, *screen_size)
    return PoseErrors(per_frame=per_frame, gauge=gauge)


def pairwise_error(
    estimate: SimilarityTransform,
    truth_src: SimilarityTransform,
    truth_dst: SimilarityTransform,
    screen_size: tuple[int, int],
) -> FramePoseError:
    """Erreur d'un mouvement estimé ``frame src → frame dst`` (pixels de dst)."""
    true_motion = truth_dst @ truth_src.inverse()
    residual = estimate @ true_motion.inverse()  # dst → dst
    return _residual_error(residual, *screen_size)


# ---------------------------------------------------------------------------
# Fidélité de reconstruction
# ---------------------------------------------------------------------------


def masked_ssim(a: ImageU8, b: ImageU8, mask: MaskU8, min_pixels: int = 100) -> float:
    """SSIM moyen (gaussien, σ = 1.5) sur ``mask`` érodé du rayon de la fenêtre.

    L'érosion garantit qu'aucune fenêtre ne déborde hors de la zone évaluée.
    """
    if a.shape != b.shape or a.shape[:2] != mask.shape:
        raise ValueError(f"Formes incompatibles : {a.shape}, {b.shape}, {mask.shape}")
    k = 2 * SSIM_WINDOW_RADIUS + 1
    inner = cv2.erode(
        (mask > 0).astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)),
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)
    if int(inner.sum()) < min_pixels:
        raise ValueError(f"Zone d'évaluation trop petite ({int(inner.sum())} px)")
    channel_axis = 2 if a.ndim == 3 else None
    _, ssim_map = structural_similarity(  # type: ignore[no-untyped-call]
        a, b, data_range=255, channel_axis=channel_axis, gaussian_weights=True,
        sigma=1.5, use_sample_covariance=False, full=True,
    )
    if ssim_map.ndim == 3:
        ssim_map = ssim_map.mean(axis=2)
    return float(ssim_map[inner].mean())


def observable_mask(
    gt: GroundTruth,
    shot_id: int,
    panel_to_canvas: SimilarityTransform,
    canvas_size: tuple[int, int],
    erode_px: int = 0,
) -> MaskU8:
    """Union, dans le canevas, des zones du panel réellement visibles dans la vidéo."""
    w, h = canvas_size
    union = np.zeros((h, w), dtype=np.uint8)
    for frame in gt.shot_frames(shot_id):
        if frame.transform is None:
            continue
        visible = gt.visibility_mask(frame, erode_px)
        union |= warp_mask(visible, panel_to_canvas @ frame.transform.inverse(), canvas_size)
    return union


@dataclass(frozen=True)
class ReconstructionMetrics:
    ssim: float
    psnr_db: float
    true_coverage: float  # part de la zone observable effectivement reconstruite
    coverage_overreach: float  # part des pixels annoncés couverts hors zone observable
    reported_covered_px: int
    observable_px: int
    pose: PoseErrors

    def check(self, thresholds: ReferenceThresholds) -> list[str]:
        """Liste des seuils non respectés (vide si tout est conforme)."""
        failures: list[str] = []
        if self.pose.max_translation_px >= thresholds.max_translation_px:
            failures.append(f"translation {self.pose.max_translation_px:.3f} px")
        if self.pose.max_scale_rel >= thresholds.max_scale_rel:
            failures.append(f"échelle {100 * self.pose.max_scale_rel:.3f} %")
        if self.ssim <= thresholds.min_ssim:
            failures.append(f"SSIM {self.ssim:.4f}")
        if self.true_coverage < thresholds.min_true_coverage:
            failures.append(f"couverture {self.true_coverage:.4f}")
        if self.coverage_overreach > thresholds.max_coverage_overreach:
            failures.append(f"couverture surestimée {self.coverage_overreach:.4f}")
        return failures


def evaluate_mosaic(
    result: MosaicResult, gt: GroundTruth, shot_id: int, panel: ImageU8 | None = None
) -> ReconstructionMetrics:
    """Compare une mosaïque reconstruite au panel original de la vérité terrain."""
    if panel is None:
        panel = gt.load_panel(shot_id)
    truth = gt.transforms(shot_id)
    errors = pose_errors(result.transforms, truth, gt.screen_size)
    crop_shift = SimilarityTransform.from_translation(-result.crop.x0, -result.crop.y0)
    panel_to_crop = crop_shift @ errors.gauge
    size = (result.crop.width, result.crop.height)
    reference = np.asarray(warp_similarity(panel, panel_to_crop, size), dtype=np.uint8)
    observable = observable_mask(gt, shot_id, panel_to_crop, size)
    covered = result.image_bgra[..., 3] > 0
    obs = observable > 0
    both = (covered & obs).astype(np.uint8) * 255
    recon_bgr = np.ascontiguousarray(result.image_bgra[..., :3])
    ssim = masked_ssim(recon_bgr, reference, both)
    diff = recon_bgr[both > 0].astype(np.float64) - reference[both > 0].astype(np.float64)
    mse = float(np.mean(diff**2))
    psnr = float("inf") if mse == 0 else 10.0 * math.log10(255.0**2 / mse)
    n_cov = int(covered.sum())
    n_obs = int(obs.sum())
    return ReconstructionMetrics(
        ssim=ssim,
        psnr_db=psnr,
        true_coverage=float((covered & obs).sum()) / max(1, n_obs),
        coverage_overreach=float((covered & ~obs).sum()) / max(1, n_cov),
        reported_covered_px=n_cov,
        observable_px=n_obs,
        pose=errors,
    )


# ---------------------------------------------------------------------------
# Reconstruction oracle (poses exactes)
# ---------------------------------------------------------------------------


def nan_median(stack: NDArray[np.float32]) -> NDArray[np.float32]:
    """Médiane selon l'axe 0 en ignorant les NaN (NaN là où rien n'est observé).

    Tri vectorisé (les NaN sont rangés en fin) puis sélection des rangs médians
    selon le nombre d'observations de chaque pixel : bien plus rapide que
    ``np.nanmedian`` et sans avertissement sur les colonnes vides.
    """
    counts = np.sum(~np.isnan(stack), axis=0)
    ordered = np.sort(stack, axis=0)
    lo = np.clip((counts - 1) // 2, 0, None)[None]
    hi = np.clip(counts // 2, 0, stack.shape[0] - 1)[None]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        median = 0.5 * (
            np.take_along_axis(ordered, lo, axis=0)[0] + np.take_along_axis(ordered, hi, axis=0)[0]
        )
    return np.where(counts > 0, median, np.nan).astype(np.float32)


def oracle_mosaic(
    frames: Iterable[tuple[int, ImageU8]],
    gt: GroundTruth,
    shot_id: int,
    erode_px: int = 2,
    max_stack_bytes: int = 1 << 30,
) -> MosaicResult:
    """Fusion médiane des frames d'un plan recalées avec les poses **exactes**.

    Le canevas est à l'échelle maximale observée (aucune perte de résolution).
    ``frames`` fournit des couples ``(index, image BGR)`` ; seules les frames pures
    du plan sont utilisées. Outil d'évaluation : la pile des observations est
    gardée en mémoire, bornée par ``max_stack_bytes``.
    """
    shot = gt.shots[shot_id]
    truths: dict[int, FrameTruth] = {f.index: f for f in gt.shot_frames(shot_id)}
    s_max = gt.max_scale(shot_id)
    panel_to_canvas = SimilarityTransform(scale=s_max)
    cw = int(math.ceil(shot.panel_width * s_max))
    ch = int(math.ceil(shot.panel_height * s_max))
    n_max = len(truths)
    needed = n_max * cw * ch * 3 * 4
    if needed > max_stack_bytes:
        raise MemoryError(
            f"Pile oracle de {needed / 2**20:.0f} Mio > limite {max_stack_bytes / 2**20:.0f} Mio"
        )
    stack = np.full((n_max, ch, cw, 3), np.nan, dtype=np.float32)
    transforms: dict[int, SimilarityTransform] = {}
    k = 0
    for index, image in frames:
        truth = truths.get(index)
        if truth is None or truth.transform is None:
            continue
        frame_to_canvas = panel_to_canvas @ truth.transform.inverse()
        warped = warp_similarity(image, frame_to_canvas, (cw, ch)).astype(np.float32)
        valid = warp_mask(gt.visibility_mask(truth, erode_px), frame_to_canvas, (cw, ch)) > 0
        layer = stack[k]
        layer[valid] = warped[valid]
        transforms[index] = frame_to_canvas
        k += 1
    if k == 0:
        raise ValueError(f"Aucune frame du plan {shot_id} fournie")
    observations = stack[:k]
    coverage = np.sum(~np.isnan(observations[..., 0]), axis=0).astype(np.uint16)
    fused = nan_median(observations)
    covered = coverage > 0
    ys, xs = np.nonzero(covered)
    crop = CropBox(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    rgba = np.zeros((ch, cw, 4), dtype=np.uint8)
    rgba[..., :3] = to_u8(np.nan_to_num(fused, nan=0.0))
    rgba[..., 3] = np.where(covered, 255, 0).astype(np.uint8)
    sl = (slice(crop.y0, crop.y1), slice(crop.x0, crop.x1))
    indices = sorted(transforms)
    return MosaicResult(
        sequence=Sequence(min(indices), max(indices)),
        image_bgra=np.ascontiguousarray(rgba[sl]),
        coverage=np.ascontiguousarray(coverage[sl]),
        transforms=transforms,
        crop=crop,
        canvas_scale=s_max,
    )
