"""Contrôle qualité automatique d'une séquence reconstruite.

* **Accord frame / panel** : le panel reconstruit est reprojeté dans chaque frame
  (transformation inverse du recalage) et comparé à la frame par SSIM, sur la
  zone entièrement couverte par le panel. Une frame mal recalée, ou floue (flou
  de mouvement), obtient un SSIM faible.
* **Métriques** (spécification §7) : taux d'inliers moyen et minimum, erreur de
  reprojection RMS, ratio de couverture du rectangle recadré, SSIM moyen et
  minimum, netteté (variance du Laplacien du panel).
* **Verdict** : OK / À VÉRIFIER / ÉCHEC selon les seuils de ``QualityConfig``,
  avec les raisons.

Le calcul se fait à une résolution réduite (``eval_long_side``) : il ne coûte
qu'une fraction de la fusion.
"""

from __future__ import annotations

import logging
from collections.abc import Collection, Mapping

import cv2
import numpy as np
from skimage.metrics import structural_similarity

from panelrecon.core.config import PipelineConfig, QualityConfig
from panelrecon.core.geometry import warp_similarity
from panelrecon.core.models import (
    ImageU8,
    MosaicResult,
    QualityReport,
    SimilarityTransform,
    Verdict,
)
from panelrecon.core.registration import RegistrationResult

logger = logging.getLogger(__name__)

_FULL_ALPHA = 254.5


def evaluation_sample(indices: Collection[int], limit: int) -> list[int]:
    """Au plus ``limit`` indices bien répartis (extrémités comprises)."""
    ordered = sorted(indices)
    if len(ordered) <= limit:
        return ordered
    picks = np.linspace(0, len(ordered) - 1, limit)
    return sorted({ordered[int(round(p))] for p in picks})


def _native_to_scaled(factor: float) -> SimilarityTransform:
    """Application ``natif → réduit`` réalisée par ``cv2.resize`` (centres de pixels alignés)."""
    return SimilarityTransform(scale=factor, tx=0.5 * factor - 0.5, ty=0.5 * factor - 0.5)


def _to_eval_size(gray: ImageU8, factor: float, long_side: int) -> tuple[ImageU8, float]:
    """Réduit ``gray`` (image à ``factor`` de la taille native) à ``long_side`` au plus."""
    h, w = gray.shape[:2]
    if max(h, w) <= long_side:
        return gray, factor
    ratio = long_side / float(max(h, w))
    size = (max(1, round(w * ratio)), max(1, round(h * ratio)))
    small = np.asarray(cv2.resize(gray, size, interpolation=cv2.INTER_AREA), dtype=np.uint8)
    return small, factor * size[0] / float(w)


def frame_agreement(
    gray: ImageU8,
    factor: float,
    frame_to_canvas: SimilarityTransform,
    result: MosaicResult,
    cfg: QualityConfig,
) -> float | None:
    """SSIM entre une frame (niveaux de gris, réduite d'un facteur ``factor``) et le
    panel reprojeté dans cette frame ; ``None`` si le recouvrement est trop faible.

    ``frame_to_canvas`` envoie les coordonnées natives de la frame dans le canevas
    **avant** recadrage (comme ``MosaicResult.transforms``).
    """
    small, f = _to_eval_size(gray, factor, cfg.eval_long_side)
    h, w = small.shape[:2]
    crop = SimilarityTransform.from_translation(-float(result.crop.x0), -float(result.crop.y0))
    panel_to_frame = _native_to_scaled(f) @ (crop @ frame_to_canvas).inverse()
    bgra = result.image_bgra
    panel_gray = np.asarray(cv2.cvtColor(np.ascontiguousarray(bgra[..., :3]),
                                         cv2.COLOR_BGR2GRAY), dtype=np.float32)
    alpha = bgra[..., 3].astype(np.float32)
    # Couleur prémultipliée : l'interpolation ne mélange pas les pixels non observés.
    rendered = warp_similarity(panel_gray * (alpha / 255.0), panel_to_frame, (w, h),
                               interpolation=cv2.INTER_LINEAR)
    weight = warp_similarity(alpha, panel_to_frame, (w, h), interpolation=cv2.INTER_LINEAR)
    valid = weight >= _FULL_ALPHA
    if cfg.eval_erode_px > 0:
        size = 2 * cfg.eval_erode_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
        valid = cv2.erode(valid.astype(np.uint8), kernel, borderType=cv2.BORDER_CONSTANT,
                          borderValue=0) > 0
    if valid.mean() < max(cfg.min_eval_overlap, 1e-9):
        return None
    reference = np.clip(rendered * (255.0 / np.maximum(weight, 1e-3)), 0.0, 255.0)
    _, ssim_map = structural_similarity(  # type: ignore[no-untyped-call]
        small.astype(np.float32), reference.astype(np.float32), data_range=255.0, full=True,
        gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
    )
    return float(np.asarray(ssim_map)[valid].mean())


def sharpness(result: MosaicResult) -> float:
    """Variance du Laplacien du panel sur sa zone observée (bords exclus)."""
    gray = cv2.cvtColor(np.ascontiguousarray(result.image_bgra[..., :3]), cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    observed = cv2.erode((result.image_bgra[..., 3] == 255).astype(np.uint8),
                         np.ones((5, 5), np.uint8), borderType=cv2.BORDER_CONSTANT,
                         borderValue=0) > 0
    values = np.asarray(lap)[observed]
    return float(values.var()) if values.size else 0.0


def coverage_ratio(result: MosaicResult) -> float:
    return float((result.image_bgra[..., 3] > 0).mean())


def assess(
    result: MosaicResult,
    registration: RegistrationResult,
    agreements: Mapping[int, float | None],
    config: PipelineConfig,
    excluded: Collection[int] = (),
) -> QualityReport:
    """Métriques et verdict d'une séquence.

    ``agreements`` : SSIM frame / panel des frames évaluées (``None`` = recouvrement
    insuffisant) ; ``excluded`` : frames écartées par le contrôle qualité (leur
    SSIM n'entre pas dans les moyennes).
    """
    cfg = config.quality
    kept = {i: v for i, v in agreements.items() if v is not None and i not in excluded}
    values = list(kept.values())
    mean_ssim = float(np.mean(values)) if values else 0.0
    min_ssim = float(min(values)) if values else None
    cover = coverage_ratio(result)
    sharp = sharpness(result)
    rms = registration.rms_reprojection_error
    failures: list[str] = []
    reviews: list[str] = []
    if not values:
        failures.append("aucune frame comparable au panel reconstruit")
    elif mean_ssim < cfg.fail_min_mean_ssim:
        failures.append(f"accord frame/panel très faible (SSIM moyen {mean_ssim:.3f})")
    elif mean_ssim < cfg.ok_min_mean_ssim:
        reviews.append(f"accord frame/panel faible (SSIM moyen {mean_ssim:.3f})")
    if min_ssim is not None and min_ssim < cfg.ok_min_frame_ssim:
        worst = min(kept, key=lambda i: kept[i])
        reviews.append(f"frame {worst} mal reprojetée (SSIM {min_ssim:.3f})")
    if cover < cfg.fail_min_coverage_ratio:
        failures.append(f"couverture très incomplète ({cover:.1%} du rectangle)")
    elif cover < cfg.ok_min_coverage_ratio:
        reviews.append(f"couverture incomplète ({cover:.1%} du rectangle)")
    if rms > cfg.ok_max_rms_px:
        reviews.append(f"erreur de reprojection élevée ({rms:.2f} px)")
    if sharp < cfg.ok_min_sharpness:
        reviews.append(f"panel peu net (variance du Laplacien {sharp:.1f})")
    if registration.interrupted:
        reviews.append(f"recalage interrompu ({registration.interruption_reason})")
    verdict = Verdict.FAILED if failures else Verdict.TO_REVIEW if reviews else Verdict.OK
    report = QualityReport(
        sequence=result.sequence,
        n_frames_used=len(result.transforms),
        mean_inlier_ratio=registration.mean_inlier_ratio,
        min_inlier_ratio=registration.min_inlier_ratio,
        rms_reprojection_error=rms,
        coverage_ratio=cover,
        mean_ssim=mean_ssim,
        sharpness=sharp,
        verdict=verdict,
        reasons=tuple(failures + reviews),
        min_ssim=min_ssim,
        frames_evaluated=len(values),
        excluded_frames=tuple(sorted(excluded)),
        frame_ssim=tuple(sorted((i, v) for i, v in agreements.items() if v is not None)),
    )
    logger.info("Séquence %d–%d : %s (SSIM moyen %.3f, couverture %.1f%%)%s",
                result.sequence.start_idx, result.sequence.end_idx, verdict.value, mean_ssim,
                100.0 * cover, "" if not report.reasons else " — " + "; ".join(report.reasons))
    return report
