"""Tests de référence : métriques d'évaluation et reconstruction oracle (poses exactes).

La reconstruction oracle valide toute la chaîne synthétique (rendu, encodage,
décodage, vérité terrain, métriques) et fixe la borne supérieure que le pipeline
devra approcher : les seuils de la spécification (translation < 1 px, échelle
< 0,5 %, SSIM > 0,95, couverture juste) doivent y être respectés largement.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from panelrecon.core.config import PipelineConfig
from panelrecon.core.evaluation import (
    ReferenceThresholds,
    evaluate_mosaic,
    masked_ssim,
    nan_median,
    oracle_mosaic,
    pairwise_error,
    pose_errors,
)
from panelrecon.core.models import CropBox, MosaicResult, SimilarityTransform
from panelrecon.core.synthetic import GroundTruth
from panelrecon.core.video_io import VideoReader
from panelrecon.tests.conftest import SyntheticCache

SCREEN = (640, 360)


def _truth_poses(n: int = 12) -> dict[int, SimilarityTransform]:
    return {
        i: SimilarityTransform(0.5 + 0.03 * i, 0.0, -20.0 * i + 100.0, 5.0 * i - 50.0)
        for i in range(n)
    }


def _as_estimates(
    truth: dict[int, SimilarityTransform], gauge: SimilarityTransform
) -> dict[int, SimilarityTransform]:
    return {i: gauge @ t.inverse() for i, t in truth.items()}


# ------------------------------------------------------------------ poses


def test_fit_similarity_recovers_transform() -> None:
    rng = np.random.default_rng(0)
    truth = SimilarityTransform(1.7, 0.3, -12.0, 40.0)
    src = rng.uniform(-100, 100, (50, 2))
    est = SimilarityTransform.fit(src, truth.apply(src) + rng.normal(0, 1e-6, (50, 2)))
    np.testing.assert_allclose(est.matrix(), truth.matrix(), atol=1e-6)
    no_rot = SimilarityTransform.fit(src, SimilarityTransform(2.0, 0, 3, 4).apply(src),
                                     allow_rotation=False)
    assert no_rot.theta == 0.0 and no_rot.scale == pytest.approx(2.0)
    weights = np.ones(50)
    weights[0] = 0.0
    dst = truth.apply(src)
    dst[0] += 1000.0  # point aberrant de poids nul
    np.testing.assert_allclose(
        SimilarityTransform.fit(src, dst, weights).matrix(), truth.matrix(), atol=1e-8
    )
    with pytest.raises(ValueError):
        SimilarityTransform.fit(src[:1], src[:1])
    with pytest.raises(ValueError):
        SimilarityTransform.fit(np.zeros((5, 2)), src[:5])


def test_pose_errors_are_gauge_invariant() -> None:
    truth = _truth_poses()
    gauge = SimilarityTransform(2.3, 0.4, 500.0, -80.0)
    errors = pose_errors(_as_estimates(truth, gauge), truth, SCREEN)
    assert errors.max_translation_px < 1e-8 and errors.max_scale_rel < 1e-12
    assert errors.max_rotation_rad < 1e-12
    np.testing.assert_allclose(errors.gauge.matrix(), gauge.matrix(), atol=1e-6)


def test_pose_errors_detect_perturbations() -> None:
    truth = _truth_poses(20)
    estimates = _as_estimates(truth, SimilarityTransform(1.5, 0.0, 10.0, 10.0))
    # Erreur de 2 px écran sur la frame 7 (translation appliquée côté écran).
    estimates[7] = estimates[7] @ SimilarityTransform.from_translation(2.0, 0.0)
    errors = pose_errors(estimates, truth, SCREEN)
    assert errors.per_frame[7].translation_px == pytest.approx(2.0, rel=0.1)
    others = [e.translation_px for i, e in errors.per_frame.items() if i != 7]
    assert max(others) < 0.3
    # Erreur d'échelle de 1 % autour du centre de l'écran.
    estimates = _as_estimates(truth, SimilarityTransform())
    estimates[3] = estimates[3] @ SimilarityTransform.from_scale_about(1.01, 319.5, 179.5)
    errors = pose_errors(estimates, truth, SCREEN)
    assert errors.per_frame[3].scale_rel == pytest.approx(0.01, rel=0.1)
    assert errors.per_frame[3].corner_px > 2.0


def test_pairwise_error() -> None:
    truth = _truth_poses(2)
    true_motion = truth[1] @ truth[0].inverse()
    exact = pairwise_error(true_motion, truth[0], truth[1], SCREEN)
    assert exact.translation_px < 1e-9 and exact.scale_rel < 1e-12
    shifted = SimilarityTransform.from_translation(0.5, 0.0) @ true_motion
    err = pairwise_error(shifted, truth[0], truth[1], SCREEN)
    assert err.translation_px == pytest.approx(0.5) and err.corner_px == pytest.approx(0.5)


# ------------------------------------------------------------------- SSIM


def test_masked_ssim() -> None:
    rng = np.random.default_rng(0)
    raw = rng.integers(0, 256, (80, 100, 3), dtype=np.uint8)
    smooth = cv2.GaussianBlur(raw, (0, 0), 2.0).astype(np.float64)
    a = np.rint(255 * (smooth - smooth.min()) / np.ptp(smooth)).astype(np.uint8)
    mask = np.zeros((80, 100), np.uint8)
    mask[10:70, 10:90] = 255
    assert masked_ssim(a, a, mask) == pytest.approx(1.0)
    b = a.copy()
    b[mask == 0] = 0  # hors masque (au-delà du rayon de la fenêtre) : sans effet
    assert masked_ssim(a, b, mask) == pytest.approx(1.0)
    noisy = np.clip(a.astype(int) + rng.normal(0, 25, a.shape), 0, 255).astype(np.uint8)
    assert masked_ssim(a, noisy, mask) < 0.9
    with pytest.raises(ValueError):
        masked_ssim(a, a, np.zeros((80, 100), np.uint8))
    with pytest.raises(ValueError):
        masked_ssim(a, a[:, :50], mask)


def test_nan_median_matches_numpy() -> None:
    rng = np.random.default_rng(0)
    stack = rng.random((7, 20, 30, 3)).astype(np.float32)
    stack[rng.random(stack.shape) < 0.4] = np.nan
    stack[:, 0, 0] = np.nan
    with np.errstate(all="ignore"), pytest.warns(RuntimeWarning):
        expected = np.nanmedian(stack, axis=0)
    np.testing.assert_allclose(nan_median(stack), expected, equal_nan=True)


# ------------------------------------------------- reconstruction oracle


def _oracle(gt: GroundTruth, shot_id: int) -> MosaicResult:
    cfg = PipelineConfig()
    with VideoReader(gt.video_path, cfg.video, cfg.preprocess) as reader:
        frames = ((f.index, f.image) for f in reader.frames(compute_proxy=False))
        return oracle_mosaic(frames, gt, shot_id)


@pytest.mark.parametrize(
    ("name", "shot_id"),
    [
        ("static", 0),
        ("pan_horizontal", 0),
        ("pan_vertical", 0),
        ("zoom_in", 0),
        ("zoom_out", 0),
        ("pan_zoom_eased", 0),
        ("crossfade", 1),
        ("subtitles", 0),
        ("flat_texture", 0),
        ("short", 0),
        ("vfr", 0),
    ],
)
def test_oracle_reconstruction_meets_reference_thresholds(
    synthetic: SyntheticCache, name: str, shot_id: int
) -> None:
    gt = synthetic.get(name)
    result = _oracle(gt, shot_id)
    metrics = evaluate_mosaic(result, gt, shot_id)
    assert metrics.check(ReferenceThresholds()) == []
    assert metrics.pose.max_translation_px < 1e-6  # poses exactes
    assert metrics.true_coverage > 0.99 and metrics.coverage_overreach == 0.0
    # Canevas à l'échelle maximale observée : aucune perte de résolution.
    assert result.canvas_scale == pytest.approx(gt.max_scale(shot_id))
    assert result.coverage.max() >= 1
    assert np.array_equal(result.image_bgra[..., 3] > 0, result.coverage > 0)


def test_evaluation_detects_misregistration(synthetic: SyntheticCache) -> None:
    """Une reconstruction décalée de 3 px doit échouer aux seuils de référence."""
    gt = synthetic.get("pan_horizontal")
    good = _oracle(gt, 0)
    shifted_rgba = np.roll(good.image_bgra, 3, axis=1)
    bad = MosaicResult(good.sequence, shifted_rgba, np.roll(good.coverage, 3, axis=1),
                       good.transforms, good.crop, good.canvas_scale)
    metrics = evaluate_mosaic(bad, gt, 0)
    assert metrics.ssim < 0.8
    assert any(f.startswith("SSIM") for f in metrics.check(ReferenceThresholds()))
    # Poses faussées de 1 % d'échelle : détectées par la métrique de pose.
    scaled = {i: SimilarityTransform(1.01) @ t for i, t in good.transforms.items()}
    scaled[min(scaled)] = good.transforms[min(scaled)]
    skewed = MosaicResult(good.sequence, good.image_bgra, good.coverage, scaled, good.crop,
                          good.canvas_scale)
    assert evaluate_mosaic(skewed, gt, 0).pose.max_scale_rel > 0.005


def test_coverage_overreach_is_reported(synthetic: SyntheticCache) -> None:
    """Annoncer couverts 20 px de marge jamais observés doit être signalé."""
    gt = synthetic.get("pan_vertical")
    good = _oracle(gt, 0)
    pad = 20
    rgba = np.pad(good.image_bgra, ((pad, pad), (pad, pad), (0, 0)), constant_values=255)
    coverage = np.pad(good.coverage, pad, constant_values=1)
    crop = CropBox(good.crop.x0 - pad, good.crop.y0 - pad, good.crop.x1 + pad, good.crop.y1 + pad)
    metrics = evaluate_mosaic(
        MosaicResult(good.sequence, rgba, coverage, good.transforms, crop, good.canvas_scale),
        gt, 0,
    )
    reference = evaluate_mosaic(good, gt, 0)
    assert metrics.true_coverage == pytest.approx(reference.true_coverage, abs=1e-3)
    assert metrics.coverage_overreach > 0.05
    assert any("surestimée" in f for f in metrics.check(ReferenceThresholds()))


def test_oracle_memory_guard(synthetic: SyntheticCache) -> None:
    gt = synthetic.get("short")
    with pytest.raises(MemoryError):
        oracle_mosaic(iter(()), gt, 0, max_stack_bytes=1024)
    with pytest.raises(ValueError):
        oracle_mosaic(iter(()), gt, 0)
    assert math.isfinite(gt.max_scale(0))
