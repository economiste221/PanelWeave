from __future__ import annotations

import logging
import math

import cv2
import numpy as np
import pytest

from panelrecon.core.config import PipelineConfig
from panelrecon.core.evaluation import pairwise_error
from panelrecon.core.models import MotionEstimate, MotionMethod, SimilarityTransform
from panelrecon.core.motion import (
    MotionEstimator,
    MotionFrame,
    photometric_consistency,
    stretch_contrast,
    to_native,
)
from panelrecon.core.synthetic import GroundTruth
from panelrecon.core.video_io import VideoReader
from panelrecon.tests.conftest import SyntheticCache


def _frames(gt: GroundTruth, config: PipelineConfig | None = None) -> list[MotionFrame]:
    cfg = config or PipelineConfig()
    with VideoReader(gt.video_path, cfg.video, cfg.preprocess) as reader:
        out = []
        for f in reader.frames():
            assert f.proxy_gray is not None
            out.append(MotionFrame(f.index, f.proxy_gray, None, f.proxy_factor))
        return out


def _error_px(gt: GroundTruth, src: MotionFrame, dst: MotionFrame, transform: SimilarityTransform
              ) -> float:
    t_src, t_dst = gt.frames[src.index].transform, gt.frames[dst.index].transform
    assert t_src is not None and t_dst is not None
    native = transform.rescaled(src.factor, dst.factor)
    return pairwise_error(native, t_src, t_dst, gt.screen_size).corner_px


# --------------------------------------------------------------------- briques


def test_stretch_contrast() -> None:
    low = np.tile(np.linspace(150, 180, 64, dtype=np.float32), (32, 1)).astype(np.uint8)
    out = stretch_contrast(low)
    assert out.min() <= 2 and out.max() >= 253
    assert np.all(np.diff(out[0].astype(int)) >= 0)  # monotone
    flat = np.full((10, 10), 77, np.uint8)
    assert np.array_equal(stretch_contrast(flat), flat)
    # Même transformation pour deux images comparées.
    other = (low.astype(int) + 5).clip(0, 255).astype(np.uint8)
    a, b = stretch_contrast(low, None, other), stretch_contrast(other, None, low)
    assert a.mean() < b.mean()


def test_rotation_regularization_shrinks_rotation() -> None:
    rng = np.random.default_rng(0)
    src = rng.uniform(0, 300, (40, 2))
    dst = SimilarityTransform(1.0, 0.02, 3.0, -2.0).apply(src)
    free = SimilarityTransform.fit(src, dst)
    damped = SimilarityTransform.fit(src, dst, rotation_regularization=9.0)
    assert free.theta == pytest.approx(0.02, abs=1e-9)
    assert damped.theta == pytest.approx(math.atan(math.tan(0.02) / 10.0), rel=1e-3)
    with pytest.raises(ValueError):
        SimilarityTransform.fit(src, dst, rotation_regularization=-1.0)


def test_motion_frame_validation() -> None:
    gray = np.zeros((20, 30), np.uint8)
    with pytest.raises(ValueError):
        MotionFrame(0, np.zeros((20, 30, 3), np.uint8))
    with pytest.raises(ValueError):
        MotionFrame(0, gray, np.zeros((10, 10), np.uint8))
    with pytest.raises(ValueError):
        MotionFrame(0, gray, factor=0.0)
    assert MotionFrame(0, gray).size == (30, 20)
    assert MotionFrame(0, gray).valid_mask().min() == 255


def test_features_respect_mask_and_are_cached(synthetic: SyntheticCache) -> None:
    gt = synthetic.get("pan_horizontal")
    frame = _frames(gt)[0]
    mask = np.zeros_like(frame.gray)
    mask[:, : frame.gray.shape[1] // 2] = 255
    masked = MotionFrame(frame.index, frame.gray, mask, frame.factor)
    est = MotionEstimator(PipelineConfig().motion)
    feats = est.features(masked)
    assert len(feats.points) > 100
    assert feats.points[:, 0].max() < frame.gray.shape[1] // 2
    assert est.features(masked) is feats


# ------------------------------------------------------- estimation sur le synthétique


@pytest.mark.parametrize(
    ("name", "max_error"),
    [("pan_horizontal", 0.2), ("zoom_in", 0.2), ("pan_zoom_eased", 0.2), ("subtitles", 0.25),
     ("vfr", 0.2), ("flat_texture", 0.75)],
)
def test_pairwise_accuracy(synthetic: SyntheticCache, name: str, max_error: float) -> None:
    gt = synthetic.get(name)
    frames = _frames(gt)
    est = MotionEstimator(PipelineConfig().motion)
    errors = []
    for src, dst in zip(frames[::3], frames[1::3]):
        estimate = est.estimate(src, dst)
        assert estimate.accepted, estimate.reason
        assert estimate.ncc is not None and estimate.ncc > 0.9
        errors.append(_error_px(gt, src, dst, estimate.transform))
    assert max(errors) < max_error


def test_orb_detector(synthetic: SyntheticCache) -> None:
    gt = synthetic.get("pan_horizontal")
    cfg = PipelineConfig()
    cfg.motion.detector = "orb"
    frames = _frames(gt)
    estimate = MotionEstimator(cfg.motion).estimate(frames[4], frames[5])
    assert estimate.accepted and estimate.method is MotionMethod.ORB
    assert _error_px(gt, frames[4], frames[5], estimate.transform) < 0.3


@pytest.mark.parametrize("stage", ["_estimate_flow", "_estimate_phase_correlation"])
def test_fallback_stages_alone(synthetic: SyntheticCache, stage: str) -> None:
    gt = synthetic.get("pan_zoom_eased")
    frames = _frames(gt)
    est = MotionEstimator(PipelineConfig().motion)
    src, dst = frames[10], frames[11]
    candidate = getattr(est, stage)(src, dst)
    assert candidate.accepted, candidate.reason
    assert _error_px(gt, src, dst, candidate.transform) < 2.0  # avant raffinement
    refined = est._refine_and_validate(candidate, src, dst)
    assert refined.accepted and refined.ecc_refined
    assert _error_px(gt, src, dst, refined.transform) < 0.3


def test_log_polar_recovers_zoom_and_sign(synthetic: SyntheticCache) -> None:
    """Zoom de 10 % entre deux frames : la corrélation log-polaire doit retrouver
    l'échelle (et son sens) sans aucun point d'intérêt."""
    gt = synthetic.get("zoom_in")
    frames = _frames(gt)
    est = MotionEstimator(PipelineConfig().motion)
    src, dst = frames[5], frames[12]
    t_src, t_dst = gt.frames[5].transform, gt.frames[12].transform
    assert t_src is not None and t_dst is not None
    true_scale = t_dst.scale / t_src.scale
    assert true_scale > 1.1
    candidate = est._estimate_phase_correlation(src, dst)
    assert candidate.transform.scale == pytest.approx(true_scale, rel=0.01)
    refined = est._refine_and_validate(candidate, src, dst)
    assert refined.accepted
    assert _error_px(gt, src, dst, refined.transform) < 0.3
    reversed_candidate = est._estimate_phase_correlation(dst, src)
    assert reversed_candidate.transform.scale == pytest.approx(1 / true_scale, rel=0.01)


def test_cascade_falls_back_and_logs(
    synthetic: SyntheticCache, caplog: pytest.LogCaptureFixture
) -> None:
    gt = synthetic.get("pan_horizontal")
    frames = _frames(gt)
    est = MotionEstimator(PipelineConfig().motion)

    def failing_features(src: MotionFrame, dst: MotionFrame) -> MotionEstimate:
        return est._failed(src, dst, MotionMethod.SIFT, 0, "échec simulé")

    est._estimate_features = failing_features  # type: ignore[method-assign]
    with caplog.at_level(logging.INFO, logger="panelrecon.core.motion"):
        estimate = est.estimate(frames[3], frames[4])
    assert estimate.accepted and estimate.method is MotionMethod.FARNEBACK
    assert any("repli farneback" in r.getMessage() for r in caplog.records)
    assert _error_px(gt, frames[3], frames[4], estimate.transform) < 0.3


def test_different_panels_are_rejected(
    synthetic: SyntheticCache, caplog: pytest.LogCaptureFixture
) -> None:
    gt = synthetic.get("crossfade")
    frames = _frames(gt)
    a = frames[gt.shots[0].end_idx]
    b = frames[gt.shots[1].start_idx]
    with caplog.at_level(logging.WARNING, logger="panelrecon.core.motion"):
        estimate = MotionEstimator(PipelineConfig().motion).estimate(a, b)
    assert not estimate.accepted
    for stage in ("sift", "farneback", "phase_correlation"):
        assert stage in estimate.reason
    assert any("rejetée" in r.getMessage() for r in caplog.records)


def test_identical_frames_give_identity(synthetic: SyntheticCache) -> None:
    gt = synthetic.get("duplicates")
    frames = _frames(gt)
    estimate = MotionEstimator(PipelineConfig().motion).estimate(frames[2], frames[3])
    assert gt.frames[3].is_duplicate
    assert estimate.accepted
    pts = np.array([[0.0, 0.0], [639.0, 359.0]])
    assert np.abs(estimate.transform.apply(pts) - pts).max() < 0.1


def test_native_conversion_with_downscaled_proxy(synthetic: SyntheticCache) -> None:
    gt = synthetic.get("pan_zoom_eased")
    cfg = PipelineConfig()
    cfg.preprocess.motion_long_side = 320  # facteur 0,5
    frames = _frames(gt, cfg)
    assert frames[0].factor == pytest.approx(0.5)
    est = MotionEstimator(cfg.motion)
    for i in (3, 17, 30):
        estimate = est.estimate(frames[i], frames[i + 1])
        assert estimate.accepted
        native = to_native(estimate, frames[i], frames[i + 1])
        t_src, t_dst = gt.frames[i].transform, gt.frames[i + 1].transform
        assert t_src is not None and t_dst is not None
        assert pairwise_error(native, t_src, t_dst, gt.screen_size).corner_px < 0.6


def test_determinism(synthetic: SyntheticCache) -> None:
    gt = synthetic.get("zoom_in")
    frames = _frames(gt)
    runs = []
    for _ in range(2):
        fresh = [MotionFrame(f.index, f.gray, f.mask, f.factor) for f in frames[:6]]
        est = MotionEstimator(PipelineConfig().motion, seed=7)
        runs.append([est.estimate(a, b).transform for a, b in zip(fresh, fresh[1:])])
    assert runs[0] == runs[1]


def test_photometric_consistency_discriminates(synthetic: SyntheticCache) -> None:
    for name in ("pan_horizontal", "flat_texture", "subtitles"):
        gt = synthetic.get(name)
        frames = _frames(gt)
        src, dst = frames[6], frames[7]
        t_src, t_dst = gt.frames[6].transform, gt.frames[7].transform
        assert t_src is not None and t_dst is not None
        true = t_dst @ t_src.inverse()
        good, overlap = photometric_consistency(src, dst, true)
        bad, _ = photometric_consistency(src, dst, SimilarityTransform.from_translation(4, 0) @ true)
        assert good > 0.95 and bad < 0.85 and overlap > 0.5


def test_masked_estimation_with_ground_truth_visibility(synthetic: SyntheticCache) -> None:
    """Avec le masque exact du panel (ce que fournira la segmentation), l'estimation
    ignore le fond flou et reste précise."""
    gt = synthetic.get("pan_vertical")
    frames = _frames(gt)
    est = MotionEstimator(PipelineConfig().motion)
    for i in (2, 15):
        masks = [np.asarray(cv2.erode(gt.visibility_mask(gt.frames[j]), np.ones((3, 3), np.uint8)),
                            dtype=np.uint8) for j in (i, i + 1)]
        src = MotionFrame(i, frames[i].gray, masks[0], frames[i].factor)
        dst = MotionFrame(i + 1, frames[i + 1].gray, masks[1], frames[i + 1].factor)
        estimate = est.estimate(src, dst)
        assert estimate.accepted
        assert _error_px(gt, src, dst, estimate.transform) < 0.2
