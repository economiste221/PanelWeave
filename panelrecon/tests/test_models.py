from __future__ import annotations

import math
import multiprocessing
import threading

import cv2
import numpy as np
import pytest

from panelrecon.core.models import (
    CancellationToken,
    CropBox,
    FrameObs,
    MosaicResult,
    MotionEstimate,
    MotionMethod,
    NotASimilarityError,
    OperationCancelled,
    QualityReport,
    Sequence,
    SimilarityTransform,
    Verdict,
)


def _random_similarities(n: int, seed: int = 0) -> list[SimilarityTransform]:
    rng = np.random.default_rng(seed)
    return [
        SimilarityTransform(
            scale=float(rng.uniform(0.3, 3.0)),
            theta=float(rng.uniform(-0.5, 0.5)),
            tx=float(rng.uniform(-500, 500)),
            ty=float(rng.uniform(-500, 500)),
        )
        for _ in range(n)
    ]


def test_matrix_layout() -> None:
    t = SimilarityTransform(2.0, math.pi / 2, 3.0, 4.0)
    m = t.matrix()
    np.testing.assert_allclose(m, [[0.0, -2.0, 3.0], [2.0, 0.0, 4.0]], atol=1e-12)
    np.testing.assert_allclose(t.homogeneous()[2], [0, 0, 1])


def test_from_matrix_roundtrip() -> None:
    for t in _random_similarities(20):
        back = SimilarityTransform.from_matrix(t.matrix())
        np.testing.assert_allclose(back.matrix(), t.matrix(), atol=1e-9)
        back3 = SimilarityTransform.from_matrix(t.homogeneous())
        assert back3.scale == pytest.approx(t.scale)


def test_from_matrix_accepts_opencv_partial_affine() -> None:
    rng = np.random.default_rng(1)
    src = rng.uniform(0, 500, size=(50, 2)).astype(np.float32)
    truth = SimilarityTransform(1.3, 0.05, 12.0, -7.0)
    dst = truth.apply(src).astype(np.float32)
    m, _ = cv2.estimateAffinePartial2D(src, dst)
    est = SimilarityTransform.from_matrix(m, tol=1e-4)
    assert est.scale == pytest.approx(1.3, rel=1e-4)
    assert est.theta == pytest.approx(0.05, abs=1e-4)


def test_from_matrix_rejects_non_similarity() -> None:
    with pytest.raises(NotASimilarityError):
        SimilarityTransform.from_matrix(np.array([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
    with pytest.raises(NotASimilarityError):
        SimilarityTransform.from_matrix(np.array([[1.0, 0, 0], [0, 1.0, 0], [1e-3, 0, 1.0]]))
    with pytest.raises(NotASimilarityError):
        SimilarityTransform.from_matrix(np.full((2, 3), np.nan))
    with pytest.raises(ValueError):
        SimilarityTransform.from_matrix(np.eye(4))


def test_invalid_parameters() -> None:
    with pytest.raises(ValueError):
        SimilarityTransform(scale=0.0)
    with pytest.raises(ValueError):
        SimilarityTransform(tx=math.inf)


def test_compose_and_inverse() -> None:
    ts = _random_similarities(10, seed=2)
    pts = np.random.default_rng(3).uniform(-100, 100, size=(30, 2))
    for a, b in zip(ts[:-1], ts[1:]):
        composed = a @ b
        np.testing.assert_allclose(composed.apply(pts), a.apply(b.apply(pts)), atol=1e-7)
        np.testing.assert_allclose(
            composed.homogeneous(), a.homogeneous() @ b.homogeneous(), atol=1e-9
        )
        ident = a @ a.inverse()
        np.testing.assert_allclose(ident.matrix(), np.eye(3)[:2], atol=1e-9)
        np.testing.assert_allclose(a.inverse().apply(a.apply(pts)), pts, atol=1e-8)


def test_helpers() -> None:
    zoom = SimilarityTransform.from_scale_about(2.0, 10.0, 20.0)
    np.testing.assert_allclose(zoom.apply(np.array([[10.0, 20.0]])), [[10.0, 20.0]])
    np.testing.assert_allclose(zoom.apply(np.array([[11.0, 20.0]])), [[12.0, 20.0]])
    shift = SimilarityTransform.from_translation(3.0, -1.0)
    np.testing.assert_allclose(shift.apply(np.zeros((1, 2))), [[3.0, -1.0]])
    assert SimilarityTransform.identity().log_scale == 0.0
    assert SimilarityTransform.from_dict(zoom.to_dict()) == zoom
    with pytest.raises(ValueError):
        zoom.apply(np.zeros((3, 3)))


@pytest.mark.parametrize(("f_src", "f_dst"), [(0.5, 0.5), (0.25, 0.5), (0.8, 0.3)])
def test_rescaled_is_conjugation(f_src: float, f_dst: float) -> None:
    """Une similarité estimée en coordonnées proxy, convertie en natif, doit
    commuter avec le changement d'échelle des points."""
    m_proxy = SimilarityTransform(1.1, 0.02, 15.0, -4.0)
    native_pts = np.random.default_rng(4).uniform(0, 1000, size=(20, 2))
    expected = m_proxy.apply(native_pts * f_src) / f_dst
    m_native = m_proxy.rescaled(f_src, f_dst)
    np.testing.assert_allclose(m_native.apply(native_pts), expected, atol=1e-9)
    with pytest.raises(ValueError):
        m_proxy.rescaled(0.0)


def test_rescaled_matches_warp_on_images() -> None:
    """Vérifie la conjugaison sur de vraies images : warp en proxy ≈ warp natif réduit."""
    rng = np.random.default_rng(5)
    native = cv2.GaussianBlur(rng.integers(0, 256, (400, 600), dtype=np.uint8), (0, 0), 3)
    m_native = SimilarityTransform(1.05, 0.0, 20.0, 10.0)
    f = 0.5
    m_proxy = m_native.rescaled(1.0 / f)  # natif -> proxy est l'opération inverse
    np.testing.assert_allclose(m_proxy.rescaled(f).matrix(), m_native.matrix(), atol=1e-12)
    warped_native = cv2.warpAffine(native, m_native.matrix(), (600, 400), flags=cv2.INTER_LINEAR)
    proxy = cv2.resize(native, (300, 200), interpolation=cv2.INTER_AREA)
    warped_proxy = cv2.warpAffine(proxy, m_proxy.matrix(), (300, 200), flags=cv2.INTER_LINEAR)
    reduced = cv2.resize(warped_native, (300, 200), interpolation=cv2.INTER_AREA)
    inner = (slice(30, 170), slice(30, 270))
    diff = np.abs(reduced[inner].astype(float) - warped_proxy[inner].astype(float))
    assert diff.mean() < 2.0


def test_sequence() -> None:
    seq = Sequence(10, 14, excluded=(12,))
    assert len(seq) == 5
    assert 12 in seq and 15 not in seq and "x" not in seq
    assert seq.usable_indices == (10, 11, 13, 14)
    assert seq.to_dict() == {"start_idx": 10, "end_idx": 14, "excluded": [12]}
    with pytest.raises(ValueError):
        Sequence(5, 4)
    with pytest.raises(ValueError):
        Sequence(0, 3, excluded=(4,))


def test_frame_obs_validation() -> None:
    img = np.zeros((10, 20, 3), np.uint8)
    obs = FrameObs(index=0, time_s=0.0, pts=0, image=img)
    assert (obs.width, obs.height) == (20, 10)
    with pytest.raises(ValueError):
        FrameObs(index=0, time_s=0.0, pts=0, image=np.zeros((10, 20), np.uint8))
    with pytest.raises(ValueError):
        FrameObs(index=-1, time_s=0.0, pts=0, image=img)
    with pytest.raises(ValueError):
        FrameObs(index=0, time_s=0.0, pts=0, image=img, proxy_factor=2.0)


def test_motion_estimate_validation() -> None:
    est = MotionEstimate(0, 1, SimilarityTransform(), 100, 80, 0.8, 0.5, MotionMethod.SIFT)
    assert est.to_dict()["method"] == "sift"
    with pytest.raises(ValueError):
        MotionEstimate(0, 1, SimilarityTransform(), 10, 20, 0.8, 0.5, MotionMethod.SIFT)
    with pytest.raises(ValueError):
        MotionEstimate(0, 1, SimilarityTransform(), 10, 5, 1.5, 0.5, MotionMethod.SIFT)
    with pytest.raises(ValueError):
        MotionEstimate(0, 1, SimilarityTransform(), 10, 5, 0.5, float("nan"), MotionMethod.ORB)


def test_mosaic_result_and_report() -> None:
    crop = CropBox(5, 5, 25, 15)
    assert (crop.width, crop.height) == (20, 10)
    res = MosaicResult(
        sequence=Sequence(0, 3),
        image_bgra=np.zeros((10, 20, 4), np.uint8),
        coverage=np.zeros((10, 20), np.uint16),
        transforms={0: SimilarityTransform()},
        crop=crop,
        canvas_scale=1.0,
    )
    assert res.crop.to_dict()["x1"] == 25
    with pytest.raises(ValueError):
        CropBox(5, 5, 5, 10)
    with pytest.raises(ValueError):
        MosaicResult(Sequence(0, 1), np.zeros((10, 20, 3), np.uint8),
                     np.zeros((10, 20), np.uint16), {}, crop, 1.0)
    report = QualityReport(Sequence(0, 3), 4, 0.9, 0.8, 0.4, 0.98, 0.97, 120.0,
                           Verdict.TO_REVIEW, ("couverture faible",))
    assert report.to_dict()["verdict"] == "À VÉRIFIER"


def test_cancellation_token_threading() -> None:
    token = CancellationToken()
    token.raise_if_cancelled()
    worker_saw_cancel = threading.Event()

    def worker() -> None:
        while not token.cancelled:
            pass
        worker_saw_cancel.set()

    thread = threading.Thread(target=worker)
    thread.start()
    token.cancel()
    thread.join(timeout=5)
    assert worker_saw_cancel.is_set()
    with pytest.raises(OperationCancelled):
        token.raise_if_cancelled()


def test_cancellation_token_multiprocessing_event() -> None:
    event = multiprocessing.get_context("spawn").Event()
    token = CancellationToken(event)
    assert not token.cancelled
    event.set()
    assert token.cancelled
