"""Tests de la segmentation panel / fond : briques, segmenteur image par image et
estimation de l'emprise du panel sur les séquences synthétiques (sans masque exact)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from panelrecon.core.config import PipelineConfig
from panelrecon.core.evaluation import ReferenceThresholds, evaluate_mosaic, fit_gauge
from panelrecon.core.geometry import corners
from panelrecon.core.models import FrameObs, MosaicResult, SimilarityTransform
from panelrecon.core.mosaic import build_mosaic
from panelrecon.core.segmentation import (
    ClassicSegmenter,
    PanelRegion,
    PanelRegionEstimator,
    SegmenterState,
    local_sharpness,
    polygon_mask,
    sharp_evidence,
)
from panelrecon.core.synthetic import PanelSpec, make_panel
from panelrecon.core.video_io import VideoReader
from panelrecon.tests.conftest import RegistrationCache

THRESHOLDS = ReferenceThresholds()


def test_sharpness_separates_blur() -> None:
    panel = make_panel(PanelSpec(600, 400, "rich", seed=2)).mean(axis=2).astype(np.uint8)
    blurred = np.asarray(cv2.GaussianBlur(panel, (0, 0), 8), dtype=np.uint8)
    cfg = PipelineConfig().segmentation
    assert local_sharpness(panel, 9).mean() > 20 * local_sharpness(blurred, 9).mean()
    # Frame composite (cas réel) : panel net à gauche, fond flou à droite.
    frame = np.hstack([panel[:, :300], blurred[:, 300:]])
    evidence = sharp_evidence(frame, cfg)
    assert evidence[:, :290].mean() > 0.2
    assert evidence[:, 310:].mean() < 0.01


def test_polygon_mask() -> None:
    mask = polygon_mask(np.array([[9.5, 4.5], [29.5, 4.5], [29.5, 14.5], [9.5, 14.5]]), (40, 20))
    assert mask.shape == (20, 40)
    assert mask[5:15, 10:30].all() and not mask[:4].any() and not mask[:, :9].any()


def _region(registered: RegistrationCache, name: str, shot_id: int = 0) -> PanelRegion:
    gt = registered.synthetic.get(name)
    registration = registered.get(name, shot_id, with_masks=False)
    cfg = PipelineConfig()
    frames = list(registered.frames(gt, shot_id))
    estimator = PanelRegionEstimator(cfg, registration.transforms, registration.frame_sizes,
                                     frames[0].proxy_factor)
    for frame in frames:
        estimator.add(frame)
    return estimator.estimate()


def _mosaic(registered: RegistrationCache, name: str, shot_id: int = 0) -> MosaicResult:
    gt = registered.synthetic.get(name)
    region = _region(registered, name, shot_id)
    registration = registered.get(name, shot_id, with_masks=False)
    return build_mosaic(registered.frames(gt, shot_id, proxy=False), registration,
                        PipelineConfig(), region.mask, clip=region.bounds())


@pytest.mark.parametrize(
    ("name", "shot_id"),
    [("static", 0), ("pan_horizontal", 0), ("pan_vertical", 0), ("zoom_in", 0), ("zoom_out", 0),
     ("pan_zoom_eased", 0), ("duplicates", 0), ("crossfade", 0), ("crossfade", 1),
     ("flat_texture", 0), ("short", 0), ("vfr", 0)],
)
def test_automatic_segmentation_meets_reference(
    registered: RegistrationCache, name: str, shot_id: int
) -> None:
    """Segmentation automatique (aucun masque exact) : le fond n'est jamais compté
    comme couvert et la reconstruction respecte les seuils de référence."""
    gt = registered.synthetic.get(name)
    metrics = evaluate_mosaic(_mosaic(registered, name, shot_id), gt, shot_id)
    assert metrics.check(THRESHOLDS) == [], metrics
    assert metrics.coverage_overreach < 0.002


def test_region_matches_true_panel_rectangle(registered: RegistrationCache) -> None:
    """Quand le panel est entièrement délimité (panel statique et visible en entier),
    le rectangle estimé coïncide avec le vrai bord du panel à ~2 px près."""
    gt = registered.synthetic.get("static")
    region = _region(registered, "static")
    registration = registered.get("static", 0, with_masks=False)
    gauge = fit_gauge(registration.transforms, gt.transforms(0), gt.screen_size)
    shot = gt.shots[0]
    truth = gauge.apply(corners(shot.panel_width, shot.panel_height))
    assert np.abs(region.polygon - truth).max() < 2.5
    assert region.segmented


def test_region_mask_per_frame(registered: RegistrationCache) -> None:
    gt = registered.synthetic.get("pan_vertical")
    region = _region(registered, "pan_vertical")
    frame = next(registered.frames(gt, 0))
    mask = region.mask(frame) > 0
    truth = gt.visibility_mask(gt.frames[0]) > 0
    disagreement = (mask ^ truth).mean()
    assert disagreement < 0.01
    x0, y0, x1, y1 = region.bounds()
    assert x1 > x0 and y1 > y0


def test_no_evidence_falls_back_to_full_frame() -> None:
    cfg = PipelineConfig()
    gray = np.full((36, 64), 128, np.uint8)
    frames = [FrameObs(i, 0.0, i, np.full((36, 64, 3), 128, np.uint8), gray, 1.0)
              for i in range(3)]
    transforms = {i: SimilarityTransform() for i in range(3)}
    estimator = PanelRegionEstimator(cfg, transforms, {i: (64, 36) for i in range(3)}, 1.0)
    for frame in frames:
        estimator.add(frame)
    region = estimator.estimate()
    assert not region.segmented
    assert (region.mask(frames[0]) > 0).mean() > 0.9


def test_classic_segmenter_per_frame(registered: RegistrationCache) -> None:
    """Segmenteur image par image : sur zoom_in, le masque recouvre le panel visible
    et exclut l'essentiel du fond flou ; le lissage temporel borne les sauts."""
    gt = registered.synthetic.get("zoom_in")
    segmenter = ClassicSegmenter(PipelineConfig())
    state = SegmenterState()
    cfg = PipelineConfig()
    previous_transform = None
    registration = registered.get("zoom_in", 0, with_masks=False)
    with VideoReader(gt.video_path, cfg.video, cfg.preprocess) as reader:
        for frame in reader.frames(stop_index=12):
            if previous_transform is not None:
                state.motion = registration.transforms[frame.index].inverse() @ previous_transform
            mask, state = segmenter.segment(frame.image, state)
            previous_transform = registration.transforms[frame.index]
            truth = gt.visibility_mask(gt.frames[frame.index]) > 0
            m = mask > 0
            recall = (m & truth).sum() / truth.sum()
            precision = (m & truth).sum() / max(1, m.sum())
            assert recall > 0.95 and precision > 0.9, (frame.index, recall, precision)
