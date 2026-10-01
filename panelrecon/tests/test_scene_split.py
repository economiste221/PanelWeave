"""Tests du découpage en séquences (coupes franches, fondus, perturbations isolées)."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from panelrecon.core.config import PipelineConfig
from panelrecon.core.evaluation import ReferenceThresholds, pose_errors
from panelrecon.core.models import CancellationToken, FrameObs, OperationCancelled
from panelrecon.core.scene_split import (
    SequenceTracker,
    histogram_correlation,
    hsv_histogram,
    split_and_register,
    thumbnail,
)
from panelrecon.core.synthetic import GroundTruth, generate_video, scenario
from panelrecon.core.video_io import VideoReader
from panelrecon.tests.conftest import SyntheticCache

THRESHOLDS = ReferenceThresholds()


def _frames(gt: GroundTruth, config: PipelineConfig | None = None) -> Iterator[FrameObs]:
    cfg = config or PipelineConfig()
    with VideoReader(gt.video_path, cfg.video, cfg.preprocess) as reader:
        yield from reader.frames()


def _spans(gt: GroundTruth, config: PipelineConfig | None = None) -> list[tuple[int, int]]:
    result = split_and_register(_frames(gt, config), config or PipelineConfig())
    return [(s.sequence.start_idx, s.sequence.end_idx) for s in result.sequences]


def test_histogram_helpers() -> None:
    a = np.zeros((90, 320, 3), np.uint8)
    a[:] = (255, 0, 0)  # bleu vif
    b = np.zeros((90, 320, 3), np.uint8)
    b[:] = (0, 120, 0)  # vert sombre
    small = thumbnail(a, 160)
    assert small.shape == (45, 160, 3) and thumbnail(small, 640) is small
    ha, hb = hsv_histogram(a), hsv_histogram(b)
    assert ha.sum() == pytest.approx(1.0)
    assert histogram_correlation(ha, ha) == pytest.approx(1.0)
    assert histogram_correlation(ha, hb) < 0.5


def test_dissolve_is_split_and_mixtures_excluded(synthetic: SyntheticCache) -> None:
    gt = synthetic.get("crossfade")
    result = split_and_register(_frames(gt), PipelineConfig())
    assert [(s.sequence.start_idx, s.sequence.end_idx) for s in result.sequences] == [
        (s.start_idx, s.end_idx) for s in gt.expected_sequences()
    ]
    (transition,) = result.transitions
    assert transition.kind == "dissolve"
    assert transition.excluded == tuple(f.index for f in gt.frames if f.shot_id is None)
    for k, item in enumerate(result.sequences):
        errors = pose_errors(item.registration.transforms, gt.transforms(k), gt.screen_size)
        assert errors.max_translation_px < THRESHOLDS.max_translation_px
        assert errors.max_scale_rel < THRESHOLDS.max_scale_rel


def test_hard_cut(synthetic: SyntheticCache, tmp_path: Path) -> None:
    gt = generate_video(replace(scenario("crossfade"), name="cut", crossfade_frames=0), tmp_path)
    result = split_and_register(_frames(gt), PipelineConfig())
    assert [(s.sequence.start_idx, s.sequence.end_idx) for s in result.sequences] == [
        (0, 19), (20, 39)]
    (transition,) = result.transitions
    assert transition.kind == "cut" and transition.excluded == ()
    assert any("histogramme" in r for r in transition.reasons)


@pytest.mark.parametrize(
    "name", ["static", "pan_horizontal", "pan_vertical", "zoom_in", "zoom_out", "pan_zoom_eased",
             "duplicates", "flat_texture", "short", "vfr"],
)
def test_single_panel_videos_stay_whole(synthetic: SyntheticCache, name: str) -> None:
    gt = synthetic.get(name)
    result = split_and_register(_frames(gt), PipelineConfig())
    assert [(s.sequence.start_idx, s.sequence.end_idx) for s in result.sequences] == [
        (0, len(gt.frames) - 1)]
    assert result.transitions == [] and result.sequences[0].sequence.excluded == ()


def test_isolated_glitch_is_merged(synthetic: SyntheticCache) -> None:
    """Une frame étrangère au milieu d'un plan est écartée sans couper la séquence."""
    gt = synthetic.get("pan_horizontal")
    intruder_video = synthetic.get("crossfade")
    cfg = PipelineConfig()
    with VideoReader(intruder_video.video_path, cfg.video, cfg.preprocess) as reader:
        intruder = next(f for f in reader.frames() if f.index == 35)

    def frames() -> Iterator[FrameObs]:
        for frame in _frames(gt):
            if frame.index == 12:
                yield FrameObs(12, frame.time_s, frame.pts, intruder.image, intruder.proxy_gray,
                               intruder.proxy_factor)
            else:
                yield frame

    result = split_and_register(frames(), cfg)
    (item,) = result.sequences
    assert (item.sequence.start_idx, item.sequence.end_idx) == (0, 29)
    assert item.sequence.excluded == (12,)
    assert 12 not in item.registration.transforms
    (transition,) = result.transitions
    assert transition.kind == "glitch"
    errors = pose_errors(item.registration.transforms, gt.transforms(0), gt.screen_size)
    assert errors.max_translation_px < THRESHOLDS.max_translation_px


def test_lag_detects_dissolve_without_histogram(synthetic: SyntheticCache) -> None:
    """Histogramme désactivé : le contrôle de stabilité à décalage détecte le fondu."""
    gt = synthetic.get("crossfade")
    cfg = PipelineConfig()
    cfg.scenes.histogram_min_correlation = -1.0
    result = split_and_register(_frames(gt), cfg)
    assert len(result.sequences) == 2
    first, second = (s.sequence for s in result.sequences)
    assert first.start_idx == 0 and first.end_idx < gt.shots[1].start_idx
    assert second.start_idx >= gt.shots[1].start_idx - 1
    assert any("progressif" in r or "mouvement" in r for t in result.transitions for r in t.reasons)


def test_min_sequence_frames_and_long_transitions(synthetic: SyntheticCache) -> None:
    gt = synthetic.get("crossfade")
    cfg = PipelineConfig()
    cfg.scenes.min_sequence_frames = 25
    result = split_and_register(_frames(gt), cfg)
    assert result.sequences == [] and len(result.dropped) == 2
    cfg = PipelineConfig()
    cfg.scenes.max_transition_frames = 2
    result = split_and_register(_frames(gt), cfg)
    assert len(result.sequences) == 2 and result.discarded
    excluded = set(result.discarded) | {i for t in result.transitions for i in t.excluded}
    assert excluded == {f.index for f in gt.frames if f.shot_id is None}


def test_tracker_lifecycle_and_cancellation(synthetic: SyntheticCache) -> None:
    gt = synthetic.get("short")
    tracker = SequenceTracker(PipelineConfig())
    for frame in _frames(gt):
        tracker.push(frame)
    result = tracker.finish()
    assert tracker.finish() is result and result.frames_seen == 4
    with pytest.raises(RuntimeError):
        tracker.push(next(_frames(gt)))
    token = CancellationToken()
    token.cancel()
    with pytest.raises(OperationCancelled):
        split_and_register(_frames(gt), PipelineConfig(), token)
