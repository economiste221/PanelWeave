"""Tests de référence du recalage par chaînage (spécification : translation < 1 px,
échelle < 0,5 %, repère canonique à l'échelle maximale observée)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from panelrecon import cli
from panelrecon.core.config import PipelineConfig
from panelrecon.core.evaluation import ReferenceThresholds, pose_errors
from panelrecon.core.models import CancellationToken, FrameObs, MaskU8, OperationCancelled
from panelrecon.core.registration import (
    ChainRegistrar,
    RegistrationResult,
    make_motion_frame,
    register_frames,
)
from panelrecon.core.synthetic import GroundTruth
from panelrecon.core.video_io import VideoReader
from panelrecon.tests.conftest import SyntheticCache

THRESHOLDS = ReferenceThresholds()


def _register(gt: GroundTruth, shot_id: int, config: PipelineConfig | None = None,
              **kwargs: object) -> RegistrationResult:
    cfg = config or PipelineConfig()
    shot = gt.shots[shot_id]
    with VideoReader(gt.video_path, cfg.video, cfg.preprocess) as reader:
        frames = reader.frames(start_index=shot.start_idx, stop_index=shot.end_idx + 1)
        return register_frames(frames, cfg, **kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("name", "shot_id"),
    [("static", 0), ("pan_horizontal", 0), ("pan_vertical", 0), ("zoom_in", 0),
     ("zoom_out", 0), ("pan_zoom_eased", 0), ("duplicates", 0), ("crossfade", 0),
     ("crossfade", 1), ("subtitles", 0), ("flat_texture", 0), ("short", 0), ("vfr", 0)],
)
def test_chained_registration_meets_reference(
    synthetic: SyntheticCache, name: str, shot_id: int
) -> None:
    gt = synthetic.get(name)
    result = _register(gt, shot_id)
    shot = gt.shots[shot_id]
    assert not result.interrupted and not result.rejected
    assert result.registered_indices == list(range(shot.start_idx, shot.end_idx + 1))
    errors = pose_errors(result.transforms, gt.transforms(shot_id), gt.screen_size)
    assert errors.max_translation_px < THRESHOLDS.max_translation_px
    assert errors.max_scale_rel < THRESHOLDS.max_scale_rel
    # Repère canonique : la frame la plus zoomée est à l'échelle 1, donc la jauge
    # panel → canevas a l'échelle maximale observée.
    assert min(t.scale for t in result.transforms.values()) == pytest.approx(1.0)
    assert errors.gauge.scale == pytest.approx(gt.max_scale(shot_id), rel=THRESHOLDS.max_scale_rel)
    assert result.min_inlier_ratio > 0.2 and result.rms_reprojection_error < 1.0


def test_registration_with_downscaled_proxy(synthetic: SyntheticCache) -> None:
    gt = synthetic.get("pan_zoom_eased")
    cfg = PipelineConfig()
    cfg.preprocess.motion_long_side = 320
    errors = pose_errors(_register(gt, 0, cfg).transforms, gt.transforms(0), gt.screen_size)
    assert errors.max_translation_px < THRESHOLDS.max_translation_px
    assert errors.max_scale_rel < THRESHOLDS.max_scale_rel


def test_exclusion_zone_and_panel_masks(synthetic: SyntheticCache) -> None:
    gt = synthetic.get("subtitles")
    assert gt.subtitle_box is not None
    x0, y0, x1, y1 = gt.subtitle_box
    cfg = PipelineConfig()
    cfg.preprocess.exclusion_zones = ((x0 / gt.screen_width, y0 / gt.screen_height,
                                       x1 / gt.screen_width, y1 / gt.screen_height),)
    truth = {f.index: f for f in gt.frames}

    def panel_mask(frame: FrameObs) -> MaskU8:
        return gt.visibility_mask(truth[frame.index])

    result = _register(gt, 0, cfg, panel_masks=panel_mask)
    errors = pose_errors(result.transforms, gt.transforms(0), gt.screen_size)
    assert errors.max_translation_px < 0.5 and errors.max_scale_rel < 0.003


def test_make_motion_frame_combines_masks(synthetic: SyntheticCache) -> None:
    gt = synthetic.get("subtitles")
    cfg = PipelineConfig()
    cfg.preprocess.motion_long_side = 320
    cfg.preprocess.exclusion_zones = ((0.0, 0.0, 0.5, 1.0),)
    with VideoReader(gt.video_path, cfg.video, cfg.preprocess) as reader:
        frame = next(reader.frames())
    panel = np.zeros((frame.height, frame.width), np.uint8)
    panel[: frame.height // 2] = 255
    mf = make_motion_frame(frame, cfg, panel)
    assert mf.mask is not None and mf.mask.shape == (180, 320) and mf.factor == 0.5
    assert not mf.mask[:, :150].any() and not mf.mask[100:].any() and mf.mask[:80, 170:].all()
    with pytest.raises(ValueError):
        make_motion_frame(frame, cfg, np.zeros((10, 10), np.uint8))
    with VideoReader(gt.video_path, cfg.video, cfg.preprocess) as reader:
        raw = next(reader.frames(compute_proxy=False))
    with pytest.raises(ValueError):
        make_motion_frame(raw, cfg)


def test_rejection_and_interruption_across_panels(synthetic: SyntheticCache) -> None:
    """Sans découpage en séquences, les frames d'un autre panel sont rejetées (et
    exclues) puis la séquence est déclarée interrompue."""
    gt = synthetic.get("crossfade")
    cfg = PipelineConfig()
    cfg.registration.max_consecutive_failures = 2
    a, b = gt.shots
    wanted = set(range(a.end_idx - 3, a.end_idx + 1)) | set(range(b.start_idx, b.start_idx + 5))

    def frames() -> Iterator[FrameObs]:
        with VideoReader(gt.video_path, cfg.video, cfg.preprocess) as reader:
            yield from (f for f in reader.frames() if f.index in wanted)

    result = register_frames(frames(), cfg)
    assert result.interrupted and "échecs consécutifs" in result.interruption_reason
    assert result.registered_indices == list(range(a.end_idx - 3, a.end_idx + 1))
    assert result.excluded == [b.start_idx, b.start_idx + 1, b.start_idx + 2]
    assert all(not e.accepted for e in result.rejected)


def test_isolated_failure_keeps_anchor(synthetic: SyntheticCache) -> None:
    """Une frame aberrante isolée est exclue ; la suivante se recale sur l'ancre."""
    gt = synthetic.get("pan_horizontal")
    other = synthetic.get("crossfade")
    cfg = PipelineConfig()
    with VideoReader(other.video_path, cfg.video, cfg.preprocess) as reader:
        intruder = next(f for f in reader.frames() if f.index == other.shots[1].start_idx + 5)
    registrar = ChainRegistrar(cfg)
    with VideoReader(gt.video_path, cfg.video, cfg.preprocess) as reader:
        for frame in reader.frames(stop_index=10):
            if frame.index == 5:
                fake = FrameObs(5, frame.time_s, frame.pts, intruder.image, intruder.proxy_gray,
                                intruder.proxy_factor)
                estimate = registrar.add(make_motion_frame(fake, cfg), (640, 360))
                assert estimate is not None and not estimate.accepted
                continue
            registrar.add(make_motion_frame(frame, cfg), (frame.width, frame.height))
    result = registrar.result()
    assert result.excluded == [5] and 5 not in result.transforms
    errors = pose_errors(result.transforms, gt.transforms(0), gt.screen_size)
    assert errors.max_translation_px < 0.5


def test_registrar_errors(synthetic: SyntheticCache) -> None:
    cfg = PipelineConfig()
    registrar = ChainRegistrar(cfg)
    with pytest.raises(ValueError):
        registrar.result()
    gt = synthetic.get("short")
    with VideoReader(gt.video_path, cfg.video, cfg.preprocess) as reader:
        frames = list(reader.frames())
    registrar.add(make_motion_frame(frames[1], cfg), (640, 360))
    with pytest.raises(ValueError):
        registrar.add(make_motion_frame(frames[0], cfg), (640, 360))


def test_cancellation(synthetic: SyntheticCache) -> None:
    gt = synthetic.get("short")
    token = CancellationToken()
    token.cancel()
    with pytest.raises(OperationCancelled):
        _register(gt, 0, cancel=token)


def test_cli_register_mode(synthetic: SyntheticCache, tmp_path: Path) -> None:
    gt = synthetic.get("pan_horizontal")
    out = tmp_path / "out"
    assert cli.main(["--mode", "register", "-i", str(gt.video_path), "-o", str(out)]) == 0
    report = json.loads((out / "pan_horizontal_registration.json").read_text(encoding="utf-8"))
    assert len(report["transforms"]) == len(gt.frames) and not report["interrupted"]
    assert len(report["estimates"]) == len(gt.frames) - 1
    assert report["estimates"][0]["method"] == "sift"
    # Une vidéo à deux panels (sans découpage) est signalée en échec, sans planter.
    cf = synthetic.get("crossfade")
    assert cli.main(["--mode", "register", "-i", str(cf.video_path), "-o", str(out)]) in (0, 1)
    assert (out / "crossfade_registration.json").is_file()


def test_registration_result_statistics() -> None:
    result = RegistrationResult({}, [], [], [], 0, 1.0)
    assert result.mean_inlier_ratio == 0.0 and result.min_inlier_ratio == 0.0
    assert result.rms_reprojection_error == 0.0
