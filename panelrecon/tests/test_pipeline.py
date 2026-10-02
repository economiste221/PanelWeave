"""Tests de l'export, de l'orchestration et de la CLI de reconstruction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from panelrecon import cli
from panelrecon.core.config import PipelineConfig
from panelrecon.core.evaluation import ReferenceThresholds, evaluate_mosaic
from panelrecon.core.export import coverage_colormap, export_sequence, sequence_basename
from panelrecon.core.models import (
    CancellationToken,
    CropBox,
    MosaicResult,
    OperationCancelled,
    Sequence,
    SimilarityTransform,
)
from panelrecon.core.mosaic import build_mosaic
from panelrecon.core.pipeline import plan_chunks, process_video
from panelrecon.core.scene_split import SequenceRegistration, SplitResult, concatenate_splits
from panelrecon.tests.conftest import RegistrationCache, SyntheticCache


def _imread(path: Path) -> NDArray[Any]:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    assert image is not None, path
    return np.asarray(image)


def _result_from_files(image: Path, coverage: Path | None, report: dict[str, Any]) -> MosaicResult:
    """Reconstitue un MosaicResult à partir des fichiers exportés."""
    assert coverage is not None
    transforms = {int(k): v for k, v in report["registration"]["transforms"].items()}
    return MosaicResult(
        Sequence(report["sequence"]["start_idx"], report["sequence"]["end_idx"],
                 tuple(report["sequence"]["excluded"])),
        _imread(image), _imread(coverage),
        {i: SimilarityTransform.from_dict(t) for i, t in transforms.items()},
        CropBox(**report["crop"]), report["canvas_scale"],
    )


def test_sequence_basename() -> None:
    assert sequence_basename("vid", 3, Sequence(12, 345)) == "vid_seq003_000012-000345"


def test_coverage_colormap() -> None:
    cov = np.array([[0, 1], [5, 10]], np.uint16)
    colored = coverage_colormap(cov)
    assert colored.shape == (2, 2, 3) and not colored[0, 0].any() and colored[1, 1].any()


def test_export_sequence(registered: RegistrationCache, tmp_path: Path) -> None:
    gt = registered.synthetic.get("short")
    registration = registered.get("short")
    result = build_mosaic(registered.frames(gt, 0, False), registration, PipelineConfig(),
                          registered.masks(gt))
    files = export_sequence(result, registration, tmp_path, "x", PipelineConfig())
    image = _imread(files.image)
    assert image.shape[2] == 4 and np.array_equal(image, result.image_bgra)
    assert files.coverage is not None and files.coverage_color is not None
    coverage = _imread(files.coverage)
    assert coverage.dtype == np.uint16 and np.array_equal(coverage, result.coverage)
    report = json.loads(files.report.read_text(encoding="utf-8"))
    assert report["image"] == {"width": result.crop.width, "height": result.crop.height}
    assert report["frames_used"] == 4 and len(report["registration"]["transforms"]) == 4
    assert report["coverage"]["observations_max"] == int(result.coverage.max())
    assert 0 < report["coverage"]["coverage_ratio"] <= 1
    assert not list(tmp_path.glob("*.tmp*"))
    cfg = PipelineConfig()
    cfg.export.save_coverage_map = False
    cfg.export.save_registration = False
    minimal = export_sequence(result, registration, tmp_path / "m", "y", cfg)
    assert minimal.coverage is None
    assert "transforms" not in json.loads(minimal.report.read_text())["registration"]


def test_process_video(synthetic: SyntheticCache, tmp_path: Path) -> None:
    gt = synthetic.get("pan_horizontal")
    events: list[tuple[float, str]] = []
    outcome = process_video(gt.video_path, tmp_path, PipelineConfig(),
                            progress=lambda f, m: events.append((f, m)))
    assert outcome.ok and outcome.info is not None and len(outcome.sequences) == 1
    files = outcome.sequences[0].files
    assert files is not None and files.image.is_file()
    assert files.image.parent == tmp_path / "pan_horizontal"
    fractions = [f for f, _ in events]
    assert fractions == sorted(fractions) and fractions[-1] == 1.0
    report = json.loads(files.report.read_text(encoding="utf-8"))
    result = _result_from_files(files.image, files.coverage, report)
    assert evaluate_mosaic(result, gt, 0).check(ReferenceThresholds()) == []


def test_process_video_failure_is_captured(corrupt_mp4: Path, tmp_path: Path) -> None:
    outcome = process_video(corrupt_mp4, tmp_path, PipelineConfig())
    assert not outcome.ok and outcome.error is not None and "VideoOpenError" in outcome.error
    assert outcome.traceback is not None


def test_process_video_cancellation(synthetic: SyntheticCache, tmp_path: Path) -> None:
    token = CancellationToken()
    token.cancel()
    with pytest.raises(OperationCancelled):
        process_video(synthetic.get("short").video_path, tmp_path, PipelineConfig(), cancel=token)


def test_process_video_splits_panels(synthetic: SyntheticCache, tmp_path: Path) -> None:
    """Vidéo à deux panels séparés par un fondu : deux séquences, deux PNG conformes,
    frames de mélange écartées et rapportées."""
    gt = synthetic.get("crossfade")
    outcome = process_video(gt.video_path, tmp_path, PipelineConfig())
    assert outcome.ok and len(outcome.sequences) == 2
    assert outcome.split is not None
    (transition,) = outcome.split.transitions
    assert transition.kind == "dissolve" and len(transition.excluded) == 6
    for k, seq in enumerate(outcome.sequences):
        assert seq.files is not None
        report = json.loads(seq.files.report.read_text(encoding="utf-8"))
        assert report["segmentation"]["segmented"]
        result = _result_from_files(seq.files.image, seq.files.coverage, report)
        assert evaluate_mosaic(result, gt, k).check(ReferenceThresholds()) == []
    summary = json.loads((tmp_path / "crossfade" / "crossfade_sequences.json").read_text())
    assert len(summary["sequences"]) == 2 and summary["transitions"][0]["kind"] == "dissolve"


def test_cli_reconstruct(synthetic: SyntheticCache, corrupt_mp4: Path, tmp_path: Path) -> None:
    gt = synthetic.get("short")
    out = tmp_path / "out"
    code = cli.main(["-i", str(gt.video_path), str(corrupt_mp4), "-o", str(out)])
    assert code == cli.EXIT_FAILURES
    report = json.loads((out / cli.BATCH_REPORT_FILENAME).read_text(encoding="utf-8"))
    assert report["n_ok"] == 1 and report["n_failed"] == 1
    assert (out / "short" / "short_seq000_000000-000003.png").is_file()
    assert cli.main(["-i", str(gt.video_path), "-o", str(tmp_path / "ok")]) == cli.EXIT_OK


# ------------------------------------------------------------------ tronçons


def test_plan_chunks() -> None:
    cfg = PipelineConfig()
    # 1 h à 30 i/s, tronçons de 5 min : 12 tronçons contigus.
    chunks = plan_chunks(108000, 30.0, cfg, 8)
    assert len(chunks) == 12 and chunks[0][0] == 0 and chunks[-1][1] == 108000
    assert all(a[1] == b[0] for a, b in zip(chunks[:-1], chunks[1:], strict=True))
    # 3 min : réparties sur les processus (tronçons d'au moins 30 s).
    assert len(plan_chunks(5400, 30.0, cfg, 8)) == 6
    assert plan_chunks(5400, 30.0, cfg, 1) == [(0, 5400)]
    cfg.runtime.chunk_seconds = 0.0
    assert plan_chunks(108000, 30.0, cfg, 8) == [(0, 108000)]
    assert plan_chunks(0, 30.0, cfg, 8) == []


def _chunked_config() -> PipelineConfig:
    cfg = PipelineConfig()
    cfg.runtime.num_workers = 2
    cfg.runtime.chunk_seconds = 0.4  # 10 frames à 25 i/s
    cfg.runtime.min_chunk_seconds = 0.4
    return cfg


def test_chunked_split_stitches_same_panel(synthetic: SyntheticCache, tmp_path: Path) -> None:
    """Panoramique découpé en 3 tronçons (processus séparés) : les frontières montrent
    le même panel, les morceaux sont réunis en une seule séquence conforme."""
    gt = synthetic.get("pan_horizontal")
    outcome = process_video(gt.video_path, tmp_path, _chunked_config())
    assert outcome.ok and outcome.chunks == 3 and outcome.workers == 2
    assert outcome.split is not None and outcome.split.transitions == []
    (seq,) = outcome.sequences
    assert (seq.sequence.start_idx, seq.sequence.end_idx) == (0, 29) and seq.frames_used == 30
    assert seq.files is not None
    report = json.loads(seq.files.report.read_text(encoding="utf-8"))
    result = _result_from_files(seq.files.image, seq.files.coverage, report)
    assert evaluate_mosaic(result, gt, 0).check(ReferenceThresholds()) == []


def test_chunked_split_keeps_transitions(synthetic: SyntheticCache, tmp_path: Path) -> None:
    """Deux panels, 5 tronçons : mêmes séquences et même fondu qu'en lecture continue."""
    gt = synthetic.get("crossfade")
    outcome = process_video(gt.video_path, tmp_path, _chunked_config())
    assert outcome.ok and outcome.chunks == 5
    assert [(s.sequence.start_idx, s.sequence.end_idx) for s in outcome.sequences] == [
        (shot.start_idx, shot.end_idx) for shot in gt.shots]
    assert outcome.split is not None
    (transition,) = outcome.split.transitions
    assert transition.kind == "dissolve" and len(transition.excluded) == 6
    for k, seq in enumerate(outcome.sequences):
        assert seq.files is not None
        report = json.loads(seq.files.report.read_text(encoding="utf-8"))
        result = _result_from_files(seq.files.image, seq.files.coverage, report)
        assert evaluate_mosaic(result, gt, k).check(ReferenceThresholds()) == []


def test_concatenate_splits_without_link_records_a_cut(registered: RegistrationCache) -> None:
    gt_a = registered.synthetic.get("crossfade")
    first = registered.get("crossfade", 0)
    second = registered.get("crossfade", 1)
    a = SplitResult([SequenceRegistration(Sequence(gt_a.shots[0].start_idx,
                                                   gt_a.shots[0].end_idx), first)], frames_seen=20)
    b = SplitResult([SequenceRegistration(Sequence(gt_a.shots[1].start_idx,
                                                   gt_a.shots[1].end_idx), second)], frames_seen=20)
    merged = concatenate_splits(a, b, None, "frontière de tronçon")
    assert len(merged.sequences) == 2 and merged.frames_seen == 40
    (cut,) = merged.transitions
    assert cut.kind == "cut" and cut.first_after == gt_a.shots[1].start_idx
    assert concatenate_splits(SplitResult(), b, None, "x").transitions == []
