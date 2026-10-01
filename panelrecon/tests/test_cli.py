from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from panelrecon import cli, synth_cli
from panelrecon.core.config import PipelineConfig
from panelrecon.core.synthetic import GroundTruth


def test_write_default_config(tmp_path: Path) -> None:
    target = tmp_path / "cfg.json"
    assert cli.main(["--write-default-config", str(target)]) == cli.EXIT_OK
    assert PipelineConfig.load(target) == PipelineConfig()


def test_missing_arguments() -> None:
    assert cli.main([]) == cli.EXIT_USAGE


def test_invalid_config(tmp_path: Path, cfr_mp4: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"video": {"frame_step": 0}}), encoding="utf-8")
    code = cli.main(["-i", str(cfr_mp4), "-o", str(tmp_path / "out"), "-c", str(bad)])
    assert code == cli.EXIT_USAGE


def test_inventory_mixed_batch(
    tmp_path: Path, cfr_mp4: Path, vfr_mkv: Path, corrupt_mp4: Path
) -> None:
    out = tmp_path / "out"
    code = cli.main(["--input", str(tmp_path), "--output", str(out), "--log-level", "WARNING"])
    assert code == cli.EXIT_FAILURES  # la vidéo corrompue fait échouer le lot…
    report = json.loads((out / cli.INVENTORY_FILENAME).read_text(encoding="utf-8"))
    by_name = {Path(v["path"]).name: v for v in report["videos"]}
    assert set(by_name) == {"cfr.mp4", "vfr.mkv", "corrupt.mp4"}  # …sans l'interrompre
    assert report["n_ok"] == 2 and report["n_failed"] == 1
    assert by_name["corrupt.mp4"]["status"] == "error"
    assert "VideoOpenError" in by_name["corrupt.mp4"]["error"]
    cfr = by_name["cfr.mp4"]
    assert cfr["status"] == "ok" and cfr["frames_kept"] == 24
    assert cfr["variable_frame_rate"] is False
    assert cfr["interval_median_s"] == pytest.approx(0.04, abs=1e-6)
    assert by_name["vfr.mkv"]["variable_frame_rate"] is True
    assert (out / cli.LOG_FILENAME).read_text(encoding="utf-8").count("corrupt.mp4") >= 1


def test_inventory_with_config_ok(tmp_path: Path, cfr_mp4: Path) -> None:
    cfg = PipelineConfig()
    cfg.video.frame_step = 2
    cfg_path = tmp_path / "cfg.json"
    cfg.save(cfg_path)
    out = tmp_path / "out"
    assert cli.main(["-i", str(cfr_mp4), "-o", str(out), "-c", str(cfg_path)]) == cli.EXIT_OK
    report = json.loads((out / cli.INVENTORY_FILENAME).read_text(encoding="utf-8"))
    assert report["videos"][0]["frames_kept"] == 12
    assert report["config"]["video"]["frame_step"] == 2


def test_no_video_found(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    assert cli.main(["-i", str(empty), "-o", str(tmp_path / "out")]) == cli.EXIT_FAILURES


def test_module_entry_point(tmp_path: Path, cfr_mp4: Path) -> None:
    out = tmp_path / "out"
    proc = subprocess.run(
        [sys.executable, "-m", "panelrecon.cli", "-i", str(cfr_mp4), "-o", str(out)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert (out / cli.INVENTORY_FILENAME).is_file()


# ------------------------------------------------------------ générateur synthétique


def test_synth_cli_list_and_usage() -> None:
    assert synth_cli.main(["--list"]) == 0
    assert synth_cli.main([]) == 2


def test_synth_cli_generates_and_evaluates_oracle(tmp_path: Path) -> None:
    assert synth_cli.main(["-o", str(tmp_path), "-s", "short", "--oracle"]) == 0
    gt = GroundTruth.load(tmp_path / "short_ground_truth.json")
    assert gt.video_path.is_file() and len(gt.frames) == 4


def test_synth_cli_custom_panel_image(tmp_path: Path) -> None:
    noise = np.random.default_rng(0).integers(0, 256, (900, 1200, 3), dtype=np.uint8)
    image = np.asarray(cv2.GaussianBlur(noise, (0, 0), 1.5), dtype=np.uint8)
    panel = tmp_path / "mon_panel.png"
    cv2.imwrite(str(panel), image)
    out = tmp_path / "out"
    assert synth_cli.main(["-o", str(out), "-s", "short", "--panel-image", str(panel)]) == 0
    gt = GroundTruth.load(out / "short_ground_truth.json")
    assert np.array_equal(gt.load_panel(0), image)
    assert synth_cli.main(["-o", str(out), "-s", "short", "--panel-image",
                           str(tmp_path / "absent.png")]) == 2
