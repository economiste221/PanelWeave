from __future__ import annotations

from pathlib import Path

import pytest

from panelrecon.core.config import PipelineConfig
from panelrecon.tests.videofactory import index_frame, write_video

N_FRAMES = 24
FPS = 25
# Instants irréguliers (VFR) : intervalles de 20 à 100 ms.
VFR_TIMES = [round(sum((0.02, 0.1, 0.04)[k % 3] for k in range(i)), 3) for i in range(N_FRAMES)]


@pytest.fixture
def small_config() -> PipelineConfig:
    config = PipelineConfig()
    config.preprocess.motion_long_side = 160
    return config


@pytest.fixture
def cfr_mp4(tmp_path: Path) -> Path:
    return write_video(
        tmp_path / "cfr.mp4", (index_frame(i) for i in range(N_FRAMES)), fps=FPS
    )


@pytest.fixture
def vfr_mkv(tmp_path: Path) -> Path:
    return write_video(
        tmp_path / "vfr.mkv", (index_frame(i) for i in range(N_FRAMES)), fps=FPS, times_s=VFR_TIMES
    )


@pytest.fixture
def corrupt_mp4(tmp_path: Path) -> Path:
    path = tmp_path / "corrupt.mp4"
    path.write_bytes(bytes(range(256)) * 64)
    return path
