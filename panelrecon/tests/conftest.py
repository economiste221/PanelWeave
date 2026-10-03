from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from panelrecon.core.config import PipelineConfig
from panelrecon.core.models import FrameObs
from panelrecon.core.registration import PanelMaskProvider, RegistrationResult, register_frames
from panelrecon.core.synthetic import GroundTruth, generate_video, scenario
from panelrecon.core.video_io import VideoReader
from panelrecon.tests.videofactory import index_frame, write_video

# Tests de l'interface sans écran (doit précéder la création de la QApplication).
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

N_FRAMES = 24
FPS = 25
# Instants irréguliers (VFR) : intervalles de 20 à 100 ms.
VFR_TIMES = [round(sum((0.02, 0.1, 0.04)[k % 3] for k in range(i)), 3) for i in range(N_FRAMES)]


@pytest.fixture
def small_config() -> PipelineConfig:
    config = PipelineConfig()
    config.preprocess.motion_long_side = 160
    config.video.max_fps = 0.0  # lecture brute : toutes les frames
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


class SyntheticCache:
    """Génère chaque scénario au plus une fois par session de tests."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._cache: dict[str, GroundTruth] = {}

    def get(self, name: str) -> GroundTruth:
        if name not in self._cache:
            self._cache[name] = generate_video(scenario(name), self.root / name)
        return self._cache[name]


@pytest.fixture(scope="session")
def synthetic(tmp_path_factory: pytest.TempPathFactory) -> SyntheticCache:
    return SyntheticCache(tmp_path_factory.mktemp("synthetic"))


class RegistrationCache:
    """Recalage (phase 3) mis en cache par scénario, plan et usage des masques exacts."""

    def __init__(self, synthetic: SyntheticCache) -> None:
        self.synthetic = synthetic
        self._cache: dict[tuple[str, int, bool], RegistrationResult] = {}

    def masks(self, gt: GroundTruth) -> PanelMaskProvider:
        truth = {f.index: f for f in gt.frames}
        return lambda frame: gt.visibility_mask(truth[frame.index])

    def frames(self, gt: GroundTruth, shot_id: int, proxy: bool = True) -> Iterator[FrameObs]:
        cfg = PipelineConfig()
        shot = gt.shots[shot_id]
        with VideoReader(gt.video_path, cfg.video, cfg.preprocess) as reader:
            yield from reader.frames(start_index=shot.start_idx, stop_index=shot.end_idx + 1,
                                     compute_proxy=proxy)

    def get(self, name: str, shot_id: int = 0, with_masks: bool = True) -> RegistrationResult:
        key = (name, shot_id, with_masks)
        if key not in self._cache:
            gt = self.synthetic.get(name)
            masks = self.masks(gt) if with_masks else None
            self._cache[key] = register_frames(self.frames(gt, shot_id), PipelineConfig(), masks)
        return self._cache[key]


@pytest.fixture(scope="session")
def registered(synthetic: SyntheticCache) -> RegistrationCache:
    return RegistrationCache(synthetic)
