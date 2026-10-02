from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from panelrecon.core import video_io
from panelrecon.core.config import PipelineConfig
from panelrecon.core.models import CancellationToken, FrameObs, OperationCancelled
from panelrecon.core.video_io import (
    FrameRingBuffer,
    VideoOpenError,
    VideoReader,
    build_exclusion_mask,
    compute_proxy_factor,
    discover_videos,
    make_proxy,
    probe_video,
    to_gray,
)
from panelrecon.tests.conftest import FPS, N_FRAMES, VFR_TIMES
from panelrecon.tests.videofactory import decoded_index, index_frame, write_video


def _read_all(path: Path, config: PipelineConfig, **kwargs: object) -> list[FrameObs]:
    with VideoReader(path, config.video, config.preprocess) as reader:
        return list(reader.frames(**kwargs))  # type: ignore[arg-type]


# ----------------------------------------------------------------- découverte


def test_discover_videos(tmp_path: Path) -> None:
    (tmp_path / "sub" / "deep").mkdir(parents=True)
    for name in ("a.mp4", "b.MKV", "c.txt", ".hidden.mp4", "sub/d.webm", "sub/deep/e.mov"):
        (tmp_path / name).write_bytes(b"x")
    exts = PipelineConfig().video.extensions
    names = [p.name for p in discover_videos([tmp_path], exts, recursive=True)]
    assert names == sorted(["a.mp4", "b.MKV", "d.webm", "e.mov"])
    flat = [p.name for p in discover_videos([tmp_path], exts, recursive=False)]
    assert flat == ["a.mp4", "b.MKV"]
    # Fichier explicite + doublon + entrée absente + extension refusée.
    explicit = discover_videos(
        [tmp_path / "a.mp4", tmp_path, tmp_path / "absent.mp4", tmp_path / "c.txt"], exts
    )
    assert [p.name for p in explicit].count("a.mp4") == 1


# -------------------------------------------------------------- prétraitement


def test_proxy_factor_and_resize() -> None:
    assert compute_proxy_factor(1920, 1080, 960) == pytest.approx(0.5)
    assert compute_proxy_factor(1080, 1920, 960) == pytest.approx(0.5)
    assert compute_proxy_factor(640, 480, 960) == 1.0  # jamais d'agrandissement
    with pytest.raises(ValueError):
        compute_proxy_factor(0, 10, 960)
    image = np.random.default_rng(0).integers(0, 256, (1080, 1920, 3), dtype=np.uint8)
    proxy = make_proxy(image, 0.5)
    assert proxy.shape == (540, 960, 3)
    same = make_proxy(image, 1.0)
    assert same.shape == image.shape and same is not image
    assert to_gray(image).shape == (1080, 1920)
    gray = to_gray(image)
    assert to_gray(gray) is gray


def test_exclusion_mask() -> None:
    mask = build_exclusion_mask(100, 200, [(0.0, 0.8, 1.0, 1.0), (0.9, 0.0, 1.0, 0.1)])
    assert mask.dtype == np.uint8 and mask.shape == (100, 200)
    assert (mask[80:, :] == 0).all()
    assert (mask[:10, 180:] == 0).all()
    assert mask[50, 50] == 255 and mask[79, 10] == 255
    assert (build_exclusion_mask(4, 4, []) == 255).all()


def test_ring_buffer() -> None:
    buf = FrameRingBuffer(3)
    img = np.zeros((4, 4, 3), np.uint8)
    for i in range(5):
        buf.append(FrameObs(index=i * 2, time_s=float(i), pts=i, image=img))
    assert len(buf) == 3 and buf.capacity == 3
    assert [f.index for f in buf] == [4, 6, 8]
    assert buf.latest().index == 8
    assert buf.get(6) is not None and buf.get(2) is None and buf.get(7) is None
    with pytest.raises(ValueError):
        buf.append(FrameObs(index=8, time_s=0.0, pts=0, image=img))
    buf.clear()
    with pytest.raises(IndexError):
        buf.latest()
    with pytest.raises(ValueError):
        FrameRingBuffer(0)


# --------------------------------------------------------------------- lecture


def test_read_cfr_mp4(cfr_mp4: Path, small_config: PipelineConfig) -> None:
    info = probe_video(cfr_mp4, small_config.video)
    assert (info.width, info.height) == (320, 240)
    assert info.fps == pytest.approx(FPS)
    assert info.backend == video_io.BACKEND_PYAV
    assert info.frame_count == N_FRAMES
    frames = _read_all(cfr_mp4, small_config)
    assert [f.index for f in frames] == list(range(N_FRAMES))
    assert [decoded_index(f.image) for f in frames] == list(range(N_FRAMES))
    np.testing.assert_allclose([f.time_s for f in frames], np.arange(N_FRAMES) / FPS, atol=1e-6)
    first = frames[0]
    assert first.image.shape == (240, 320, 3) and first.image.flags.c_contiguous
    assert first.proxy_gray is not None and first.proxy_gray.shape == (120, 160)
    assert first.proxy_factor == pytest.approx(0.5)
    assert first.pts is not None


@pytest.mark.parametrize("suffix", [".mkv", ".webm", ".mov"])
def test_read_other_containers(tmp_path: Path, small_config: PipelineConfig, suffix: str) -> None:
    path = write_video(tmp_path / f"v{suffix}", (index_frame(i) for i in range(10)), fps=FPS)
    frames = _read_all(path, small_config)
    assert [decoded_index(f.image) for f in frames] == list(range(10))


def test_read_vfr(vfr_mkv: Path, small_config: PipelineConfig) -> None:
    frames = _read_all(vfr_mkv, small_config)
    assert len(frames) == N_FRAMES
    np.testing.assert_allclose([f.time_s for f in frames], VFR_TIMES, atol=1.5e-3)
    assert [decoded_index(f.image) for f in frames] == list(range(N_FRAMES))


def test_frame_step_and_range(cfr_mp4: Path, small_config: PipelineConfig) -> None:
    small_config.video.frame_step = 3
    frames = _read_all(cfr_mp4, small_config)
    assert [f.index for f in frames] == list(range(0, N_FRAMES, 3))
    assert [decoded_index(f.image) for f in frames] == [f.index for f in frames]
    frames = _read_all(cfr_mp4, small_config, start_index=4, stop_index=14)
    assert [f.index for f in frames] == [4, 7, 10, 13]
    with pytest.raises(ValueError):
        _read_all(cfr_mp4, small_config, start_index=5, stop_index=2)


def test_max_fps_uses_timestamps(vfr_mkv: Path, small_config: PipelineConfig) -> None:
    small_config.video.max_fps = 10.0  # au plus une frame par créneau de 100 ms
    frames = _read_all(vfr_mkv, small_config)
    buckets = [int(np.floor(f.time_s * 10.0 + 0.1)) for f in frames]
    assert buckets == sorted(set(buckets))  # une frame par créneau, ordre conservé
    expected_buckets = sorted({int(np.floor(t * 10.0 + 0.1)) for t in VFR_TIMES})
    assert buckets == expected_buckets  # chaque créneau occupé garde sa première frame
    first_of_bucket: dict[int, int] = {}
    for i, t in enumerate(VFR_TIMES):
        first_of_bucket.setdefault(int(np.floor(t * 10.0 + 0.1)), i)
    assert [f.index for f in frames] == [first_of_bucket[b] for b in expected_buckets]


def test_max_fps_halves_60fps_with_rounded_timestamps(tmp_path: Path,
                                                      small_config: PipelineConfig) -> None:
    """Vidéo à ~60 i/s dont les pts sont arrondis à la milliseconde (16/17 ms) :
    plafonnée à 30 i/s, elle doit garder exactement une frame sur deux."""
    times = [round(i / 59.94, 3) for i in range(60)]
    path = write_video(tmp_path / "v60.mkv", (index_frame(i % 24) for i in range(60)),
                       fps=60, times_s=times)
    small_config.video.max_fps = 30.0
    frames = _read_all(path, small_config)
    assert [f.index for f in frames] == list(range(0, 60, 2))


def test_without_proxy(cfr_mp4: Path, small_config: PipelineConfig) -> None:
    frames = _read_all(cfr_mp4, small_config, compute_proxy=False)
    assert frames[0].proxy_gray is None and frames[0].proxy_factor == 1.0


def test_reader_can_iterate_twice(cfr_mp4: Path, small_config: PipelineConfig) -> None:
    with VideoReader(cfr_mp4, small_config.video, small_config.preprocess) as reader:
        a = [f.index for f in reader.frames(compute_proxy=False)]
        b = [f.index for f in reader.frames(compute_proxy=False)]
        assert reader.frames_decoded == N_FRAMES
    assert a == b == list(range(N_FRAMES))


def test_cancellation(cfr_mp4: Path, small_config: PipelineConfig) -> None:
    token = CancellationToken()
    seen = []
    with VideoReader(cfr_mp4, small_config.video, small_config.preprocess, cancel=token) as r:
        with pytest.raises(OperationCancelled):
            for obs in r.frames():
                seen.append(obs.index)
                if obs.index == 2:
                    token.cancel()
    assert seen == [0, 1, 2]


# ---------------------------------------------------------------- robustesse


def test_corrupt_file(corrupt_mp4: Path, small_config: PipelineConfig) -> None:
    with pytest.raises(VideoOpenError):
        probe_video(corrupt_mp4, small_config.video)
    small_config.video.allow_opencv_fallback = False
    with pytest.raises(VideoOpenError, match="PyAV"):
        VideoReader(corrupt_mp4, small_config.video, small_config.preprocess)
    with pytest.raises(VideoOpenError, match="introuvable"):
        VideoReader(corrupt_mp4.with_name("absent.mp4"), small_config.video, small_config.preprocess)


def test_truncated_file_does_not_crash(tmp_path: Path, small_config: PipelineConfig) -> None:
    full = write_video(tmp_path / "full.mkv", (index_frame(i) for i in range(N_FRAMES)), fps=FPS)
    data = full.read_bytes()
    truncated = tmp_path / "trunc.mkv"
    truncated.write_bytes(data[: len(data) // 2])
    frames = _read_all(truncated, small_config)
    assert 0 < len(frames) < N_FRAMES
    # Les frames perdues (B-frames sans référence) ne décalent pas les timestamps :
    # le contenu de chaque frame correspond toujours à son instant de présentation.
    assert [decoded_index(f.image) for f in frames] == [round(f.time_s * FPS) for f in frames]
    assert [f.index for f in frames] == list(range(len(frames)))


def test_opencv_fallback(
    cfr_mp4: Path, small_config: PipelineConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _Broken:
        def __init__(self, path: Path, *args: object, **kwargs: object) -> None:
            raise VideoOpenError(f"PyAV indisponible pour {path}")

    monkeypatch.setattr(video_io, "_PyAVBackend", _Broken)
    with VideoReader(cfr_mp4, small_config.video, small_config.preprocess) as reader:
        assert reader.info.backend == video_io.BACKEND_OPENCV
        frames = list(reader.frames())
    assert [decoded_index(f.image) for f in frames] == list(range(N_FRAMES))
    np.testing.assert_allclose([f.time_s for f in frames], np.arange(N_FRAMES) / FPS, atol=2e-3)
    small_config.video.allow_opencv_fallback = False
    with pytest.raises(VideoOpenError):
        VideoReader(cfr_mp4, small_config.video, small_config.preprocess)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="binaire ffmpeg requis pour ce test")
def test_display_rotation_matches_ffmpeg(
    cfr_mp4: Path, tmp_path: Path, small_config: PipelineConfig
) -> None:
    rotated = tmp_path / "rot.mp4"
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-display_rotation", "90",
         "-i", str(cfr_mp4), "-c", "copy", str(rotated)],
        check=True,
    )
    info = probe_video(rotated, small_config.video)
    assert info.rotation_deg == 90 and (info.width, info.height) == (240, 320)
    frames = _read_all(rotated, small_config, stop_index=1)
    ours = frames[0].image
    raw = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", str(rotated), "-frames:v", "1",
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
        check=True, capture_output=True,
    ).stdout
    reference = np.frombuffer(raw, np.uint8).reshape(ours.shape)
    assert np.abs(reference.astype(int) - ours.astype(int)).mean() < 1.0

    small_config.video.apply_display_rotation = False
    frames = _read_all(rotated, small_config, stop_index=1)
    assert frames[0].image.shape == (240, 320, 3)


def test_letterbox_mask() -> None:
    from panelrecon.core.video_io import letterbox_mask

    gray = np.full((60, 100), 120, np.uint8)
    gray[:, :15] = 3  # bande noire à gauche
    gray[:, 90:] = 0  # bande noire à droite
    gray[:5] = 2  # bande en haut
    mask = letterbox_mask(gray, 20.0, 4.0)
    assert not mask[:, :15].any() and not mask[:, 90:].any() and not mask[:5].any()
    assert mask[5:, 15:90].all()
    # Une zone sombre texturée au bord n'est pas une bande uniforme.
    textured = gray.copy()
    textured[:, :15] = np.tile([0, 40], (60, 8))[:, :15].astype(np.uint8)
    assert letterbox_mask(textured, 20.0, 4.0)[:, :15].any()
    assert not letterbox_mask(np.zeros((10, 10), np.uint8), 20.0, 4.0).any()
