from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from panelrecon.core.config import PipelineConfig
from panelrecon.core.geometry import corners, coverage_alpha, full_coverage_mask, warp_mask
from panelrecon.core.models import ImageU8
from panelrecon.core.synthetic import (
    EASINGS,
    SCENARIOS,
    BackgroundSpec,
    GroundTruth,
    PanelSpec,
    Pose,
    ShotSpec,
    VideoSpec,
    generate_video,
    interpolate_pose,
    make_panel,
    render_video,
    scenario,
    subtitle_overlay,
)
from panelrecon.core.video_io import VideoReader
from panelrecon.tests.conftest import SyntheticCache

# ----------------------------------------------------------------------------- briques


@pytest.mark.parametrize("name", sorted(EASINGS))
def test_easing_endpoints_and_monotonicity(name: str) -> None:
    f = EASINGS[name]
    assert f(0.0) == pytest.approx(0.0, abs=1e-12) and f(1.0) == pytest.approx(1.0, abs=1e-12)
    values = [f(u) for u in np.linspace(0, 1, 101)]
    assert all(b >= a - 1e-12 for a, b in zip(values, values[1:]))


def test_pose_transform_maps_center_to_screen_center() -> None:
    pose = Pose(0.7, 400.0, 250.0, theta=0.1)
    t = pose.to_transform(640, 360)
    np.testing.assert_allclose(t.apply(np.array([[400.0, 250.0]])), [[319.5, 179.5]], atol=1e-9)
    assert t.scale == pytest.approx(0.7) and t.theta == pytest.approx(0.1)


def test_interpolation_and_easing() -> None:
    a, b = Pose(0.5, 0.0, 0.0), Pose(2.0, 100.0, 50.0)
    mid = interpolate_pose(a, b, 0.5)
    assert mid.scale == pytest.approx(1.0)  # moyenne géométrique
    assert (mid.center_u, mid.center_v) == pytest.approx((50.0, 25.0))
    linear = ShotSpec(PanelSpec(), ((0.0, a), (1.0, b)), n_poses=11)
    eased = replace(linear, easing="ease_in_out")
    assert linear.pose_at(0) == a and linear.pose_at(10) == b
    assert eased.pose_at(0) == a and eased.pose_at(10).scale == pytest.approx(b.scale)
    # L'easing ralentit le début : à 20 % du temps, moins de chemin parcouru.
    assert eased.pose_at(2).center_u < linear.pose_at(2).center_u
    # Trois poses clés : passage exact par la pose intermédiaire.
    c = Pose(1.0, 30.0, 30.0)
    three = ShotSpec(PanelSpec(), ((0.0, a), (0.5, c), (1.0, b)), n_poses=5)
    assert three.pose_at(2).center_u == pytest.approx(30.0)


def test_spec_validation() -> None:
    with pytest.raises(ValueError):
        ShotSpec(PanelSpec(), (), n_poses=3)
    with pytest.raises(ValueError):
        ShotSpec(PanelSpec(), ((0.0, Pose(1, 0, 0)), (0.5, Pose(1, 0, 0))), n_poses=3)
    with pytest.raises(ValueError):
        ShotSpec(PanelSpec(), ((0.0, Pose(1, 0, 0)),), n_poses=3, easing="bounce")
    with pytest.raises(ValueError):
        PanelSpec(texture="noise")
    with pytest.raises(ValueError):
        BackgroundSpec(mode="video")
    with pytest.raises(ValueError):
        Pose(0.0, 0, 0)
    with pytest.raises(ValueError):
        scenario("inconnu")
    for name in SCENARIOS:
        assert scenario(name).name == name


def test_panels_are_deterministic_and_textures_differ() -> None:
    a = make_panel(PanelSpec(600, 400, "rich", seed=3))
    b = make_panel(PanelSpec(600, 400, "rich", seed=3))
    c = make_panel(PanelSpec(600, 400, "rich", seed=4))
    flat = make_panel(PanelSpec(600, 400, "flat", seed=3))
    assert a.shape == (400, 600, 3) and np.array_equal(a, b) and not np.array_equal(a, c)

    def detail(img: ImageU8) -> float:
        return float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())

    assert detail(flat) < 0.05 * detail(a)  # aplats : très peu de texture
    sift = cv2.SIFT.create()
    assert len(sift.detect(a, None)) > 10 * max(1, len(sift.detect(flat, None)))


def test_panel_from_image(tmp_path: Path) -> None:
    image = np.random.default_rng(0).integers(0, 256, (50, 80, 3), dtype=np.uint8)
    path = tmp_path / "p.png"
    cv2.imwrite(str(path), image)
    assert np.array_equal(make_panel(PanelSpec(image_path=path)), image)
    with pytest.raises(ValueError):
        make_panel(PanelSpec(image_path=tmp_path / "absent.png"))


# ------------------------------------------------------- exactitude géométrique


def _blob_panel(path: Path, points: NDArray[np.float64], size: tuple[int, int], sigma: float) -> None:
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    darkness = np.zeros((h, w))
    for u, v in points:
        darkness += np.exp(-((xx - u) ** 2 + (yy - v) ** 2) / (2 * sigma**2))
    image = np.clip(np.rint(200.0 - 150.0 * darkness), 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), np.dstack([image] * 3))


@pytest.mark.parametrize(
    "pose",
    [Pose(0.45, 300.0, 200.0), Pose(1.0, 310.3, 190.7), Pose(1.7, 280.0, 210.0),
     Pose(0.8, 300.0, 200.0, theta=0.12)],
    ids=["reduction", "identite", "agrandissement", "rotation"],
)
def test_ground_truth_matches_rendered_blobs(tmp_path: Path, pose: Pose) -> None:
    """Vérification indépendante du warp : le centroïde de chaque tache gaussienne
    du panel doit tomber à ``T(p)`` à mieux que 0,05 px."""
    rng = np.random.default_rng(1)
    points = np.column_stack([rng.uniform(200, 420, 12), rng.uniform(120, 280, 12)])
    points = points[np.argsort(points[:, 0])]
    keep = [0]
    for i in range(1, len(points)):  # taches bien séparées
        if np.min(np.linalg.norm(points[keep] - points[i], axis=1)) > 40:
            keep.append(i)
    points = points[keep]
    sigma = 4.0
    panel_path = tmp_path / "blobs.png"
    _blob_panel(panel_path, points, (600, 400), sigma)
    spec = VideoSpec(
        "blobs",
        (ShotSpec(PanelSpec(image_path=panel_path), ((0.0, pose),), n_poses=1),),
        background=BackgroundSpec(mode="solid", color=(200, 200, 200)),
    )
    image, truth = next(render_video(spec))
    assert truth.transform is not None
    gray = 200.0 - image[..., 0].astype(np.float64)
    expected = truth.transform.apply(points)
    radius = int(math.ceil(4 * sigma * pose.scale)) + 2
    checked = 0
    for x, y in expected:
        x0, x1 = int(round(x)) - radius, int(round(x)) + radius + 1
        y0, y1 = int(round(y)) - radius, int(round(y)) + radius + 1
        if x0 < 0 or y0 < 0 or x1 > image.shape[1] or y1 > image.shape[0]:
            continue
        window = np.clip(gray[y0:y1, x0:x1], 0, None)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        cx = float((window * xx).sum() / window.sum())
        cy = float((window * yy).sum() / window.sum())
        assert math.hypot(cx - x, cy - y) < 0.05, (cx, cy, x, y)
        checked += 1
    assert checked >= 3


def test_coverage_mask_matches_polygon() -> None:
    t = Pose(0.6, 500.0, 300.0, theta=0.05).to_transform(640, 360)
    mask = full_coverage_mask((1000, 800), t, (640, 360))
    alpha = coverage_alpha((1000, 800), t, (640, 360))
    quad = t.apply(corners(1000, 800)).astype(np.float32)
    yy, xx = np.mgrid[0:360, 0:640]
    pts = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float32)
    dist = np.array([cv2.pointPolygonTest(quad, (float(px), float(py)), True) for px, py in
                     pts[:: 97]])
    sampled = mask.ravel()[:: 97]
    assert np.all(sampled[dist > 1.5] == 255)
    assert np.all(sampled[dist < -1.0] == 0)
    assert alpha.min() >= 0.0 and alpha.max() <= 1.0
    # Masque transporté : jamais plus grand que le support réel.
    back = warp_mask(mask, t.inverse(), (1000, 800))
    assert back.sum() > 0 and np.all(back[0, :] == 0) or np.all(back[:, 0] == 0)


# ------------------------------------------- génération → encodage → décodage


def _decode(gt: GroundTruth) -> list[tuple[int, float, ImageU8]]:
    cfg = PipelineConfig()
    with VideoReader(gt.video_path, cfg.video, cfg.preprocess) as reader:
        return [(f.index, f.time_s, f.image) for f in reader.frames(compute_proxy=False)]


@pytest.mark.parametrize("name", ["pan_zoom_eased", "vfr"])
def test_encoded_video_matches_render(synthetic: SyntheticCache, name: str) -> None:
    gt = synthetic.get(name)
    decoded = _decode(gt)
    rendered = list(render_video(scenario(name)))
    assert len(decoded) == len(gt.frames) == len(rendered)
    np.testing.assert_allclose([t for _, t, _ in decoded], [f.time_s for f in gt.frames],
                               atol=1.5e-3)
    for (_, _, dec), (ren, _) in zip(decoded[::7], rendered[::7]):
        mse = np.mean((dec.astype(float) - ren.astype(float)) ** 2)
        assert 10 * math.log10(255**2 / mse) > 32.0
    if name == "vfr":
        intervals = np.diff([f.time_s for f in gt.frames])
        assert intervals.max() > 1.4 * intervals.min()


def test_ground_truth_roundtrip(synthetic: SyntheticCache) -> None:
    gt = synthetic.get("crossfade")
    path = gt.root / "crossfade_ground_truth.json"
    loaded = GroundTruth.load(path)
    assert loaded.frames == gt.frames and loaded.shots == gt.shots
    assert loaded.video_path == gt.video_path and loaded.video_path.is_file()
    assert np.array_equal(loaded.load_panel(1), make_panel(scenario("crossfade").shots[1].panel))


def test_crossfade_truth(synthetic: SyntheticCache) -> None:
    gt = synthetic.get("crossfade")
    seqs = gt.expected_sequences()
    assert [(s.start_idx, s.end_idx) for s in seqs] == [(0, 19), (26, 45)]
    transition = [f for f in gt.frames if f.shot_id is None]
    assert [f.index for f in transition] == list(range(20, 26))
    for f in transition:
        assert f.transform is None and sum(w for _, w in f.blend) == pytest.approx(1.0)
        assert not gt.visibility_mask(f).any()
    weights_b = [dict(f.blend)[1] for f in transition]
    assert all(b > a for a, b in zip(weights_b, weights_b[1:]))


def test_duplicates_truth(synthetic: SyntheticCache) -> None:
    gt = synthetic.get("duplicates")
    flags = [f.is_duplicate for f in gt.frames]
    assert flags == [False, True] * 15
    decoded = _decode(gt)
    first, second = decoded[4][2].astype(float), decoded[5][2].astype(float)
    third = decoded[6][2].astype(float)
    assert np.abs(first - second).mean() < 0.25 * np.abs(second - third).mean()
    assert gt.frames[4].transform == gt.frames[5].transform


def test_subtitles_are_masked(synthetic: SyntheticCache) -> None:
    gt = synthetic.get("subtitles")
    assert gt.subtitle_box is not None
    x0, y0, x1, y1 = gt.subtitle_box
    assert 0 <= x0 < x1 <= gt.screen_width and gt.screen_height // 2 < y0 < y1
    mask = gt.visibility_mask(gt.frames[0])
    assert not mask[y0:y1, x0:x1].any() and mask[: y0 - 1].any()
    # Le sous-titre est identique d'une frame à l'autre (élément fixe parasite).
    decoded = _decode(gt)
    spec = scenario("subtitles").subtitle
    assert spec is not None
    layer, alpha, _ = subtitle_overlay(spec, gt.screen_width, gt.screen_height)
    core = (alpha[..., 0] > 0.99) & (layer.min(axis=2) > 250)
    assert core.sum() > 100
    for _, _, image in (decoded[0], decoded[len(decoded) // 2], decoded[-1]):
        assert image[core].mean() > 200  # texte blanc, quelle que soit la position du panel


def test_static_panel_fully_visible(synthetic: SyntheticCache) -> None:
    gt = synthetic.get("static")
    panel = gt.shots[0]
    t = gt.frames[0].transform
    assert t is not None
    quad = t.apply(corners(panel.panel_width, panel.panel_height))
    assert quad.min() > 0 and quad[:, 0].max() < gt.screen_width - 1
    assert quad[:, 1].max() < gt.screen_height - 1
    assert len({f.transform for f in gt.frames}) == 1


def test_partial_visibility_in_moving_scenarios(synthetic: SyntheticCache) -> None:
    """Dans les scénarios animés, aucune frame ne montre le panel en entier."""
    for name in ("pan_horizontal", "zoom_in", "pan_vertical"):
        gt = synthetic.get(name)
        shot = gt.shots[0]
        panel_area = shot.panel_width * shot.panel_height
        for f in gt.frames:
            assert f.transform is not None
            visible_panel_px = (gt.visibility_mask(f) > 0).sum() / f.transform.scale**2
            assert visible_panel_px < 0.9 * panel_area


def test_generate_short_and_flat(tmp_path: Path) -> None:
    gt = generate_video(scenario("short"), tmp_path)
    assert len(gt.frames) == 4 and gt.expected_sequences()[0].end_idx == 3
    assert (tmp_path / "short_ground_truth.json").is_file()
    assert (tmp_path / "short_panel0.png").is_file()
