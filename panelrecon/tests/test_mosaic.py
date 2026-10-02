"""Tests de la mosaïque : médiane pondérée, masques, canevas, recadrage, et tests de
référence de bout en bout (recalage phase 3 + fusion) contre le panel original."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from panelrecon.core.config import PipelineConfig
from panelrecon.core.evaluation import ReferenceThresholds, evaluate_mosaic
from panelrecon.core.models import FrameObs, MosaicResult, SimilarityTransform
from panelrecon.core.mosaic import (
    CanvasTooLargeError,
    build_mosaic,
    crop_box,
    edge_weight,
    plan_canvas,
    scale_weight,
    select_observations,
    validity_mask,
    weighted_median,
)
from panelrecon.tests.conftest import RegistrationCache

THRESHOLDS = ReferenceThresholds()


# ------------------------------------------------------------- médiane pondérée


def _reference_weighted_median(values: NDArray[Any], weights: NDArray[Any]) -> float:
    keep = weights > 0
    v, w = values[keep].astype(float), weights[keep].astype(float)
    if v.size == 0:
        return float("nan")
    order = np.argsort(v, kind="stable")
    v, w = v[order], w[order]
    cum = np.cumsum(w)
    half = cum[-1] / 2
    lo = v[np.argmax(cum >= half - 1e-9 * cum[-1])]
    hi = v[np.argmax(cum > half + 1e-9 * cum[-1])]
    return float(0.5 * (lo + hi))


def test_weighted_median_matches_reference() -> None:
    rng = np.random.default_rng(0)
    values = rng.integers(0, 256, (9, 6, 7, 3), dtype=np.uint8)
    weights = rng.uniform(0, 1, (9, 6, 7)).astype(np.float32)
    weights[rng.random(weights.shape) < 0.3] = 0.0
    weights[:, 0, 0] = 0.0
    out = weighted_median(values, weights)
    assert np.isnan(out[0, 0]).all()
    for y in range(6):
        for x in range(7):
            for c in range(3):
                expected = _reference_weighted_median(values[:, y, x, c], weights[:, y, x])
                if np.isnan(expected):
                    assert np.isnan(out[y, x, c])
                else:
                    assert out[y, x, c] == pytest.approx(expected)


@pytest.mark.parametrize("n", [1, 2, 5, 6])
def test_equal_weights_give_usual_median(n: int) -> None:
    rng = np.random.default_rng(n)
    values = rng.integers(0, 256, (n, 4, 5, 3), dtype=np.uint8)
    out = weighted_median(values, np.ones((n, 4, 5), np.float32))
    np.testing.assert_allclose(out, np.median(values.astype(np.float32), axis=0))


def test_weighted_median_rejects_minority_outlier() -> None:
    values = np.array([10, 12, 11, 250], np.uint8).reshape(4, 1, 1, 1)
    weights = np.ones((4, 1, 1), np.float32)
    assert weighted_median(values, weights)[0, 0, 0] == pytest.approx(11.5)
    weights[0] = 10.0  # une observation dominante l'emporte
    assert weighted_median(values, weights)[0, 0, 0] == pytest.approx(10.0)


# ------------------------------------------------------------- masques et poids


def _frame(h: int = 60, w: int = 80) -> FrameObs:
    # Gris moyen : une frame entièrement noire serait (à juste titre) une bande letterbox.
    return FrameObs(0, 0.0, 0, np.full((h, w, 3), 128, np.uint8))


def test_validity_mask() -> None:
    cfg = PipelineConfig()
    cfg.mosaic.frame_border_px = 2
    cfg.mosaic.mask_erode_px = 0
    cfg.preprocess.exclusion_zones = ((0.0, 0.5, 1.0, 1.0),)
    mask = validity_mask(_frame(), cfg)
    assert not mask[:2].any() and not mask[:, -2:].any() and not mask[30:].any()
    assert mask[2:30, 2:78].all()
    panel = np.zeros((60, 80), np.uint8)
    panel[:, :40] = 255
    cfg.mosaic.mask_erode_px = 3
    eroded = validity_mask(_frame(), cfg, panel)
    assert not eroded[:, 37:].any() and eroded[10:20, 10:30].all()
    assert not eroded[:5].any()  # bord d'écran (2) + érosion (3)
    with pytest.raises(ValueError):
        validity_mask(_frame(), cfg, np.zeros((5, 5), np.uint8))


def test_edge_and_scale_weights() -> None:
    mask = np.zeros((50, 50), np.uint8)
    mask[10:40, 10:40] = 255
    w = edge_weight(mask, feather_px=10.0, min_weight=0.05)
    assert w[0, 0] == 0.0 and w[25, 25] == pytest.approx(1.0)
    assert 0.05 <= w[10, 25] < w[15, 25] < w[25, 25]
    assert np.array_equal(edge_weight(mask, 0.0, 0.05) > 0, mask > 0)
    assert scale_weight(SimilarityTransform(1.0), 2.0) == 1.0
    assert scale_weight(SimilarityTransform(2.0), 2.0) == pytest.approx(0.25)
    assert scale_weight(SimilarityTransform(2.0), 0.0) == 1.0


def test_plan_canvas_and_limit() -> None:
    region = np.array([[0.0, 0.0], [99.0, 0.0], [99.0, 49.0], [0.0, 49.0]])
    transforms = {0: SimilarityTransform(), 1: SimilarityTransform.from_translation(-30.0, 20.0)}
    canvas = plan_canvas(transforms, {0: region, 1: region}, 10.0)
    corners_canvas = np.vstack([(canvas.offset @ t).apply(region) for t in transforms.values()])
    assert corners_canvas.min() >= 0
    assert corners_canvas[:, 0].max() <= canvas.width - 1
    assert corners_canvas[:, 1].max() <= canvas.height - 1
    assert canvas.width <= 133 and canvas.height <= 73
    with pytest.raises(CanvasTooLargeError, match="max_canvas_megapixels"):
        plan_canvas(transforms, {0: region, 1: region}, 0.001)


def test_crop_box_modes() -> None:
    covered = np.zeros((40, 60), bool)
    covered[5:35, 10:50] = True
    covered[5:8, 10:20] = False  # encoche
    covered[20, 55] = True  # pixel isolé
    bbox = crop_box(covered, "bbox")
    assert (bbox.x0, bbox.y0, bbox.x1, bbox.y1) == (10, 5, 56, 35)
    tight = crop_box(covered, "covered")
    assert covered[tight.y0 : tight.y1, tight.x0 : tight.x1].all()
    assert tight.width * tight.height >= 0.8 * 40 * 30
    with pytest.raises(ValueError):
        crop_box(np.zeros((5, 5), bool), "bbox")
    with pytest.raises(ValueError):
        crop_box(covered, "autre")


# ------------------------------------------------------- tests de référence


def _mosaic(registered: RegistrationCache, name: str, shot_id: int = 0, with_masks: bool = True,
            config: PipelineConfig | None = None) -> MosaicResult:
    gt = registered.synthetic.get(name)
    registration = registered.get(name, shot_id, with_masks)
    masks = registered.masks(gt) if with_masks else None
    return build_mosaic(registered.frames(gt, shot_id, proxy=False), registration,
                        config or PipelineConfig(), masks)


@pytest.mark.parametrize(
    ("name", "shot_id"),
    [("static", 0), ("pan_horizontal", 0), ("pan_vertical", 0), ("zoom_in", 0),
     ("zoom_out", 0), ("pan_zoom_eased", 0), ("duplicates", 0), ("crossfade", 0),
     ("crossfade", 1), ("subtitles", 0), ("flat_texture", 0), ("short", 0), ("vfr", 0)],
)
def test_reconstruction_meets_reference(
    registered: RegistrationCache, name: str, shot_id: int
) -> None:
    """Recalage estimé + fusion, masques du panel exacts : SSIM > 0,95, couverture juste."""
    gt = registered.synthetic.get(name)
    result = _mosaic(registered, name, shot_id)
    metrics = evaluate_mosaic(result, gt, shot_id)
    assert metrics.check(THRESHOLDS) == [], metrics
    assert np.array_equal(result.image_bgra[..., 3] > 0, result.coverage >= 1)
    assert result.coverage.max() <= len(result.transforms)
    assert not result.image_bgra[result.image_bgra[..., 3] == 0, :3].any()


@pytest.mark.parametrize("name", ["pan_horizontal", "duplicates", "flat_texture"])
def test_reconstruction_without_masks_when_panel_fills_screen(
    registered: RegistrationCache, name: str
) -> None:
    gt = registered.synthetic.get(name)
    metrics = evaluate_mosaic(_mosaic(registered, name, with_masks=False), gt, 0)
    assert metrics.check(THRESHOLDS) == []


def test_median_removes_static_subtitles(registered: RegistrationCache) -> None:
    """Sans zone d'exclusion ni masque, le sous-titre fixe est rejeté par la médiane
    là où le panel est vu sous plusieurs positions."""
    gt = registered.synthetic.get("subtitles")
    metrics = evaluate_mosaic(_mosaic(registered, "subtitles", with_masks=False), gt, 0)
    assert metrics.ssim > THRESHOLDS.min_ssim


def test_scale_weighting_favours_zoomed_frames(registered: RegistrationCache) -> None:
    gt = registered.synthetic.get("zoom_out")
    weighted = evaluate_mosaic(_mosaic(registered, "zoom_out"), gt, 0)
    cfg = PipelineConfig()
    cfg.mosaic.scale_weight_power = 0.0
    flat = evaluate_mosaic(_mosaic(registered, "zoom_out", config=cfg), gt, 0)
    assert weighted.ssim > flat.ssim + 0.002


def test_disk_stack_and_tiling_are_equivalent(registered: RegistrationCache, tmp_path: Path) -> None:
    reference = _mosaic(registered, "short")
    cfg = PipelineConfig()
    cfg.mosaic.in_memory_stack_mb = 0.0
    cfg.mosaic.temp_dir = str(tmp_path)
    cfg.mosaic.tile_size = 37
    other = _mosaic(registered, "short", config=cfg)
    assert np.array_equal(reference.image_bgra, other.image_bgra)
    assert np.array_equal(reference.coverage, other.coverage)
    assert list(tmp_path.iterdir()) == []  # dossier temporaire nettoyé


def test_min_coverage_and_crop_mode(registered: RegistrationCache) -> None:
    base = _mosaic(registered, "zoom_in")
    cfg = PipelineConfig()
    cfg.mosaic.min_coverage = 5
    strict = _mosaic(registered, "zoom_in", config=cfg)
    assert (strict.image_bgra[..., 3] > 0).sum() < (base.image_bgra[..., 3] > 0).sum()
    assert strict.coverage[strict.image_bgra[..., 3] > 0].min() >= 5
    cfg = PipelineConfig()
    cfg.mosaic.crop_mode = "covered"
    full = _mosaic(registered, "zoom_in", config=cfg)
    assert (full.image_bgra[..., 3] == 255).all()


def test_errors(registered: RegistrationCache) -> None:
    gt = registered.synthetic.get("short")
    registration = registered.get("short")
    cfg = PipelineConfig()
    cfg.mosaic.max_canvas_megapixels = 0.01
    with pytest.raises(CanvasTooLargeError):
        build_mosaic(registered.frames(gt, 0, False), registration, cfg)
    with pytest.raises(ValueError, match="absentes"):
        build_mosaic(iter(()), registration, PipelineConfig())
    with pytest.raises(ValueError):
        build_mosaic(iter(()), replace(registration, transforms={}), PipelineConfig())


def test_select_observations_keeps_coverage() -> None:
    """60 frames qui défilent : la sélection borne les observations par zone sans
    laisser de zone non couverte, et préfère les frames les plus zoomées."""
    canvas = plan_canvas({0: SimilarityTransform()}, {0: np.array([[0.0, 0.0], [999.0, 299.0]])},
                         10.0)
    transforms = {i: SimilarityTransform.from_translation(10.0 * i, 0.0) for i in range(60)}
    transforms[30] = SimilarityTransform(0.8, 0.0, 300.0, 0.0)  # frame la plus zoomée
    sizes = {i: (400, 300) for i in transforms}
    chosen = select_observations(transforms, sizes, canvas, 4, 16, 2.0)
    assert 30 in chosen and len(chosen) < 30

    def coverage(indices: list[int]) -> NDArray[np.int32]:
        cov = np.zeros((canvas.height, canvas.width), np.int32)
        for i in indices:
            x0 = int(max(0, transforms[i].tx + canvas.offset.tx))
            x1 = int(min(canvas.width, x0 + 400 * transforms[i].scale))
            cov[:, x0:x1] += 1
        return cov

    full, kept = coverage(sorted(transforms)), coverage(chosen)
    assert np.array_equal(full > 0, kept > 0)
    # Borne indicative (~K) : une frame retenue pour une zone en manque compte aussi
    # pour ses autres zones, d'où un dépassement local, mais la réduction est forte.
    assert np.median(kept[kept > 0]) <= 2 * 4
    assert kept.max() <= full.max() // 2
    assert select_observations(transforms, sizes, canvas, 0, 16, 2.0) == sorted(transforms)
