"""Segmentation panel / fond.

Deux niveaux :

* :class:`ClassicSegmenter` (interface :class:`PanelSegmenter`, image par image,
  utilisable sans recalage, par ex. pour la prévisualisation) :
  netteté locale (variance du Laplacien), cohérence avec le mouvement du panel
  si celui-ci est connu, nettoyage morphologique, plus grande composante,
  rectangle orienté (``cv2.minAreaRect``) et lissage temporel du rectangle.

* :class:`PanelRegionEstimator` (niveau séquence, utilisé par le pipeline) : le
  panel étant rigide et recalé, son emprise dans le repère canonique est un
  rectangle **fixe**. Les preuves de toutes les frames y sont cumulées :

  - netteté : un point régulièrement net appartient au panel (le fond est flou) ;
  - cohérence temporelle : un point du canevas vu à des positions d'écran
    différentes garde sa valeur s'il appartient au panel ; un point de fond
    (immobile à l'écran ou animé autrement) change de valeur.

  Le rectangle de départ est la boîte de la plus grande composante des points
  régulièrement nets. Chaque côté est ensuite étendu jusqu'à la limite observée
  (panel plus grand que l'écran, aplats sans détail), sauf s'il porte une
  bordure (ligne nette continue : limite panel/fond) ou si la bande au-delà se
  comporte comme du fond (variation temporelle nettement supérieure à celle des
  zones plates du panel). Les masques par frame en sont la projection, ce qui
  garantit des contours cohérents dans le temps.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Protocol

import cv2
import numpy as np
from numpy.typing import NDArray

from panelrecon.core.config import PipelineConfig, SegmentationConfig
from panelrecon.core.geometry import corners
from panelrecon.core.models import FrameObs, ImageU8, MaskU8, SimilarityTransform
from panelrecon.core.video_io import compute_proxy_factor, letterbox_mask, make_proxy, to_gray

logger = logging.getLogger(__name__)

F32 = NDArray[np.float32]


# ---------------------------------------------------------------------------
# Netteté
# ---------------------------------------------------------------------------


def local_sharpness(gray: ImageU8, window: int) -> F32:
    """Variance locale du Laplacien sur une fenêtre ``window × window``."""
    lap = np.asarray(cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F, ksize=3), np.float32)
    mean = cv2.boxFilter(lap, -1, (window, window), normalize=True)
    mean_sq = cv2.boxFilter(lap * lap, -1, (window, window), normalize=True)
    return np.maximum(np.asarray(mean_sq - mean * mean, dtype=np.float32), 0.0)


def sharp_evidence(gray: ImageU8, cfg: SegmentationConfig) -> NDArray[np.bool_]:
    """Pixels nets : variance du Laplacien au-dessus d'un seuil absolu et relatif."""
    sharp = local_sharpness(gray, cfg.sharpness_window)
    threshold = max(cfg.sharpness_abs_threshold,
                    cfg.sharpness_rel_threshold * float(np.percentile(sharp, 99)))
    return np.asarray(sharp > threshold)


def _largest_component(mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return np.zeros_like(mask)
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.asarray(labels == best)


def _dilate(mask: NDArray[np.bool_], radius: int) -> NDArray[np.bool_]:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * radius + 1, 2 * radius + 1))
    return np.asarray(cv2.dilate(mask.astype(np.uint8), kernel) > 0)


def _close(mask: NDArray[np.bool_], radius: int) -> NDArray[np.bool_]:
    if radius <= 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    padded = cv2.copyMakeBorder(mask.astype(np.uint8), radius, radius, radius, radius,
                                cv2.BORDER_CONSTANT, value=0)
    closed = cv2.morphologyEx(padded, cv2.MORPH_CLOSE, kernel)
    return np.asarray(closed[radius:-radius, radius:-radius] > 0)


def polygon_mask(polygon: NDArray[np.float64], size: tuple[int, int]) -> MaskU8:
    """Masque 255 à l'intérieur d'un polygone convexe (coordonnées sous-pixel)."""
    w, h = size
    shift = 4
    pts = np.rint(polygon * (1 << shift)).astype(np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, pts, 255, lineType=cv2.LINE_8, shift=shift)
    return mask


# ---------------------------------------------------------------------------
# Segmenteur image par image
# ---------------------------------------------------------------------------


@dataclass
class SegmenterState:
    """État transmis d'une frame à la suivante.

    ``motion`` (facultatif) : similarité **native** frame précédente → frame
    courante, si elle est connue (recalage) ; elle sert à la cohérence de
    mouvement et à la prédiction du rectangle.
    """

    previous_gray: ImageU8 | None = None
    previous_box: NDArray[np.float64] | None = None  # 4 coins, coordonnées natives
    motion: SimilarityTransform | None = None


class PanelSegmenter(Protocol):
    def segment(self, frame_bgr: ImageU8, state: SegmenterState) -> tuple[MaskU8, SegmenterState]:
        """Masque uint8 (255 = panel) de taille native et nouvel état."""
        ...


class ClassicSegmenter:
    """Segmentation classique image par image (voir le module)."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.cfg = config.segmentation

    def segment(self, frame_bgr: ImageU8, state: SegmenterState) -> tuple[MaskU8, SegmenterState]:
        h, w = frame_bgr.shape[:2]
        factor = compute_proxy_factor(w, h, self.config.preprocess.motion_long_side)
        gray = make_proxy(to_gray(frame_bgr), factor)
        evidence = sharp_evidence(gray, self.cfg)
        motion = state.motion
        if motion is not None and state.previous_gray is not None:
            if state.previous_gray.shape == gray.shape:
                evidence = self._apply_motion_coherence(evidence, gray, state.previous_gray,
                                                        motion.rescaled(1.0 / factor, 1.0 / factor))
        region = _largest_component(_close(evidence, self.cfg.closing_px))
        if not region.any():
            box = corners(w, h) if state.previous_box is None else state.previous_box
        else:
            pts = np.column_stack(np.nonzero(region)[::-1]).astype(np.float32)
            rect = cv2.minAreaRect(pts)
            box = cv2.boxPoints(rect).astype(np.float64) / factor
            box = _order_corners(box)
            box = self._smooth(box, state)
        mask = polygon_mask(box, (w, h))
        return mask, SegmenterState(previous_gray=gray, previous_box=box, motion=None)

    def _apply_motion_coherence(
        self, evidence: NDArray[np.bool_], gray: ImageU8, previous: ImageU8,
        motion: SimilarityTransform,
    ) -> NDArray[np.bool_]:
        """Ajoute les pixels qui suivent le mouvement du panel, retire ceux qui n'en
        suivent pas (immobiles alors que le panel bouge)."""
        h, w = gray.shape
        predicted = cv2.warpAffine(previous, motion.matrix(), (w, h), flags=cv2.INTER_LINEAR)
        valid = cv2.warpAffine(np.ones_like(previous), motion.matrix(), (w, h),
                               flags=cv2.INTER_NEAREST) > 0
        follow = np.abs(predicted.astype(np.int16) - gray.astype(np.int16))
        still = np.abs(previous.astype(np.int16) - gray.astype(np.int16))
        tol = self.cfg.panel_max_std
        coherent = valid & (follow <= tol) & (still > 2 * tol)
        incoherent = valid & (follow > 2 * tol) & (still <= tol)
        return np.asarray((evidence | coherent) & ~incoherent)

    def _smooth(self, box: NDArray[np.float64], state: SegmenterState) -> NDArray[np.float64]:
        if state.previous_box is None:
            return box
        predicted = state.previous_box
        if state.motion is not None:
            predicted = state.motion.apply(state.previous_box)
        size = max(np.ptp(predicted[:, 0]), np.ptp(predicted[:, 1]), 1.0)
        jump = float(np.abs(box - predicted).max()) / size
        if jump > self.cfg.max_rect_jump:
            return predicted
        alpha = self.cfg.temporal_smoothing
        return alpha * predicted + (1.0 - alpha) * box


def _order_corners(box: NDArray[np.float64]) -> NDArray[np.float64]:
    """Coins dans l'ordre haut-gauche, haut-droit, bas-droit, bas-gauche."""
    s = box.sum(axis=1)
    d = box[:, 0] - box[:, 1]
    return np.array([box[np.argmin(s)], box[np.argmax(d)], box[np.argmax(s)], box[np.argmin(d)]])


# ---------------------------------------------------------------------------
# Estimation de l'emprise du panel sur une séquence
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PanelRegion:
    """Emprise du panel : rectangle (4 coins) dans le repère canonique natif."""

    polygon: NDArray[np.float64]
    transforms: dict[int, SimilarityTransform] = field(repr=False)
    segmented: bool = True

    def mask(self, frame: FrameObs) -> MaskU8:
        """Masque natif du panel dans ``frame``."""
        to_frame = self.transforms[frame.index].inverse()
        corners_native = to_frame.apply(self.polygon)
        if not frame.is_native:  # image réduite : coordonnées natives → image
            corners_native = corners_native * (frame.width / float(frame.native_width))
        return polygon_mask(corners_native, (frame.width, frame.height))

    def bounds(self) -> tuple[float, float, float, float]:
        return (float(self.polygon[:, 0].min()), float(self.polygon[:, 1].min()),
                float(self.polygon[:, 0].max()), float(self.polygon[:, 1].max()))


class PanelRegionEstimator:
    """Accumule les preuves des frames d'une séquence dans un canevas d'analyse
    (repère canonique réduit) puis estime le rectangle du panel."""

    def __init__(
        self,
        config: PipelineConfig,
        transforms: dict[int, SimilarityTransform],
        frame_sizes: dict[int, tuple[int, int]],
        analysis_factor: float,
    ) -> None:
        self.cfg = config.segmentation
        self.config = config
        self.transforms = transforms
        self.factor = analysis_factor
        scale = SimilarityTransform(scale=analysis_factor)
        pts = np.vstack([(scale @ transforms[i]).apply(corners(*frame_sizes[i]))
                         for i in transforms])
        x0, y0 = np.floor(pts.min(axis=0)) - 1
        x1, y1 = np.ceil(pts.max(axis=0)) + 1
        self.width, self.height = int(x1 - x0) + 1, int(y1 - y0) + 1
        limit = config.mosaic.max_canvas_megapixels * 1e6
        if self.width * self.height > limit:
            raise MemoryError(
                f"Canevas d'analyse {self.width}x{self.height} > mosaic.max_canvas_megapixels"
            )
        self.to_analysis = SimilarityTransform.from_translation(-x0, -y0) @ scale
        shape = (self.height, self.width)
        self.count = np.zeros(shape, np.uint16)
        self.hits = np.zeros(shape, np.uint16)
        self.coarse = np.zeros(shape, np.uint16)
        self.total = np.zeros(shape, np.float32)
        self.total_sq = np.zeros(shape, np.float32)
        self.x_min = np.full(shape, np.inf, np.float32)
        self.x_max = np.full(shape, -np.inf, np.float32)
        self.y_min = np.full(shape, np.inf, np.float32)
        self.y_max = np.full(shape, -np.inf, np.float32)

    def add(self, frame: FrameObs) -> None:
        if frame.index not in self.transforms:
            return
        if frame.proxy_gray is None:
            raise ValueError("Image réduite requise pour la segmentation")
        gray = frame.proxy_gray
        f = frame.proxy_factor
        h, w = gray.shape
        # Frame réduite → canevas d'analyse.
        m = self.to_analysis @ self.transforms[frame.index] @ SimilarityTransform(scale=1.0 / f)
        quad = m.apply(corners(w, h))
        x0 = max(0, int(math.floor(quad[:, 0].min())) - 1)
        y0 = max(0, int(math.floor(quad[:, 1].min())) - 1)
        x1 = min(self.width, int(math.ceil(quad[:, 0].max())) + 2)
        y1 = min(self.height, int(math.ceil(quad[:, 1].max())) + 2)
        local = SimilarityTransform.from_translation(-x0, -y0) @ m
        size = (x1 - x0, y1 - y0)
        mat = local.matrix()
        observable: MaskU8 = np.full((h, w), 255, np.uint8)
        if self.config.preprocess.letterbox_detection:
            observable = letterbox_mask(gray, self.config.preprocess.letterbox_max_level,
                                        self.config.preprocess.letterbox_max_std)
        footprint = cv2.warpAffine(observable, mat, size,
                                   flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=0)
        valid: NDArray[np.bool_] = np.asarray(
            cv2.erode(footprint, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT,
                      borderValue=0) > 0)
        values = np.asarray(cv2.warpAffine(gray.astype(np.float32), mat, size,
                                           flags=cv2.INTER_LINEAR), dtype=np.float32)
        # Une preuve de netteté ne compte que si toute sa fenêtre de mesure est dans
        # la zone observable (le contraste avec une bande noire n'est pas du panel).
        reach = self.cfg.sharpness_window // 2 + 2
        inner = cv2.erode(observable, np.ones((2 * reach + 1, 2 * reach + 1), np.uint8),
                          borderType=cv2.BORDER_REPLICATE) > 0
        sharp_local = sharp_evidence(gray, self.cfg) & inner
        sharp: NDArray[np.bool_] = np.asarray(
            cv2.warpAffine(sharp_local.astype(np.uint8), mat, size, flags=cv2.INTER_NEAREST) > 0)
        smooth = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), self.cfg.blur_gradient_sigma)
        coarse_mag = cv2.magnitude(cv2.Sobel(smooth, cv2.CV_32F, 1, 0),
                                   cv2.Sobel(smooth, cv2.CV_32F, 0, 1))
        coarse: NDArray[np.bool_] = np.asarray(
            cv2.warpAffine((coarse_mag > self.cfg.blur_min_gradient).astype(np.uint8), mat, size,
                           flags=cv2.INTER_NEAREST) > 0)
        # Position à l'écran (px réduits) de chaque point du canevas.
        inv = local.inverse().matrix()
        gy, gx = np.mgrid[0 : size[1], 0 : size[0]].astype(np.float32)
        sx = inv[0, 0] * gx + inv[0, 1] * gy + inv[0, 2]
        sy = inv[1, 0] * gx + inv[1, 1] * gy + inv[1, 2]
        sl = (slice(y0, y1), slice(x0, x1))
        self.count[sl] += valid
        self.hits[sl] += valid & sharp
        self.coarse[sl] += valid & coarse
        self.total[sl] += np.where(valid, values, 0)
        self.total_sq[sl] += np.where(valid, values * values, 0)
        self.x_min[sl] = np.where(valid, np.minimum(self.x_min[sl], sx), self.x_min[sl])
        self.x_max[sl] = np.where(valid, np.maximum(self.x_max[sl], sx), self.x_max[sl])
        self.y_min[sl] = np.where(valid, np.minimum(self.y_min[sl], sy), self.y_min[sl])
        self.y_max[sl] = np.where(valid, np.maximum(self.y_max[sl], sy), self.y_max[sl])

    # ------------------------------------------------------------------ cartes
    def statistics(self) -> tuple[NDArray[np.bool_], NDArray[np.bool_], F32, NDArray[np.bool_]]:
        """(observé, net de façon répétée, écart-type temporel, déplacé à l'écran)."""
        observed = self.count > 0
        n = np.maximum(self.count, 1).astype(np.float32)
        mean = self.total / n
        std = np.sqrt(np.maximum(self.total_sq / n - mean * mean, 0.0)).astype(np.float32)
        span = np.where(observed, (self.x_max - self.x_min) + (self.y_max - self.y_min), 0.0)
        moved = observed & (self.count >= 2) & (span >= self.cfg.min_displacement_px)
        sharp = observed & (self.hits >= 1) & (self.hits >= self.cfg.min_sharp_ratio * self.count)
        return observed, sharp, std, moved

    def estimate(self) -> PanelRegion:
        observed, sharp, std, moved = self.statistics()
        component = _largest_component(_close(sharp, self.cfg.closing_px))
        if not component.any():
            logger.warning("Aucune preuve de panel : la frame entière est utilisée")
            ys, xs = np.nonzero(observed)
            box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
            return self._region(box, segmented=False)
        ys, xs = np.nonzero(component)
        box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        flat_reference = self._flat_panel_std(box, sharp, std, moved)
        closed = [True, True, True, True]
        for _ in range(2):  # les extensions d'un axe modifient l'étendue de l'autre
            box, opened = self._open_sides(box, observed, sharp, std, moved, flat_reference)
            closed = [c and not o for c, o in zip(closed, opened)]
        box = self._refine_closed_sides(box, closed)
        return self._region(box, segmented=True)

    def _refine_closed_sides(
        self, box: tuple[int, int, int, int], closed: list[bool]
    ) -> tuple[int, int, int, int]:
        """Place chaque côté fermé sur le maximum de gradient de l'image moyenne :
        la preuve de netteté déborde d'une demi-fenêtre au-delà de la vraie limite."""
        x0, y0, x1, y1 = box
        n = np.maximum(self.count, 1).astype(np.float32)
        mean = np.where(self.count > 0, self.total / n, 0.0).astype(np.float32)
        gx = np.abs(cv2.Sobel(mean, cv2.CV_32F, 1, 0, ksize=3))
        gy = np.abs(cv2.Sobel(mean, cv2.CV_32F, 0, 1, ksize=3))
        reach = self.cfg.sharpness_window // 2 + self.cfg.boundary_band_px + 2
        h, w = mean.shape

        def peak(profile: F32, lo: int, hi: int) -> int | None:
            lo, hi = max(lo, 1), min(hi, len(profile) - 1)
            if hi <= lo:
                return None
            return lo + int(np.argmax(profile[lo:hi]))

        if closed[0]:
            c = peak(gx[y0:y1].mean(axis=0), x0 - 2, x0 + reach)
            x0 = x0 if c is None else c + 1
        if closed[1]:
            c = peak(gx[y0:y1].mean(axis=0), x1 - reach, x1 + 2)
            x1 = x1 if c is None else c
        if closed[2]:
            c = peak(gy[:, x0:x1].mean(axis=1), y0 - 2, y0 + reach)
            y0 = y0 if c is None else c + 1
        if closed[3]:
            c = peak(gy[:, x0:x1].mean(axis=1), y1 - reach, y1 + 2)
            y1 = y1 if c is None else c
        if x1 - x0 < 2 or y1 - y0 < 2:
            return box
        return max(0, x0), max(0, y0), min(w, x1), min(h, y1)

    def _flat_panel_std(
        self, box: tuple[int, int, int, int], sharp: NDArray[np.bool_], std: F32,
        moved: NDArray[np.bool_],
    ) -> float:
        """Écart-type temporel médian des zones plates du panel (bruit de référence)."""
        x0, y0, x1, y1 = box
        inside = np.zeros_like(sharp)
        inside[y0:y1, x0:x1] = True
        n = np.maximum(self.count, 1).astype(np.float32)
        mean = self.total / n
        grad = cv2.magnitude(cv2.Sobel(mean, cv2.CV_32F, 1, 0), cv2.Sobel(mean, cv2.CV_32F, 0, 1))
        flat = inside & moved & ~_dilate(sharp, 3) & (grad < 4.0)
        return float(np.median(std[flat])) if flat.any() else 0.0

    def _open_sides(
        self, box: tuple[int, int, int, int], observed: NDArray[np.bool_],
        sharp: NDArray[np.bool_], std: F32, moved: NDArray[np.bool_], flat_reference: float,
    ) -> tuple[tuple[int, int, int, int], list[bool]]:
        """Étend chaque côté jusqu'à la limite observée, sauf s'il porte une bordure
        nette ou si la bande au-delà se comporte comme du fond. Renvoie aussi, pour
        chaque côté (gauche, droite, haut, bas), s'il est ouvert (non délimitant)."""
        x0, y0, x1, y1 = box
        band = self.cfg.boundary_band_px
        bg_threshold = max(self.cfg.background_min_std,
                           self.cfg.background_std_ratio * flat_reference)

        def is_border(lines: NDArray[np.bool_]) -> bool:
            # ``lines`` : (positions le long du côté, épaisseur de bande)
            return bool(lines.any(axis=1).mean() >= self.cfg.boundary_min_fraction)

        textured = observed & (self.coarse >= 0.5 * np.maximum(self.count, 1))

        def is_background(region: tuple[slice, slice]) -> bool:
            # Fond animé autrement que le panel : forte variation temporelle.
            values = std[region][moved[region]]
            if values.size >= 16 and float(np.median(values)) > bg_threshold:
                return True
            # Fond flou (agrandissement du panel) : variations à grande échelle sans
            # aucun détail net, même s'il glisse avec le panel.
            obs = observed[region]
            n = int(obs.sum())
            if n < 16:
                return False
            sharp_fraction = float(sharp[region][obs].mean())
            texture_fraction = float(textured[region][obs].mean())
            return (sharp_fraction < self.cfg.blur_max_sharp_fraction
                    and texture_fraction >= self.cfg.blur_min_texture_fraction)

        rows = observed[y0:y1]
        cols = observed[:, x0:x1]
        # Limites observées dans la bande perpendiculaire au côté.
        obs_cols = np.nonzero(rows.any(axis=0))[0]
        obs_rows = np.nonzero(cols.any(axis=1))[0]
        lim_x0, lim_x1 = int(obs_cols.min()), int(obs_cols.max()) + 1
        lim_y0, lim_y1 = int(obs_rows.min()), int(obs_rows.max()) + 1

        opened = [False, False, False, False]
        if lim_x0 < x0 and not is_border(sharp[y0:y1, x0 : x0 + band]) and not is_background(
                (slice(y0, y1), slice(lim_x0, x0))):
            x0, opened[0] = lim_x0, True
        if lim_x1 > x1 and not is_border(sharp[y0:y1, x1 - band : x1]) and not is_background(
                (slice(y0, y1), slice(x1, lim_x1))):
            x1, opened[1] = lim_x1, True
        if lim_y0 < y0 and not is_border(sharp[y0 : y0 + band, x0:x1].T) and not is_background(
                (slice(lim_y0, y0), slice(x0, x1))):
            y0, opened[2] = lim_y0, True
        if lim_y1 > y1 and not is_border(sharp[y1 - band : y1, x0:x1].T) and not is_background(
                (slice(y1, lim_y1), slice(x0, x1))):
            y1, opened[3] = lim_y1, True
        # Un côté déjà à la limite observée n'a rien au-delà : il ne délimite pas le panel.
        opened[0] |= x0 <= lim_x0
        opened[1] |= x1 >= lim_x1
        opened[2] |= y0 <= lim_y0
        opened[3] |= y1 >= lim_y1
        return (x0, y0, x1, y1), opened

    def _region(self, box: tuple[int, int, int, int], segmented: bool) -> PanelRegion:
        x0, y0, x1, y1 = box
        # Bords du rectangle sur les bords de pixels (−0,5 / +0,5 autour des centres).
        rect = np.array([[x0 - 0.5, y0 - 0.5], [x1 - 0.5, y0 - 0.5],
                         [x1 - 0.5, y1 - 0.5], [x0 - 0.5, y1 - 0.5]])
        polygon = self.to_analysis.inverse().apply(rect)
        return PanelRegion(polygon=polygon, transforms=self.transforms, segmented=segmented)


