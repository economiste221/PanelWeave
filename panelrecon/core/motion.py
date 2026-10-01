"""Estimation de la similarité entre deux frames (sur les images réduites).

Cascade, chaque étage n'étant tenté que si le précédent est rejeté :

1. points d'intérêt (SIFT par défaut, ORB en option) restreints au masque du
   panel, appariement kNN + ratio test de Lowe, RANSAC
   (``cv2.estimateAffinePartial2D``), puis moindres carrés sur les inliers avec
   rotation régularisée ;
2. flot optique dense (Farneback) échantillonné sur une grille, similarité
   ajustée par RANSAC sur les vecteurs ;
3. corrélation de phase en espace log-polaire (échelle + rotation), puis
   corrélation de phase classique (translation).

Toute estimation candidate est raffinée au sous-pixel par ECC (initialisée par
le candidat), puis validée : bornes de rotation et d'échelle, recouvrement
minimal et corrélation photométrique (NCC) minimale sur le recouvrement. Une
paire non validée est renvoyée avec ``accepted=False`` et la raison, et
journalisée — jamais acceptée silencieusement.

Convention : ``MotionEstimate.transform`` envoie les coordonnées **réduites** de
la source vers celles de la cible ; :func:`to_native` la convertit en
coordonnées natives.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field, replace
from typing import Any, Final

import cv2
import numpy as np
from numpy.typing import NDArray

from panelrecon.core.config import MotionConfig
from panelrecon.core.geometry import corners
from panelrecon.core.models import (
    FloatArray,
    ImageU8,
    MaskU8,
    MotionEstimate,
    MotionMethod,
    NotASimilarityError,
    SimilarityTransform,
)

logger = logging.getLogger(__name__)

_FLOW_MIN_VECTORS: Final[int] = 20
_ECC_SIMILARITY_TOL: Final[float] = 0.02
_STRETCH_PERCENTILES: Final[tuple[float, float]] = (0.5, 99.5)
_FALLBACK_METHODS: Final[frozenset[MotionMethod]] = frozenset(
    {MotionMethod.FARNEBACK, MotionMethod.RAFT, MotionMethod.PHASE_CORRELATION}
)


def stretch_contrast(
    image: ImageU8, mask: MaskU8 | None = None, reference: ImageU8 | None = None
) -> ImageU8:
    """Étire linéairement les niveaux entre les percentiles 0,5 et 99,5 %.

    Les percentiles sont calculés sur ``image`` (et ``reference`` si fournie, pour
    appliquer la **même** transformation à deux images comparées), dans ``mask``.
    """
    samples = image[mask > 0] if mask is not None else image.ravel()
    if reference is not None:
        samples = np.concatenate([samples, reference[mask > 0] if mask is not None else reference.ravel()])
    if samples.size == 0:
        return image
    lo, hi = np.percentile(samples, _STRETCH_PERCENTILES)
    if hi - lo < 1.0:
        return image
    lut = np.clip((np.arange(256, dtype=np.float32) - lo) * (255.0 / (hi - lo)), 0, 255)
    return np.asarray(lut.astype(np.uint8)[image], dtype=np.uint8)


@dataclass(frozen=True)
class Features:
    points: NDArray[np.float32]  # (N, 2)
    descriptors: NDArray[Any] | None


@dataclass(eq=False)
class MotionFrame:
    """Entrée de l'estimation : image réduite en niveaux de gris et masque du panel.

    ``factor`` relie les coordonnées : ``p_réduit = factor · p_natif``. Les points
    d'intérêt sont calculés au premier besoin puis mis en cache.
    """

    index: int
    gray: ImageU8
    mask: MaskU8 | None = None
    factor: float = 1.0
    _features: dict[str, Features] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.gray.ndim != 2 or self.gray.dtype != np.uint8:
            raise ValueError("gray doit être une image uint8 2D")
        if self.mask is not None and self.mask.shape != self.gray.shape:
            raise ValueError("Le masque doit avoir la taille de l'image")
        if not 0.0 < self.factor <= 1.0:
            raise ValueError("factor hors de ]0, 1]")

    @property
    def size(self) -> tuple[int, int]:
        return int(self.gray.shape[1]), int(self.gray.shape[0])

    def valid_mask(self) -> MaskU8:
        if self.mask is None:
            return np.full(self.gray.shape, 255, dtype=np.uint8)
        return self.mask


def to_native(estimate: MotionEstimate, src: MotionFrame, dst: MotionFrame) -> SimilarityTransform:
    """Transformation de l'estimation exprimée en coordonnées natives."""
    return estimate.transform.rescaled(src.factor, dst.factor)


def _erode(mask: MaskU8, radius: int) -> MaskU8:
    if radius <= 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * radius + 1, 2 * radius + 1))
    return np.asarray(
        cv2.erode(mask, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=255), dtype=np.uint8
    )


def _warp_valid(src: MotionFrame, transform: SimilarityTransform, out_size: tuple[int, int]) -> MaskU8:
    """Pixels de la cible dont l'antécédent tombe dans la zone valide de la source."""
    warped = cv2.warpAffine(
        src.valid_mask(), transform.matrix(), out_size, flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    return np.asarray(warped, dtype=np.uint8)


class MotionEstimator:
    """Estimateur de similarité inter-frames configurable et déterministe."""

    def __init__(self, config: MotionConfig, seed: int = 0) -> None:
        self.cfg = config
        self.seed = seed
        self._low_texture_detector: cv2.Feature2D | None = None
        self._detector: cv2.Feature2D
        if config.detector == "sift":
            self._detector = cv2.SIFT.create(nfeatures=config.max_features)
            self._low_texture_detector = cv2.SIFT.create(
                nfeatures=config.max_features,
                contrastThreshold=config.low_texture_contrast_threshold,
            )
            self._norm = cv2.NORM_L2
            self._method = MotionMethod.SIFT
        else:
            self._detector = cv2.ORB.create(nfeatures=config.max_features)
            self._norm = cv2.NORM_HAMMING
            self._method = MotionMethod.ORB
        self._matcher = cv2.BFMatcher(self._norm, crossCheck=False)

    # ------------------------------------------------------------------ API
    def estimate(self, src: MotionFrame, dst: MotionFrame) -> MotionEstimate:
        """Estime la similarité ``src → dst`` (coordonnées réduites)."""
        if src.gray.shape != dst.gray.shape:
            raise ValueError("Les deux frames doivent avoir la même taille réduite")
        failures: list[MotionEstimate] = []
        stages = [self._estimate_features]
        if self.cfg.fallback_flow == "farneback":
            stages.append(self._estimate_flow)
        if self.cfg.fallback_phase_correlation:
            stages.append(self._estimate_phase_correlation)
        for stage in stages:
            candidate = stage(src, dst)
            if candidate.accepted:
                candidate = self._refine_and_validate(candidate, src, dst)
            if candidate.accepted:
                if failures:
                    logger.info(
                        "Paire %d→%d : repli %s accepté après %s",
                        src.index, dst.index, candidate.method.value,
                        ", ".join(f"{f.method.value} ({f.reason})" for f in failures),
                    )
                return candidate
            failures.append(candidate)
        best = max(failures, key=lambda e: (e.n_inliers, e.inlier_ratio))
        reason = " ; ".join(f"{f.method.value}: {f.reason}" for f in failures)
        logger.warning("Paire %d→%d rejetée : %s", src.index, dst.index, reason)
        return replace(best, accepted=False, reason=reason)

    # ---------------------------------------------------------- points d'intérêt
    def features(self, frame: MotionFrame) -> Features:
        key = self._method.value
        cached = frame._features.get(key)
        if cached is not None:
            return cached
        mask = None
        if frame.mask is not None:
            mask = _erode(frame.mask, self.cfg.mask_erode_px)
        gray = stretch_contrast(frame.gray, mask) if self.cfg.contrast_stretch else frame.gray
        keypoints, descriptors = self._detector.detectAndCompute(gray, mask)
        if (
            len(keypoints) < self.cfg.low_texture_min_features
            and self._low_texture_detector is not None
        ):
            keypoints, descriptors = self._low_texture_detector.detectAndCompute(gray, mask)
            logger.debug("Frame %d peu texturée : %d points après seconde détection",
                         frame.index, len(keypoints))
        points = np.array([k.pt for k in keypoints], dtype=np.float32).reshape(-1, 2)
        features = Features(points, descriptors)
        frame._features[key] = features
        return features

    def _match(self, a: Features, b: Features) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        empty = np.zeros((0, 2), np.float32)
        if a.descriptors is None or b.descriptors is None or len(a.points) < 2 or len(b.points) < 2:
            return empty, empty
        pairs = self._matcher.knnMatch(a.descriptors, b.descriptors, k=2)
        src_idx: list[int] = []
        dst_idx: list[int] = []
        for pair in pairs:
            if len(pair) == 2 and pair[0].distance < self.cfg.lowe_ratio * pair[1].distance:
                src_idx.append(pair[0].queryIdx)
                dst_idx.append(pair[0].trainIdx)
        if not src_idx:
            return empty, empty
        return a.points[src_idx], b.points[dst_idx]

    def _estimate_features(self, src: MotionFrame, dst: MotionFrame) -> MotionEstimate:
        p, q = self._match(self.features(src), self.features(dst))
        return self._robust_fit(src, dst, p, q, self._method)

    # ------------------------------------------------------------ ajustement
    def _robust_fit(
        self,
        src: MotionFrame,
        dst: MotionFrame,
        p: NDArray[np.float32],
        q: NDArray[np.float32],
        method: MotionMethod,
    ) -> MotionEstimate:
        n = len(p)
        if n < max(3, self.cfg.min_inliers):
            return self._failed(src, dst, method, n, f"{n} correspondances")
        cv2.setRNGSeed(self.seed)
        ransac: tuple[NDArray[Any] | None, NDArray[Any] | None] = cv2.estimateAffinePartial2D(
            p, q, method=cv2.RANSAC,
            ransacReprojThreshold=self.cfg.ransac_reproj_threshold,
            maxIters=self.cfg.ransac_max_iters,
            confidence=self.cfg.ransac_confidence,
            refineIters=0,
        )
        matrix, inliers = ransac
        if matrix is None or inliers is None:
            return self._failed(src, dst, method, n, "RANSAC sans solution")
        mask = inliers.ravel().astype(bool)
        transform: SimilarityTransform | None = None
        # Deux passes : ajustement régularisé sur les inliers, puis réévaluation des inliers.
        for _ in range(2):
            if int(mask.sum()) < 3:
                break
            try:
                transform = SimilarityTransform.fit(
                    p[mask], q[mask], rotation_regularization=self.cfg.rotation_regularization
                )
            except ValueError as exc:
                return self._failed(src, dst, method, n, f"ajustement dégénéré ({exc})")
            residuals = np.linalg.norm(transform.apply(p) - q, axis=1)
            mask = residuals < self.cfg.ransac_reproj_threshold
        n_inliers = int(mask.sum())
        if transform is None or n_inliers < 3:
            return self._failed(src, dst, method, n, "trop peu d'inliers")
        residuals = np.linalg.norm(transform.apply(p[mask]) - q[mask], axis=1)
        rms = float(np.sqrt(np.mean(residuals**2)))
        ratio = n_inliers / n
        estimate = MotionEstimate(
            src.index, dst.index, transform, n, n_inliers, ratio, rms, method
        )
        if n_inliers < self.cfg.min_inliers:
            return replace(estimate, accepted=False, reason=f"{n_inliers} inliers")
        if ratio < self.cfg.min_inlier_ratio:
            return replace(estimate, accepted=False, reason=f"taux d'inliers {ratio:.2f}")
        return estimate

    def _failed(
        self, src: MotionFrame, dst: MotionFrame, method: MotionMethod, n: int, reason: str
    ) -> MotionEstimate:
        return MotionEstimate(
            src.index, dst.index, SimilarityTransform(), n, 0, 0.0, 0.0, method,
            accepted=False, reason=reason,
        )

    # ------------------------------------------------------------ flot dense
    def _estimate_flow(self, src: MotionFrame, dst: MotionFrame) -> MotionEstimate:
        """Flot de Farneback échantillonné sur les points structurés, filtré par
        cohérence aller-retour, puis similarité robuste (RANSAC + moindres carrés)."""
        a, b = src.gray, dst.gray
        if self.cfg.contrast_stretch:
            a = stretch_contrast(src.gray, src.mask, dst.gray)
            b = stretch_contrast(dst.gray, src.mask, src.gray)
        forward = _farneback(a, b)
        backward = _farneback(b, a)
        step = self.cfg.flow_grid_step
        h, w = src.gray.shape
        structure = cv2.cornerMinEigenVal(a, blockSize=7, ksize=3)
        threshold = self.cfg.flow_min_structure * float(structure.max())
        # Point le plus structuré de chaque cellule de la grille.
        grid_h, grid_w = h // step, w // step
        if grid_h == 0 or grid_w == 0:
            return self._failed(src, dst, MotionMethod.FARNEBACK, 0, "image trop petite")
        blocks = structure[: grid_h * step, : grid_w * step].reshape(grid_h, step, grid_w, step)
        cells = blocks.transpose(0, 2, 1, 3).reshape(grid_h, grid_w, step * step)
        best = cells.argmax(axis=2)
        gy, gx = np.mgrid[0:grid_h, 0:grid_w]
        ys = (gy * step + best // step).ravel()
        xs = (gx * step + best % step).ravel()
        keep = structure[ys, xs] > threshold
        keep &= _erode(src.valid_mask(), self.cfg.mask_erode_px)[ys, xs] > 0
        ys, xs = ys[keep], xs[keep]
        p = np.column_stack([xs, ys]).astype(np.float32)
        q = (p + forward[ys, xs]).astype(np.float32)
        inside = (q[:, 0] >= 0) & (q[:, 0] <= w - 1) & (q[:, 1] >= 0) & (q[:, 1] <= h - 1)
        qi = np.clip(np.rint(q).astype(int), 0, [w - 1, h - 1])
        inside &= dst.valid_mask()[qi[:, 1], qi[:, 0]] > 0
        round_trip = q + backward[qi[:, 1], qi[:, 0]]
        inside &= np.linalg.norm(round_trip - p, axis=1) < self.cfg.flow_max_fb_error
        p, q = p[inside], q[inside]
        if len(p) < _FLOW_MIN_VECTORS:
            return self._failed(src, dst, MotionMethod.FARNEBACK, len(p),
                                f"flot : {len(p)} vecteurs fiables")
        return self._robust_fit(src, dst, p, q, MotionMethod.FARNEBACK)

    # ------------------------------------------------- corrélation de phase
    def _estimate_phase_correlation(self, src: MotionFrame, dst: MotionFrame) -> MotionEstimate:
        a = _prepared(src)
        b = _prepared(dst)
        h, w = a.shape
        scale, theta = _log_polar_scale_rotation(a, b)
        theta /= 1.0 + self.cfg.rotation_regularization
        if not (1.0 / self.cfg.max_scale_change <= scale <= self.cfg.max_scale_change):
            return self._failed(src, dst, MotionMethod.PHASE_CORRELATION, 0,
                                f"échelle log-polaire hors bornes ({scale:.3f})")
        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
        about_center = (
            SimilarityTransform.from_translation(cx, cy)
            @ SimilarityTransform(scale, theta)
            @ SimilarityTransform.from_translation(-cx, -cy)
        )
        warped = cv2.warpAffine(a, about_center.matrix(), (w, h), flags=cv2.INTER_LINEAR)
        window = cv2.createHanningWindow((w, h), cv2.CV_32F)
        (dx, dy), response = cv2.phaseCorrelate(warped, b, window)
        transform = SimilarityTransform.from_translation(dx, dy) @ about_center
        ratio = float(np.clip(response, 0.0, 1.0))
        estimate = MotionEstimate(
            src.index, dst.index, transform, 0, 0, ratio, 0.0, MotionMethod.PHASE_CORRELATION
        )
        if response < self.cfg.min_phase_response:
            return replace(estimate, accepted=False, reason=f"pic de corrélation {response:.3f}")
        return estimate

    # --------------------------------------------------- raffinement + validation
    def _refine_and_validate(
        self, estimate: MotionEstimate, src: MotionFrame, dst: MotionFrame
    ) -> MotionEstimate:
        transform = estimate.transform
        refined = False
        if self.cfg.ecc_enabled:
            ecc = self._refine_ecc(transform, src, dst)
            if ecc is not None:
                transform, refined = ecc, True
            elif self.cfg.require_ecc_for_fallbacks and estimate.method in _FALLBACK_METHODS:
                return replace(estimate, accepted=False, reason="ECC sans convergence")
        if abs(math.degrees(transform.theta)) > self.cfg.max_rotation_deg:
            return replace(estimate, transform=transform, accepted=False,
                           reason=f"rotation {math.degrees(transform.theta):.2f}°")
        if not 1.0 / self.cfg.max_scale_change <= transform.scale <= self.cfg.max_scale_change:
            return replace(estimate, transform=transform, accepted=False,
                           reason=f"variation d'échelle {transform.scale:.3f}")
        ncc, overlap = photometric_consistency(src, dst, transform, self.cfg.ncc_tile_px)
        estimate = replace(estimate, transform=transform, ecc_refined=refined, ncc=ncc)
        if overlap < self.cfg.min_overlap_ratio:
            return replace(estimate, accepted=False, reason=f"recouvrement {overlap:.2f}")
        if ncc < self.cfg.min_ncc:
            return replace(estimate, accepted=False, reason=f"cohérence structurelle {ncc:.3f}")
        return estimate

    def _refine_ecc(
        self, init: SimilarityTransform, src: MotionFrame, dst: MotionFrame
    ) -> SimilarityTransform | None:
        """ECC (MOTION_AFFINE) initialisé par ``init``, reprojeté sur une similarité.

        ``findTransformECC(template, input, W)`` cherche ``input(W(x)) ≈ template(x)`` :
        avec ``template = dst`` et ``input = src``, ``W = init⁻¹``.
        """
        size = dst.size
        overlap = _erode(
            np.asarray(cv2.bitwise_and(dst.valid_mask(), _warp_valid(src, init, size)), np.uint8),
            max(1, self.cfg.mask_erode_px),
        )
        if int(np.count_nonzero(overlap)) < 0.05 * overlap.size:
            return None
        warp = init.inverse().matrix().astype(np.float32)
        criteria = (
            cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS,
            self.cfg.ecc_max_iterations,
            self.cfg.ecc_epsilon,
        )
        try:
            _, ecc_warp = cv2.findTransformECC(
                dst.gray.astype(np.float32), src.gray.astype(np.float32), warp,
                cv2.MOTION_AFFINE, criteria, overlap, self.cfg.ecc_gauss_filter_size,
            )
            warp = np.asarray(ecc_warp, dtype=np.float32)
        except cv2.error as exc:
            logger.debug("ECC %d→%d sans convergence : %s", src.index, dst.index, exc)
            return None
        try:
            refined = SimilarityTransform.from_matrix(warp, tol=_ECC_SIMILARITY_TOL).inverse()
        except (NotASimilarityError, ValueError) as exc:
            logger.debug("ECC %d→%d non similaire : %s", src.index, dst.index, exc)
            return None
        # Réajuste la similarité régularisée sur les coins (projection propre).
        pts = corners(*src.size)
        affine = cv2.invertAffineTransform(warp)
        target = pts @ affine[:, :2].T + affine[:, 2]
        refined = SimilarityTransform.fit(
            pts, target, rotation_regularization=self.cfg.rotation_regularization
        )
        shift = np.linalg.norm(refined.apply(pts) - init.apply(pts), axis=1).max()
        if shift > self.cfg.ecc_max_correction_px:
            logger.debug("ECC %d→%d rejeté : correction %.2f px", src.index, dst.index, shift)
            return None
        return refined


def _farneback(a: ImageU8, b: ImageU8) -> NDArray[np.float32]:
    """Flot dense de Farneback ``a → b`` (pyramide profonde pour les grands déplacements)."""
    flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 5, 21, 5, 7, 1.5, 0)  # type: ignore[call-overload]
    return np.asarray(flow, dtype=np.float32)


def _prepared(frame: MotionFrame) -> NDArray[np.float32]:
    """Image centrée-réduite, nulle hors masque (pour la corrélation de phase)."""
    image = frame.gray.astype(np.float32)
    valid = frame.valid_mask() > 0
    if valid.any():
        values = image[valid]
        image = (image - values.mean()) / (values.std() + 1e-6)
    image[~valid] = 0.0
    return np.ascontiguousarray(image)


def _log_polar_scale_rotation(
    a: NDArray[np.float32], b: NDArray[np.float32]
) -> tuple[float, float]:
    """Échelle et rotation ``a → b`` par corrélation de phase des spectres d'amplitude
    en coordonnées log-polaires (méthode de Reddy & Chatterji)."""
    h, w = a.shape
    window = cv2.createHanningWindow((w, h), cv2.CV_32F)
    # Zero-padding au carré : sur une image rectangulaire, les pas de fréquence
    # diffèrent selon x et y et le spectre ne serait pas isotrope.
    n = max(h, w)
    fy = np.fft.fftshift(np.fft.fftfreq(n))[:, None]
    fx = np.fft.fftshift(np.fft.fftfreq(n))[None, :]
    x = np.cos(np.pi * fx) * np.cos(np.pi * fy)
    highpass = ((1.0 - x) * (2.0 - x)).astype(np.float32)

    def spectrum(img: NDArray[np.float32]) -> NDArray[np.float32]:
        padded = np.zeros((n, n), dtype=np.float32)
        y0, x0 = (n - h) // 2, (n - w) // 2
        padded[y0 : y0 + h, x0 : x0 + w] = img * window
        mag = np.abs(np.fft.fftshift(np.fft.fft2(padded))).astype(np.float32)
        return np.ascontiguousarray(np.log1p(mag) * highpass)

    center = (n / 2.0, n / 2.0)
    radius = n / 2.0
    size = (n, n)
    flags = cv2.INTER_LINEAR | cv2.WARP_FILL_OUTLIERS | cv2.WARP_POLAR_LOG
    lp_a = cv2.warpPolar(spectrum(a), size, center, radius, flags)
    lp_b = cv2.warpPolar(spectrum(b), size, center, radius, flags)
    # Fenêtre sur l'image log-polaire : l'axe log-rayon n'est pas périodique.
    lp_window = cv2.createHanningWindow(size, cv2.CV_32F)
    (shift_r, shift_t), _ = cv2.phaseCorrelate(
        lp_a.astype(np.float32), lp_b.astype(np.float32), lp_window
    )
    # Un agrandissement s de l'image contracte son spectre d'un facteur s :
    # décalage en log-rayon = −w·ln(s)/ln(radius).
    scale = math.exp(-shift_r * math.log(radius) / n)
    theta = 2.0 * math.pi * shift_t / n
    theta = math.remainder(theta, math.pi)  # ambiguïté de 180° du spectre d'amplitude
    return scale, theta


def gradient_magnitude(gray: ImageU8) -> NDArray[np.float32]:
    """Norme du gradient (Sobel) après étirement de contraste et léger lissage."""
    smooth = cv2.GaussianBlur(stretch_contrast(gray), (0, 0), 1.0).astype(np.float32)
    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1)
    return np.asarray(cv2.magnitude(gx, gy), dtype=np.float32)


def _ncc(x: NDArray[Any], y: NDArray[Any]) -> float | None:
    xc = x.astype(np.float64) - float(x.mean())
    yc = y.astype(np.float64) - float(y.mean())
    denom = math.sqrt(float((xc * xc).sum()) * float((yc * yc).sum()))
    if denom <= 1e-9:
        return None
    return float((xc * yc).sum()) / denom


def photometric_consistency(
    src: MotionFrame, dst: MotionFrame, transform: SimilarityTransform, tile_px: int = 80
) -> tuple[float, float]:
    """Score de cohérence structurelle de ``transform`` et fraction de recouvrement.

    Score = médiane, sur des tuiles de ``tile_px``, de la NCC entre la norme du
    gradient de ``src`` warpée et celle de ``dst`` sur le recouvrement valide.
    Comparer des gradients rend le score sensible au désalignement même sur des
    aplats ; la médiane le rend insensible aux éléments fixes parasites
    (sous-titres, logos) qui n'occupent que quelques tuiles. Les tuiles sans
    structure sont ignorées. Renvoie -1 si aucune tuile n'est exploitable.
    """
    size = dst.size
    grad_src = gradient_magnitude(src.gray)
    grad_dst = gradient_magnitude(dst.gray)
    warped = cv2.warpAffine(grad_src, transform.matrix(), size, flags=cv2.INTER_LINEAR)
    overlap = np.asarray(cv2.bitwise_and(dst.valid_mask(), _warp_valid(src, transform, size)),
                         dtype=np.uint8)
    overlap_b = _erode(overlap, 2) > 0
    ratio = float(overlap_b.mean())
    if not overlap_b.any():
        return -1.0, ratio
    texture = 0.05 * float(np.percentile(grad_dst[overlap_b], 99))
    h, w = overlap_b.shape
    scores: list[float] = []
    for y0 in range(0, h, tile_px):
        for x0 in range(0, w, tile_px):
            sl = (slice(y0, min(h, y0 + tile_px)), slice(x0, min(w, x0 + tile_px)))
            m = overlap_b[sl]
            if m.mean() < 0.5:
                continue
            x = warped[sl][m]
            y = grad_dst[sl][m]
            if max(float(x.mean()), float(y.mean())) < texture:
                continue
            value = _ncc(x, y)
            if value is not None:
                scores.append(value)
    if not scores:
        global_value = _ncc(warped[overlap_b], grad_dst[overlap_b])
        return (-1.0 if global_value is None else global_value), ratio
    return float(np.median(scores)), ratio
