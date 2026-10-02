"""Recalage d'une séquence dans un repère commun, par chaînage des mouvements.

Chaque nouvelle frame est recalée sur la **dernière frame acceptée** (l'ancre) :
``T_cur = T_ancre ∘ M(cur → ancre)``. Une paire rejetée par l'estimateur exclut
la frame (journalisée), l'ancre reste inchangée et la frame suivante est
estimée par rapport à elle ; au-delà de ``max_consecutive_failures`` échecs
consécutifs, la séquence est déclarée interrompue.

Le repère final est **canonique** : l'échelle de la frame la plus zoomée est
ramenée à 1 (aucune perte de résolution), l'orientation et l'origine sont celles
de la première frame. Les transformations renvoyées envoient les coordonnées
**natives** de chaque frame vers ce repère.

Le recalage sur la mosaïque courante et l'ajustement global (anti-dérive) sont
ajoutés en phase 6.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import cv2
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from panelrecon.core.config import PipelineConfig
from panelrecon.core.models import (
    CancellationToken,
    FrameObs,
    MaskU8,
    MotionEstimate,
    SimilarityTransform,
)
from panelrecon.core.motion import MotionEstimator, MotionFrame, to_native
from panelrecon.core.video_io import build_exclusion_mask, make_proxy

logger = logging.getLogger(__name__)

PanelMaskProvider = Callable[[FrameObs], MaskU8 | None]
"""Fournit le masque natif du panel d'une frame (``None`` = toute la frame)."""


@dataclass(frozen=True)
class PoseEdge:
    """Mesure relative entre deux frames : ``transform`` envoie les coordonnées
    natives de ``src`` vers celles de ``dst`` ; ``weight`` ∈ ]0, 1] (confiance)."""

    src: int
    dst: int
    transform: SimilarityTransform
    weight: float


@dataclass
class RegistrationResult:
    """Résultat du recalage d'une séquence.

    ``transforms[i]`` : coordonnées natives de la frame ``i`` → repère canonique.
    """

    transforms: dict[int, SimilarityTransform]
    estimates: list[MotionEstimate]
    rejected: list[MotionEstimate]
    excluded: list[int]
    reference_index: int
    canonical_scale: float
    interrupted: bool = False
    interruption_reason: str = ""
    frame_sizes: dict[int, tuple[int, int]] = field(default_factory=dict)
    n_links: int = 0  # liens directs entre images clés (anti-dérive)
    adjustment_rms_px: tuple[float, float] | None = None  # résidu avant / après ajustement

    @property
    def registered_indices(self) -> list[int]:
        return sorted(self.transforms)

    @property
    def mean_inlier_ratio(self) -> float:
        values = [e.inlier_ratio for e in self.estimates]
        return float(np.mean(values)) if values else 0.0

    @property
    def min_inlier_ratio(self) -> float:
        return min((e.inlier_ratio for e in self.estimates), default=0.0)

    @property
    def rms_reprojection_error(self) -> float:
        """RMS global (px réduits) sur les estimations à base de correspondances."""
        weighted = [(e.rms_reprojection_error**2, e.n_inliers) for e in self.estimates
                    if e.n_inliers > 0]
        total = sum(n for _, n in weighted)
        if total == 0:
            return 0.0
        return float(np.sqrt(sum(v * n for v, n in weighted) / total))


def make_motion_frame(
    frame: FrameObs,
    config: PipelineConfig,
    panel_mask: MaskU8 | None = None,
) -> MotionFrame:
    """Prépare l'entrée de l'estimateur : image réduite et masque réduit.

    Le masque combine le masque du panel (s'il est fourni, en résolution
    native) et les zones d'exclusion de la configuration.
    """
    if frame.proxy_gray is None:
        raise ValueError(f"Frame {frame.index} sans image réduite (compute_proxy=False ?)")
    h, w = frame.proxy_gray.shape
    mask: MaskU8 | None = None
    if config.preprocess.exclusion_zones:
        mask = build_exclusion_mask(h, w, config.preprocess.exclusion_zones)
    if panel_mask is not None:
        if panel_mask.shape != (frame.native_height, frame.native_width):
            raise ValueError("Le masque du panel doit avoir la taille native de la frame")
        small = make_proxy(panel_mask, frame.proxy_factor)
        if small.shape != (h, w):
            small = np.asarray(cv2.resize(panel_mask, (w, h), interpolation=cv2.INTER_AREA),
                               dtype=np.uint8)
        small = np.where(small >= 128, 255, 0).astype(np.uint8)
        mask = small if mask is None else np.asarray(cv2.bitwise_and(mask, small), dtype=np.uint8)
    return MotionFrame(frame.index, frame.proxy_gray, mask, frame.proxy_factor)


class ChainRegistrar:
    """Recalage incrémental par chaînage (une instance par séquence)."""

    def __init__(self, config: PipelineConfig, estimator: MotionEstimator | None = None) -> None:
        self.config = config
        self.estimator = estimator or MotionEstimator(config.motion, seed=config.runtime.seed)
        self._anchor: MotionFrame | None = None
        self._to_reference: dict[int, SimilarityTransform] = {}
        self._sizes: dict[int, tuple[int, int]] = {}
        self._estimates: list[MotionEstimate] = []
        self._rejected: list[MotionEstimate] = []
        self._excluded: list[int] = []
        self._consecutive_failures = 0
        self._interrupted = False
        self._reason = ""
        self._reference_index: int | None = None
        self._edges: list[PoseEdge] = []
        self._keyframes: list[MotionFrame] = []
        self._accepted = 0
        self._n_links = 0

    @property
    def interrupted(self) -> bool:
        return self._interrupted

    @property
    def anchor(self) -> MotionFrame | None:
        """Dernière frame acceptée (cible de la prochaine estimation)."""
        return self._anchor

    def transform_of(self, index: int) -> SimilarityTransform:
        """Transformation native ``frame → frame de référence`` d'une frame recalée."""
        return self._to_reference[index]

    def add(
        self,
        frame: MotionFrame,
        native_size: tuple[int, int],
        estimate: MotionEstimate | None = None,
    ) -> MotionEstimate | None:
        """Recale ``frame`` ; renvoie l'estimation (``None`` pour la première frame).

        ``estimate`` permet de fournir une estimation déjà calculée de ``frame`` vers
        l'ancre courante (elle n'est alors pas recalculée).
        """
        if self._interrupted:
            raise RuntimeError(f"Séquence interrompue : {self._reason}")
        if self._anchor is None:
            self._anchor = frame
            self._reference_index = frame.index
            self._to_reference[frame.index] = SimilarityTransform.identity()
            self._sizes[frame.index] = native_size
            self._keyframes.append(frame)
            return None
        if frame.index <= self._anchor.index:
            raise ValueError(f"Indices non croissants : {frame.index} après {self._anchor.index}")
        if estimate is None:
            estimate = self.estimator.estimate(frame, self._anchor)
        elif (estimate.src_index, estimate.dst_index) != (frame.index, self._anchor.index):
            raise ValueError(
                f"Estimation {estimate.src_index}→{estimate.dst_index} fournie pour "
                f"{frame.index}→{self._anchor.index}"
            )
        if not estimate.accepted:
            self._rejected.append(estimate)
            self._excluded.append(frame.index)
            self._consecutive_failures += 1
            if self._consecutive_failures > self.config.registration.max_consecutive_failures:
                self._interrupted = True
                self._reason = (
                    f"{self._consecutive_failures} échecs consécutifs après la frame "
                    f"{self._anchor.index}"
                )
                logger.warning("Recalage interrompu : %s", self._reason)
            return estimate
        self._consecutive_failures = 0
        native = to_native(estimate, frame, self._anchor)
        self._to_reference[frame.index] = self._to_reference[self._anchor.index] @ native
        self._sizes[frame.index] = native_size
        self._estimates.append(estimate)
        self._edges.append(PoseEdge(frame.index, self._anchor.index, native,
                                    _edge_weight(estimate)))
        self._anchor = frame
        self._accepted += 1
        reg_cfg = self.config.registration
        if reg_cfg.global_adjustment and self._accepted % reg_cfg.keyframe_interval == 0:
            self._add_keyframe(frame)
        return estimate

    # --------------------------------------------------------------- images clés
    def _predicted_overlap(self, src: int, dst: int) -> float:
        """Part de la frame ``src`` qui retombe dans ``dst`` d'après les poses actuelles."""
        src_to_dst = self._to_reference[dst].inverse() @ self._to_reference[src]
        w, h = self._sizes[src]
        dw, dh = self._sizes[dst]
        quad = src_to_dst.apply(_corners(w, h))
        x0, y0 = np.maximum(quad.min(axis=0), 0.0)
        x1, y1 = np.minimum(quad.max(axis=0), (dw - 1.0, dh - 1.0))
        inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
        area = float(np.ptp(quad[:, 0]) * np.ptp(quad[:, 1]))
        return inter / area if area > 0 else 0.0

    def _add_keyframe(self, frame: MotionFrame) -> None:
        """Lie directement la nouvelle image clé à des images clés antérieures qui la
        recouvrent encore (la plus ancienne d'abord : c'est elle qui borne la dérive)."""
        reg_cfg = self.config.registration
        previous = self._keyframes[:-1] if len(self._keyframes) > 1 else []
        candidates = [k for k in previous
                      if self._predicted_overlap(frame.index, k.index) >= reg_cfg.link_min_overlap]
        chosen: list[MotionFrame] = []
        if candidates and reg_cfg.links_per_keyframe > 0:
            picks = np.linspace(0, len(candidates) - 1, reg_cfg.links_per_keyframe)
            for pos in sorted({int(round(p)) for p in picks}):
                chosen.append(candidates[pos])
        for keyframe in chosen:
            estimate = self.estimator.estimate(frame, keyframe)
            if estimate.accepted:
                self._edges.append(PoseEdge(frame.index, keyframe.index,
                                            to_native(estimate, frame, keyframe),
                                            _edge_weight(estimate)))
                self._n_links += 1
        self._keyframes.append(frame)
        for keyframe in self._keyframes:  # libère les caches volumineux
            keyframe._gradient = None
        if len(self._keyframes) > reg_cfg.max_keyframes:
            # Garde la plus ancienne (ancrage à longue portée) et une sur deux ensuite.
            self._keyframes = [self._keyframes[0]] + self._keyframes[2::2]

    def result(self) -> RegistrationResult:
        if self._reference_index is None:
            raise ValueError("Aucune frame recalée")
        poses = dict(self._to_reference)
        rms: tuple[float, float] | None = None
        reg_cfg = self.config.registration
        if reg_cfg.global_adjustment and self._n_links > 0:
            poses, rms = adjust_poses(poses, self._edges, self._sizes, self._reference_index,
                                      reg_cfg.huber_px, reg_cfg.rotation_prior_weight)
        # Taille, dans le repère de référence, d'un pixel de chaque frame : la frame la
        # plus zoomée a le plus petit pixel ; le canevas est mis à son échelle.
        finest = min(t.scale for t in poses.values())
        canonical = SimilarityTransform(scale=1.0 / finest)
        transforms = {i: canonical @ t for i, t in poses.items()}
        return RegistrationResult(
            transforms=transforms,
            estimates=list(self._estimates),
            rejected=list(self._rejected),
            excluded=list(self._excluded),
            reference_index=self._reference_index,
            canonical_scale=1.0 / finest,
            interrupted=self._interrupted,
            interruption_reason=self._reason,
            frame_sizes=dict(self._sizes),
            n_links=self._n_links,
            adjustment_rms_px=rms,
        )


def _corners(width: int, height: int) -> NDArray[np.float64]:
    return np.array([[0.0, 0.0], [width - 1.0, 0.0], [width - 1.0, height - 1.0],
                     [0.0, height - 1.0]])


def _sample_points(width: int, height: int) -> NDArray[np.float64]:
    return np.vstack([_corners(width, height), [[(width - 1) / 2.0, (height - 1) / 2.0]]])


def _edge_weight(estimate: MotionEstimate) -> float:
    """Confiance d'une mesure : nombre d'inliers (saturé) et cohérence structurelle."""
    support = min(1.0, estimate.n_inliers / 100.0) if estimate.n_inliers else 0.5
    coherence = estimate.ncc if estimate.ncc is not None else 0.9
    return float(np.clip(support * max(coherence, 0.0), 0.05, 1.0))


def adjust_poses(
    poses: dict[int, SimilarityTransform],
    edges: list[PoseEdge],
    sizes: dict[int, tuple[int, int]],
    reference: int,
    huber_px: float,
    rotation_weight: float,
) -> tuple[dict[int, SimilarityTransform], tuple[float, float]]:
    """Ajustement global d'un graphe de poses (``frame → référence``, natif).

    Pour chaque mesure ``i → j`` (``M``), le résidu est l'écart, en pixels du
    repère de référence, entre ``P_j(M(p))`` et ``P_i(p)`` aux coins et au centre de
    la frame ``i``, pondéré par la confiance. Inconnues par frame : log-échelle,
    rotation, translation ; la référence est fixe. Perte de Huber (``huber_px``)
    et rappel de la rotation vers 0 (``rotation_weight``). En cas d'échec ou de
    dégradation, les poses initiales sont conservées.
    """
    ids = [i for i in sorted(poses) if i != reference]
    if not ids or not edges:
        return poses, (0.0, 0.0)
    col = {i: k for k, i in enumerate(ids)}
    x0 = np.array([[math.log(poses[i].scale), poses[i].theta, poses[i].tx, poses[i].ty]
                   for i in ids], dtype=np.float64).ravel()
    ref_pose = poses[reference]
    ref_params = np.array([math.log(ref_pose.scale), ref_pose.theta, ref_pose.tx, ref_pose.ty])

    src_pts = np.stack([_sample_points(*sizes[e.src]) for e in edges])        # (E, 5, 2)
    dst_pts = np.stack([e.transform.apply(p) for e, p in zip(edges, src_pts)])  # M(p)
    weights = np.array([e.weight for e in edges])[:, None, None]
    src_col = np.array([col.get(e.src, -1) for e in edges])
    dst_col = np.array([col.get(e.dst, -1) for e in edges])
    half_diag = np.array([0.5 * math.hypot(*sizes[i]) for i in ids])
    prior = math.sqrt(rotation_weight)

    def params_of(x: NDArray[np.float64], cols: NDArray[np.int64]) -> NDArray[np.float64]:
        table = x.reshape(-1, 4)
        out = np.empty((len(cols), 4))
        known = cols >= 0
        out[known] = table[cols[known]]
        out[~known] = ref_params
        return out

    def apply(params: NDArray[np.float64], pts: NDArray[np.float64]) -> NDArray[np.float64]:
        s = np.exp(params[:, 0])[:, None]
        c = s * np.cos(params[:, 1])[:, None]
        sn = s * np.sin(params[:, 1])[:, None]
        x, y = pts[..., 0], pts[..., 1]
        return np.stack([c * x - sn * y + params[:, 2:3], sn * x + c * y + params[:, 3:4]],
                        axis=-1)

    def residuals(x: NDArray[np.float64]) -> NDArray[np.float64]:
        moved_dst = apply(params_of(x, dst_col), dst_pts)
        moved_src = apply(params_of(x, src_col), src_pts)
        r = ((moved_dst - moved_src) * weights).ravel()
        thetas = x.reshape(-1, 4)[:, 1]
        return np.concatenate([r, prior * thetas * half_diag])

    n_rows = len(edges) * 10 + len(ids)
    sparsity = lil_matrix((n_rows, 4 * len(ids)), dtype=np.int8)
    for e_idx in range(len(edges)):
        rows = slice(e_idx * 10, e_idx * 10 + 10)
        for c in (src_col[e_idx], dst_col[e_idx]):
            if c >= 0:
                sparsity[rows, 4 * c : 4 * c + 4] = 1
    for k in range(len(ids)):
        sparsity[len(edges) * 10 + k, 4 * k + 1] = 1

    def rms(x: NDArray[np.float64]) -> float:
        r = residuals(x)[: len(edges) * 10].reshape(-1, 2)
        return float(np.sqrt(np.mean(np.sum(r * r, axis=1))))

    before = rms(x0)
    try:
        solution = least_squares(residuals, x0, jac_sparsity=sparsity, loss="huber",
                                 f_scale=huber_px, x_scale="jac", method="trf", max_nfev=200)
    except (ValueError, np.linalg.LinAlgError) as exc:
        logger.warning("Ajustement global impossible (%s) : poses du chaînage conservées", exc)
        return poses, (before, before)
    after = rms(solution.x)
    if not np.all(np.isfinite(solution.x)) or after > before:
        logger.warning("Ajustement global non concluant (%.3f → %.3f px) : poses conservées",
                       before, after)
        return poses, (before, before)
    table = solution.x.reshape(-1, 4)
    adjusted = {reference: poses[reference]}
    for i, k in col.items():
        adjusted[i] = SimilarityTransform(float(math.exp(table[k, 0])), float(table[k, 1]),
                                          float(table[k, 2]), float(table[k, 3]))
    logger.info("Ajustement global : %d frames, %d mesures, résidu %.3f → %.3f px",
                len(ids) + 1, len(edges), before, after)
    return adjusted, (before, after)


def register_frames(
    frames: Iterable[FrameObs],
    config: PipelineConfig,
    panel_masks: PanelMaskProvider | None = None,
    cancel: CancellationToken | None = None,
) -> RegistrationResult:
    """Recale une séquence de frames (en flux : seule l'ancre est gardée en mémoire)."""
    registrar = ChainRegistrar(config)
    for frame in frames:
        if cancel is not None:
            cancel.raise_if_cancelled()
        mask = panel_masks(frame) if panel_masks is not None else None
        registrar.add(make_motion_frame(frame, config, mask),
                     (frame.native_width, frame.native_height))
        if registrar.interrupted:
            break
    result = registrar.result()
    logger.info(
        "Recalage : %d frames, %d rejetées, taux d'inliers moyen %.2f, échelle canonique %.3f",
        len(result.transforms), len(result.rejected), result.mean_inlier_ratio,
        result.canonical_scale,
    )
    return result


def merge_registrations(
    first: RegistrationResult,
    second: RegistrationResult,
    link: SimilarityTransform,
    link_estimate: MotionEstimate,
) -> RegistrationResult:
    """Réunit deux recalages d'un même panel (séquence coupée à une frontière de
    tronçon) en un seul, dans le repère de référence de ``first``.

    ``link`` envoie les coordonnées natives de la première frame de ``second``
    vers celles de la dernière frame de ``first``.
    """
    last_a, first_b = max(first.transforms), min(second.transforms)
    if first_b <= last_a:
        raise ValueError(f"Recalages qui se chevauchent : {first_b} ≤ {last_a}")
    # Poses vers la frame de référence de chaque morceau (repère canonique retiré).
    pose_a = {i: SimilarityTransform(scale=1.0 / first.canonical_scale) @ t
              for i, t in first.transforms.items()}
    pose_b = {i: SimilarityTransform(scale=1.0 / second.canonical_scale) @ t
              for i, t in second.transforms.items()}
    b_to_a = pose_a[last_a] @ link @ pose_b[first_b].inverse()
    poses = dict(pose_a)
    poses.update({i: b_to_a @ t for i, t in pose_b.items()})
    finest = min(t.scale for t in poses.values())
    canonical = SimilarityTransform(scale=1.0 / finest)
    rms: tuple[float, float] | None = None
    if first.adjustment_rms_px is not None or second.adjustment_rms_px is not None:
        values = [r for r in (first.adjustment_rms_px, second.adjustment_rms_px) if r is not None]
        rms = (max(r[0] for r in values), max(r[1] for r in values))  # pire des morceaux
    return RegistrationResult(
        transforms={i: canonical @ t for i, t in poses.items()},
        estimates=[*first.estimates, link_estimate, *second.estimates],
        rejected=[*first.rejected, *second.rejected],
        excluded=sorted(set(first.excluded) | set(second.excluded)),
        reference_index=first.reference_index,
        canonical_scale=1.0 / finest,
        interrupted=first.interrupted or second.interrupted,
        interruption_reason="; ".join(r for r in (first.interruption_reason,
                                                  second.interruption_reason) if r),
        frame_sizes={**first.frame_sizes, **second.frame_sizes},
        n_links=first.n_links + second.n_links,
        adjustment_rms_px=rms,
    )
