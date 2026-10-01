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
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import cv2
import numpy as np

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
        if panel_mask.shape != frame.image.shape[:2]:
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

    @property
    def interrupted(self) -> bool:
        return self._interrupted

    def add(self, frame: MotionFrame, native_size: tuple[int, int]) -> MotionEstimate | None:
        """Recale ``frame`` ; renvoie l'estimation (``None`` pour la première frame)."""
        if self._interrupted:
            raise RuntimeError(f"Séquence interrompue : {self._reason}")
        if self._anchor is None:
            self._anchor = frame
            self._reference_index = frame.index
            self._to_reference[frame.index] = SimilarityTransform.identity()
            self._sizes[frame.index] = native_size
            return None
        if frame.index <= self._anchor.index:
            raise ValueError(f"Indices non croissants : {frame.index} après {self._anchor.index}")
        estimate = self.estimator.estimate(frame, self._anchor)
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
        self._anchor = frame
        return estimate

    def result(self) -> RegistrationResult:
        if self._reference_index is None:
            raise ValueError("Aucune frame recalée")
        # Taille, dans le repère de référence, d'un pixel de chaque frame : la frame la
        # plus zoomée a le plus petit pixel ; le canevas est mis à son échelle.
        finest = min(t.scale for t in self._to_reference.values())
        canonical = SimilarityTransform(scale=1.0 / finest)
        transforms = {i: canonical @ t for i, t in self._to_reference.items()}
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
        )


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
        registrar.add(make_motion_frame(frame, config, mask), (frame.width, frame.height))
        if registrar.interrupted:
            break
    result = registrar.result()
    logger.info(
        "Recalage : %d frames, %d rejetées, taux d'inliers moyen %.2f, échelle canonique %.3f",
        len(result.transforms), len(result.rejected), result.mean_inlier_ratio,
        result.canonical_scale,
    )
    return result
