"""Découpage d'une vidéo en séquences (une séquence = un panel), en flux.

Pour chaque paire de frames consécutives ``(i−1, i)``, la paire est dite **en
changement** si l'un des signaux suivants se déclenche :

* PySceneDetect (``ContentDetector``) signale une coupure à ``i`` et le mouvement
  n'est pas recalé avec une forte cohérence (le détecteur réagit aussi aux
  déplacements rapides d'un même panel) ;
* la corrélation des histogrammes HSV des vignettes tombe sous le seuil ;
* l'estimation du mouvement de ``i`` vers la frame précédente est rejetée
  (effondrement du taux d'inliers, incohérence structurelle…) ;
* le score de cohérence structurelle entre ``i`` et ``i − dissolve_lag``,
  recalées par composition des mouvements consécutifs, tombe sous le seuil :
  changement progressif (fondu), y compris pendant une transition, qui ne se
  termine donc que lorsque le contenu est de nouveau stable.

Une suite maximale de paires en changement ``a..b`` forme une **transition** :
les frames ``a..b−1`` (intérieures : mélanges d'un fondu, frame parasite) sont
écartées, la séquence courante se termine en ``a−1`` et la suivante commence en
``b``. Une suite d'une seule paire est une coupe franche (aucune frame écartée).
Si ``b`` se recale avec succès sur la dernière frame de la séquence courante
(frame parasite isolée, flash), les deux morceaux sont réunis et les frames
intérieures sont simplement exclues.

Le recalage par chaînage des frames de chaque séquence est effectué au passage
(les estimations ne sont calculées qu'une fois).
"""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray
from scenedetect.detectors import ContentDetector

from panelrecon.core.config import PipelineConfig
from panelrecon.core.models import (
    CancellationToken,
    FrameObs,
    MotionEstimate,
    Sequence,
    SimilarityTransform,
)
from panelrecon.core.motion import MotionEstimator, MotionFrame, photometric_consistency
from panelrecon.core.registration import ChainRegistrar, RegistrationResult, make_motion_frame

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Transition:
    """Changement de panel (ou perturbation fusionnée) entre deux séquences.

    ``first_after`` : première frame après la transition ; ``excluded`` : frames
    intérieures écartées ; ``kind`` : ``cut`` (coupe franche), ``dissolve``
    (fondu, frames de mélange écartées) ou ``glitch`` (même panel de part et
    d'autre : morceaux réunis).
    """

    first_after: int
    kind: str
    reasons: tuple[str, ...]
    excluded: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "first_after": self.first_after,
            "kind": self.kind,
            "reasons": list(self.reasons),
            "excluded": list(self.excluded),
        }


@dataclass
class SequenceRegistration:
    sequence: Sequence
    registration: RegistrationResult


@dataclass
class SplitResult:
    sequences: list[SequenceRegistration] = field(default_factory=list)
    transitions: list[Transition] = field(default_factory=list)
    dropped: list[Sequence] = field(default_factory=list)
    discarded: list[int] = field(default_factory=list)  # frames de transitions trop longues
    frames_seen: int = 0


def thumbnail(image: NDArray[np.uint8], width: int) -> NDArray[np.uint8]:
    h, w = image.shape[:2]
    if w <= width:
        return image
    height = max(1, round(h * width / w))
    return np.asarray(cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA), np.uint8)


def hsv_histogram(image: NDArray[np.uint8]) -> NDArray[np.float32]:
    """Histogrammes H (32), S (16) et V (32) concaténés et normalisés."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    parts = [
        cv2.calcHist([hsv], [0], None, [32], [0, 180]),
        cv2.calcHist([hsv], [1], None, [16], [0, 256]),
        cv2.calcHist([hsv], [2], None, [32], [0, 256]),
    ]
    hist = np.concatenate([p.ravel() for p in parts]).astype(np.float32)
    total = float(hist.sum())
    return hist / total if total > 0 else hist


def histogram_correlation(a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
    return float(cv2.compareHist(a, b, cv2.HISTCMP_CORREL))


@dataclass
class _Pending:
    """Frame en attente pendant une transition."""

    frame: MotionFrame
    size: tuple[int, int]


class SequenceTracker:
    """Découpage + recalage incrémental. Alimenter avec :meth:`push`, conclure avec
    :meth:`finish`."""

    def __init__(self, config: PipelineConfig, estimator: MotionEstimator | None = None) -> None:
        self.config = config
        self.cfg = config.scenes
        self.estimator = estimator or MotionEstimator(config.motion, seed=config.runtime.seed)
        self._detector = (
            ContentDetector(threshold=self.cfg.content_threshold, min_scene_len=1)
            if self.cfg.use_scenedetect else None
        )
        self._result = SplitResult()
        self._registrar: ChainRegistrar | None = None
        # Dernières frames et mouvement consécutif (réduit) vers la précédente ; None
        # quand la paire a été rejetée (la chaîne est alors rompue).
        self._chain: deque[tuple[MotionFrame, SimilarityTransform | None]] = deque(
            maxlen=self.cfg.dissolve_lag + 1)
        self._extra_excluded: list[int] = []
        self._prev_frame: MotionFrame | None = None
        self._prev_hist: NDArray[np.float32] | None = None
        self._transition: list[_Pending] | None = None
        self._transition_reasons: list[str] = []
        self._finished = False

    # ---------------------------------------------------------------- flux
    def push(self, frame: FrameObs) -> None:
        if self._finished:
            raise RuntimeError("Découpage déjà terminé")
        self._result.frames_seen += 1
        small = thumbnail(frame.image, self.cfg.thumbnail_width)
        hist = hsv_histogram(small)
        content_cut = bool(self._detector.process_frame(frame.index, small)) if self._detector else False
        mf = make_motion_frame(frame, self.config)
        size = (frame.width, frame.height)
        prev, prev_hist = self._prev_frame, self._prev_hist
        self._prev_frame, self._prev_hist = mf, hist
        if prev is None or prev_hist is None:
            self._start_sequence(mf, size)
            return
        reasons: list[str] = []
        corr = histogram_correlation(prev_hist, hist)
        if corr < self.cfg.histogram_min_correlation:
            reasons.append(f"histogramme {corr:.3f}")

        if self._transition is None:
            registrar = self._registrar
            assert registrar is not None and registrar.anchor is not None
            estimate = self.estimator.estimate(mf, registrar.anchor)
            if not estimate.accepted:
                reasons.append(f"mouvement rejeté ({estimate.reason})")
            if content_cut and not self._well_registered(estimate):
                reasons.append("PySceneDetect")
            self._chain.append((mf, estimate.transform if estimate.accepted else None))
            if not reasons:
                lag_score = self._lag_score()
                if lag_score is not None and lag_score < self.cfg.dissolve_min_score:
                    reasons.append(f"changement progressif (score {lag_score:.3f})")
            if reasons:
                self._transition = [_Pending(mf, size)]
                self._transition_reasons = reasons
                logger.debug("Transition ouverte à la frame %d : %s", mf.index, reasons)
                return
            registrar.add(mf, size, estimate)
            return

        # Transition en cours : seule la paire consécutive est évaluée.
        estimate = self.estimator.estimate(mf, prev)
        if not estimate.accepted:
            reasons.append(f"mouvement rejeté ({estimate.reason})")
        if content_cut and not self._well_registered(estimate):
            reasons.append("PySceneDetect")
        self._chain.append((mf, estimate.transform if estimate.accepted else None))
        if not reasons:
            # La transition ne se termine que lorsque le contenu est de nouveau stable.
            lag_score = self._lag_score()
            if lag_score is not None and lag_score < self.cfg.dissolve_min_score:
                reasons.append(f"changement progressif (score {lag_score:.3f})")
        if reasons:
            self._transition.append(_Pending(mf, size))
            self._transition_reasons.extend(reasons)
            if len(self._transition) > self.cfg.max_transition_frames:
                dropped = self._transition.pop(0)
                self._result.discarded.append(dropped.frame.index)
            return
        self._close_transition(estimate_to_next=(estimate, mf, size))

    def finish(self) -> SplitResult:
        if self._finished:
            return self._result
        if self._transition is not None:
            self._close_transition(estimate_to_next=None)
        self._close_sequence()
        self._finished = True
        if self._result.discarded:
            logger.warning("%d frames de transitions trop longues écartées",
                           len(self._result.discarded))
        return self._result

    # ------------------------------------------------------------ internes
    def _well_registered(self, estimate: MotionEstimate) -> bool:
        """Mouvement recalé avec une forte cohérence : une coupure signalée par
        PySceneDetect (sensible aux déplacements rapides) n'est alors pas retenue."""
        return estimate.accepted and (estimate.ncc or -1.0) >= self.cfg.merge_min_score

    def _lag_score(self) -> float | None:
        """Cohérence structurelle entre la dernière frame et celle située
        ``dissolve_lag`` frames avant, recalées par composition des mouvements
        consécutifs ; ``None`` si la chaîne est incomplète ou rompue."""
        if len(self._chain) < self.cfg.dissolve_lag + 1:
            return None
        items = list(self._chain)
        oldest, newest = items[0][0], items[-1][0]
        old_to_new = SimilarityTransform.identity()
        # Chaque transformation envoie une frame vers la précédente : on compose
        # les inverses de la plus ancienne vers la plus récente.
        for _, to_previous in items[1:]:
            if to_previous is None:
                return None
            old_to_new = to_previous.inverse() @ old_to_new
        score, _ = photometric_consistency(oldest, newest, old_to_new,
                                           self.config.motion.ncc_tile_px)
        return score

    def _start_sequence(self, frame: MotionFrame, size: tuple[int, int]) -> None:
        self._registrar = ChainRegistrar(self.config, self.estimator)
        self._registrar.add(frame, size)
        self._extra_excluded = []

    def _close_sequence(self) -> None:
        registrar = self._registrar
        if registrar is None:
            return
        self._registrar = None
        result = registrar.result()
        indices = result.registered_indices
        excluded = tuple(sorted(i for i in self._extra_excluded if indices[0] <= i <= indices[-1]))
        result.excluded = sorted(set(result.excluded) | set(excluded))
        sequence = Sequence(indices[0], indices[-1], tuple(result.excluded))
        if len(indices) < self.cfg.min_sequence_frames:
            self._result.dropped.append(sequence)
            logger.info("Séquence %d–%d ignorée (%d frames)", sequence.start_idx,
                        sequence.end_idx, len(indices))
            return
        self._result.sequences.append(SequenceRegistration(sequence, result))

    def _close_transition(
        self, estimate_to_next: tuple[MotionEstimate, MotionFrame, tuple[int, int]] | None
    ) -> None:
        pending = self._transition
        assert pending is not None
        self._transition = None
        reasons = tuple(dict.fromkeys(self._transition_reasons))
        first_after = pending[-1]
        interior = tuple(p.frame.index for p in pending[:-1])
        registrar = self._registrar
        assert registrar is not None and registrar.anchor is not None
        merge = self.estimator.estimate(first_after.frame, registrar.anchor)
        if merge.accepted and (merge.ncc or -1.0) >= self.cfg.merge_min_score:
            registrar.add(first_after.frame, first_after.size, merge)
            self._extra_excluded.extend(interior)
            kind = "glitch"
            logger.info("Perturbation %s réunie (même panel), frames écartées : %s",
                        reasons, list(interior))
        else:
            self._close_sequence()
            self._start_sequence(first_after.frame, first_after.size)
            kind = "dissolve" if interior else "cut"
            logger.info("Nouvelle séquence à la frame %d (%s, %s), frames écartées : %s",
                        first_after.frame.index, kind, ", ".join(reasons), list(interior))
        self._result.transitions.append(
            Transition(first_after.frame.index, kind, reasons, interior)
        )
        if estimate_to_next is not None:
            estimate, frame, size = estimate_to_next
            new_registrar = self._registrar
            assert new_registrar is not None
            new_registrar.add(frame, size, estimate)


def split_and_register(
    frames: Iterable[FrameObs],
    config: PipelineConfig,
    cancel: CancellationToken | None = None,
) -> SplitResult:
    """Découpe une vidéo en séquences et recale chacune (une seule passe de décodage)."""
    tracker = SequenceTracker(config)
    for frame in frames:
        if cancel is not None:
            cancel.raise_if_cancelled()
        tracker.push(frame)
    result = tracker.finish()
    logger.info(
        "%d séquence(s), %d transition(s) sur %d frames",
        len(result.sequences), len(result.transitions), result.frames_seen,
    )
    return result

