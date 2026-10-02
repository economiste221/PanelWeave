"""Orchestration du traitement d'une vidéo (sans dépendance à l'interface).

Étapes (aucune vidéo n'est gardée en mémoire) :

1. **Index des frames** : lecture des paquets (sans décodage) ; chaque frame
   reçoit un indice stable (rang de son pts), ce qui permet de décoder n'importe
   quelle plage en se positionnant sur l'image clé précédente.
2. **Découpage et recalage** (images réduites), par **tronçons** traités en
   parallèle (processus séparés) : séquences, transitions, transformations de
   chaque frame vers le repère canonique de sa séquence. Aux frontières de
   tronçons, la dernière frame d'un côté est recalée sur la première de l'autre :
   si c'est le même panel, les deux séquences sont réunies, sinon la frontière
   est une coupe.
3. **Reconstruction de chaque séquence**, en parallèle (les panels sont
   indépendants) : emprise du panel (images réduites échantillonnées), puis
   fusion des frames retenues (résolution native) et export.

Sans index exploitable (repli OpenCV, conteneur sans pts), le traitement est
séquentiel en trois passes linéaires.

Une erreur sur une séquence est capturée et n'empêche pas les autres ; une
erreur sur la vidéo est capturée dans le résultat (seule l'annulation remonte).
Les callbacks de progression reçoivent ``(fraction ∈ [0, 1], message)``.
"""

from __future__ import annotations

import bisect
import copy
import logging
import math
import time
import traceback
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from panelrecon.core.config import PipelineConfig
from panelrecon.core.export import (
    ExportedFiles,
    export_sequence,
    sequence_basename,
    write_json,
)
from panelrecon.core.hardware import resolve_num_workers
from panelrecon.core.models import (
    CancellationToken,
    FrameObs,
    OperationCancelled,
    Sequence,
    VideoInfo,
)
from panelrecon.core.mosaic import build_mosaic, plan_fusion
from panelrecon.core.motion import MotionEstimator, to_native
from panelrecon.core.parallel import WorkerPool, collect, worker_cancel_token
from panelrecon.core.registration import make_motion_frame
from panelrecon.core.scene_split import (
    SequenceRegistration,
    SplitResult,
    concatenate_splits,
    drop_short_sequences,
    histogram_correlation,
    hsv_histogram,
    split_and_register,
    thumbnail,
)
from panelrecon.core.segmentation import PanelRegion, PanelRegionEstimator
from panelrecon.core.video_io import FrameIndex, VideoReader, scan_frame_index

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]
SEQUENCES_SUFFIX = "_sequences.json"
_SPLIT_SHARE = 0.4  # part de la progression attribuée au découpage


@dataclass
class SequenceOutcome:
    number: int
    sequence: Sequence
    frames_used: int
    files: ExportedFiles | None = None
    error: str | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "number": self.number,
            "sequence": self.sequence.to_dict(),
            "frames_used": self.frames_used,
            "files": None if self.files is None else self.files.to_dict(),
            "error": self.error,
            "warnings": self.warnings,
        }


@dataclass
class VideoOutcome:
    path: Path
    info: VideoInfo | None = None
    sequences: list[SequenceOutcome] = field(default_factory=list)
    split: SplitResult | None = None
    error: str | None = None
    traceback: str | None = None
    elapsed_s: float = 0.0
    chunks: int = 1
    workers: int = 1

    @property
    def ok(self) -> bool:
        return (self.error is None and bool(self.sequences)
                and all(s.error is None for s in self.sequences))

    def to_dict(self) -> dict[str, Any]:
        split = self.split
        return {
            "path": str(self.path),
            "info": None if self.info is None else self.info.to_dict(),
            "sequences": [s.to_dict() for s in self.sequences],
            "transitions": [] if split is None else [t.to_dict() for t in split.transitions],
            "dropped_sequences": [] if split is None else [s.to_dict() for s in split.dropped],
            "discarded_frames": [] if split is None else split.discarded,
            "error": self.error,
            "elapsed_s": round(self.elapsed_s, 3),
            "chunks": self.chunks,
            "workers": self.workers,
        }


@dataclass(frozen=True)
class _VideoSource:
    """Ce qu'une tâche (éventuellement dans un autre processus) doit savoir de la vidéo."""

    path: Path
    frame_index: FrameIndex | None
    config: PipelineConfig

    def reader(self, cancel: CancellationToken | None) -> VideoReader:
        cfg = self.config
        return VideoReader(self.path, cfg.video, cfg.preprocess, cancel=cancel,
                           frame_index=self.frame_index)


class _SharedStream:
    """Flux de frames partagé entre des consommateurs successifs (séquences ordonnées)."""

    def __init__(self, frames: Iterator[FrameObs]) -> None:
        self._frames = frames
        self._pending: FrameObs | None = None

    def _peek(self) -> FrameObs | None:
        if self._pending is None:
            self._pending = next(self._frames, None)
        return self._pending

    def take(self, start: int, end: int) -> Iterator[FrameObs]:
        """Frames d'indice ``start ≤ i ≤ end`` ; s'arrête sans consommer la suivante."""
        while True:
            frame = self._peek()
            if frame is None or frame.index > end:
                return
            self._pending = None
            if frame.index >= start:
                yield frame


def _scaled(progress: ProgressCallback | None, start: float, span: float) -> ProgressCallback:
    def report(fraction: float, message: str) -> None:
        if progress is not None:
            progress(start + span * min(1.0, max(0.0, fraction)), message)

    return report


# ---------------------------------------------------------------------------
# Tronçons : découpage parallèle et raccord
# ---------------------------------------------------------------------------


def plan_chunks(n_frames: int, fps: float, config: PipelineConfig,
                workers: int) -> list[tuple[int, int]]:
    """Plages ``[début, fin)`` des tronçons de découpage.

    Au plus ``chunk_seconds`` par tronçon ; une vidéo plus courte est quand même
    répartie sur les ``workers`` processus tant que chaque tronçon dure au moins
    ``min_chunk_seconds``.
    """
    rt = config.runtime
    if n_frames <= 0:
        return []
    if rt.chunk_seconds <= 0.0 or fps <= 0.0 or workers <= 1:
        return [(0, n_frames)]
    duration = n_frames / fps
    count = max(math.ceil(duration / rt.chunk_seconds),
                min(workers, int(duration // rt.min_chunk_seconds)))
    count = max(1, min(count, n_frames))
    bounds = [round(i * n_frames / count) for i in range(count + 1)]
    return [(a, b) for a, b in zip(bounds[:-1], bounds[1:], strict=True) if b > a]


def _split_chunk(source: _VideoSource, start: int, stop: int,
                 cancel: CancellationToken | None = None,
                 progress: ProgressCallback | None = None) -> SplitResult:
    """Tâche : découpage et recalage des frames ``[start, stop)``."""
    cancel = worker_cancel_token(cancel)
    # Les séquences coupées par une frontière ne doivent pas être écartées avant le
    # raccord : le filtre de longueur est appliqué après.
    config = copy.deepcopy(source.config)
    config.scenes.min_sequence_frames = 1
    span = max(1, stop - start)
    with source.reader(cancel) as reader:
        def frames() -> Iterator[FrameObs]:
            for frame in reader.frames(start, stop, proxy_only=True):
                if progress is not None:
                    progress((frame.index - start + 1) / span, f"découpage {frame.index}")
                yield frame

        return split_and_register(frames(), config, cancel)


def _read_one(reader: VideoReader, index: int) -> FrameObs | None:
    return next(iter(reader.frames(index, index + 1, proxy_only=True)), None)


def _stitch(source: _VideoSource, parts: list[SplitResult],
            cancel: CancellationToken | None) -> SplitResult:
    """Raccorde les découpages des tronçons (dans l'ordre)."""
    config = source.config
    estimator = MotionEstimator(config.motion, seed=config.runtime.seed)
    result = parts[0]
    with source.reader(cancel) as reader:
        for part in parts[1:]:
            link = None
            reason = "frontière de tronçon"
            if result.sequences and part.sequences:
                last = result.sequences[-1].sequence.end_idx
                first = part.sequences[0].sequence.start_idx
                frame_a, frame_b = _read_one(reader, last), _read_one(reader, first)
                if frame_a is None or frame_b is None:
                    reason += " : frames de raccord illisibles"
                else:
                    mf_a = make_motion_frame(frame_a, config)
                    mf_b = make_motion_frame(frame_b, config)
                    width = config.scenes.thumbnail_width
                    corr = histogram_correlation(hsv_histogram(thumbnail(frame_a.image, width)),
                                                 hsv_histogram(thumbnail(frame_b.image, width)))
                    estimate = estimator.estimate(mf_b, mf_a)
                    score = estimate.ncc if estimate.ncc is not None else -1.0
                    if (estimate.accepted and corr >= config.scenes.histogram_min_correlation
                            and score >= config.scenes.dissolve_min_score):
                        link = (estimate, to_native(estimate, mf_b, mf_a))
                    else:
                        reason += (f" : panels différents (mouvement "
                                   f"{'accepté' if estimate.accepted else 'rejeté'}, "
                                   f"cohérence {score:.3f}, histogramme {corr:.3f})")
            result = concatenate_splits(result, part, link, reason)
    return result


# ---------------------------------------------------------------------------
# Reconstruction d'une séquence
# ---------------------------------------------------------------------------


def _segmentation_sample(item: SequenceRegistration, limit: int) -> list[int]:
    """Frames bien réparties dans la séquence : un échantillon suffit à cumuler les preuves."""
    indices = sorted(item.registration.transforms)
    if len(indices) > limit:
        picks = np.linspace(0, len(indices) - 1, limit)
        indices = [indices[int(round(p))] for p in picks]
    return indices


def _new_estimator(item: SequenceRegistration, config: PipelineConfig,
                   frame: FrameObs) -> PanelRegionEstimator:
    registration = item.registration
    return PanelRegionEstimator(config, registration.transforms, registration.frame_sizes,
                                frame.proxy_factor)


def _finish_region(k: int, estimator: PanelRegionEstimator | None) -> PanelRegion | None:
    if estimator is None:
        return None
    region = estimator.estimate()
    if not region.segmented:
        logger.warning("Séquence %d : panel non segmenté (frame entière)", k)
    return region


def _fuse_and_export(
    frames: Iterable[FrameObs],
    k: int,
    item: SequenceRegistration,
    region: PanelRegion | None,
    config: PipelineConfig,
    out_dir: Path,
    video_stem: str,
    info: VideoInfo | None,
    cancel: CancellationToken | None,
    progress: ProgressCallback | None,
) -> SequenceOutcome:
    """Fusion et export d'une séquence ; une erreur est capturée dans le résultat."""
    sequence, registration = item.sequence, item.registration
    outcome = SequenceOutcome(k, sequence, len(registration.transforms))
    try:
        mosaic = build_mosaic(
            frames, registration, config,
            panel_masks=None if region is None else region.mask,
            cancel=cancel,
            progress=progress,
            clip=None if region is None else region.bounds(),
        )
        extra = {"segmentation": None if region is None else {
            "segmented": region.segmented,
            "panel_polygon_canonical": region.polygon.tolist(),
        }}
        outcome.files = export_sequence(mosaic, registration, out_dir,
                                        sequence_basename(video_stem, k, sequence),
                                        config, info, extra)
        if region is not None and not region.segmented:
            outcome.warnings.append("panel non segmenté : frame entière utilisée")
    except OperationCancelled:
        raise
    except Exception as exc:  # une séquence en échec n'arrête pas les autres
        outcome.error = f"{type(exc).__name__}: {exc}"
        logger.error("Séquence %d de %s en échec : %s\n%s", k, video_stem, outcome.error,
                     traceback.format_exc())
    return outcome


def _reconstruct_sequence(
    source: _VideoSource,
    k: int,
    item: SequenceRegistration,
    out_dir: Path,
    info: VideoInfo | None,
    cancel: CancellationToken | None = None,
    progress: ProgressCallback | None = None,
) -> SequenceOutcome:
    """Tâche : segmentation, fusion et export d'une séquence (accès direct aux frames)."""
    cancel = worker_cancel_token(cancel)
    config = source.config
    start, stop = item.sequence.start_idx, item.sequence.end_idx + 1
    report = _scaled(progress, 0.0, 1.0)
    region: PanelRegion | None = None
    try:
        with source.reader(cancel) as reader:
            if config.segmentation.method != "none":
                sample = _segmentation_sample(item, config.segmentation.max_frames)
                estimator: PanelRegionEstimator | None = None
                for frame in reader.frames(start, stop, proxy_only=True, wanted=set(sample)):
                    if estimator is None:
                        estimator = _new_estimator(item, config, frame)
                    estimator.add(frame)
                region = _finish_region(k, estimator)
            report(0.1, f"séquence {k} : fusion")
            chosen = plan_fusion(item.registration, config,
                                 None if region is None else region.bounds())[2]
            frames = reader.frames(start, stop, compute_proxy=False, wanted=set(chosen))
            return _fuse_and_export(frames, k, item, region, config, out_dir,
                                    source.path.stem, info, cancel,
                                    _scaled(progress, 0.1, 0.9))
    except OperationCancelled:
        raise
    except Exception as exc:  # planification ou lecture impossible : séquence en échec
        outcome = SequenceOutcome(k, item.sequence, len(item.registration.transforms))
        outcome.error = f"{type(exc).__name__}: {exc}"
        logger.error("Séquence %d de %s en échec : %s\n%s", k, source.path.name, outcome.error,
                     traceback.format_exc())
        return outcome


# ---------------------------------------------------------------------------
# Repli séquentiel (sans index des frames)
# ---------------------------------------------------------------------------


def _segment_streaming(reader: VideoReader, split: SplitResult, config: PipelineConfig,
                       cancel: CancellationToken | None, progress: ProgressCallback,
                       total: int) -> dict[int, PanelRegion]:
    """Emprise du panel de chaque séquence, en une passe sur la vidéo."""
    if config.segmentation.method == "none" or not split.sequences:
        return {}
    starts = [s.sequence.start_idx for s in split.sequences]
    sample: set[int] = set()
    for item in split.sequences:
        sample.update(_segmentation_sample(item, config.segmentation.max_frames))
    estimators: dict[int, PanelRegionEstimator] = {}
    for frame in reader.frames(proxy_only=True, wanted=sample):
        if cancel is not None:
            cancel.raise_if_cancelled()
        progress((frame.index + 1) / total, f"segmentation {frame.index}")
        k = bisect.bisect_right(starts, frame.index) - 1
        if k < 0 or frame.index > split.sequences[k].sequence.end_idx:
            continue
        if k not in estimators:
            estimators[k] = _new_estimator(split.sequences[k], config, frame)
        estimators[k].add(frame)
    return {k: region for k, est in estimators.items()
            if (region := _finish_region(k, est)) is not None}


def _reconstruct_streaming(source: _VideoSource, split: SplitResult, out_dir: Path,
                           info: VideoInfo, cancel: CancellationToken | None,
                           progress: ProgressCallback) -> list[SequenceOutcome]:
    config = source.config
    total = max(1, info.frame_count or 1)
    with source.reader(cancel) as reader:
        regions = _segment_streaming(reader, split, config, cancel,
                                     _scaled(progress, 0.0, 0.25), total)
        # Seules les frames retenues pour la fusion sont converties (résolution native).
        needed: set[int] = set()
        for k, item in enumerate(split.sequences):
            region = regions.get(k)
            try:
                needed.update(plan_fusion(item.registration, config,
                                          None if region is None else region.bounds())[2])
            except (ValueError, MemoryError) as exc:
                logger.warning("Séquence %d : planification de la fusion impossible (%s)", k, exc)
        stream = _SharedStream(reader.frames(compute_proxy=False, wanted=needed))
        outcomes = []
        n = len(split.sequences)
        for k, item in enumerate(split.sequences):
            if cancel is not None:
                cancel.raise_if_cancelled()
            frames = stream.take(item.sequence.start_idx, item.sequence.end_idx)
            outcomes.append(_fuse_and_export(frames, k, item, regions.get(k), config, out_dir,
                                             source.path.stem, info, cancel,
                                             _scaled(progress, 0.25 + 0.75 * k / n, 0.75 / n)))
            for _ in frames:  # consomme le reste de la séquence (cas d'échec)
                pass
    return outcomes


# ---------------------------------------------------------------------------
# Vidéo
# ---------------------------------------------------------------------------


def _run_split(source: _VideoSource, info: VideoInfo, pool: WorkerPool,
               cancel: CancellationToken | None,
               progress: ProgressCallback) -> tuple[SplitResult, int]:
    """Découpage (par tronçons si possible) ; renvoie le résultat et le nombre de tronçons."""
    index = source.frame_index
    if index is None or pool.inline:
        n_frames = len(index) if index is not None else max(1, info.frame_count or 1)
        chunks = [(0, n_frames)]
    else:
        chunks = plan_chunks(len(index), info.fps, source.config, pool.workers)
    if len(chunks) == 1:
        with source.reader(cancel) as reader:
            total = max(1, len(index) if index is not None else info.frame_count or 1)

            def frames() -> Iterator[FrameObs]:
                for frame in reader.frames(proxy_only=True):
                    progress((frame.index + 1) / total, f"découpage {frame.index}")
                    yield frame

            config = copy.deepcopy(source.config)
            config.scenes.min_sequence_frames = 1
            split = split_and_register(frames(), config, cancel)
    else:
        logger.info("%s : découpage en %d tronçons sur %d processus", source.path.name,
                    len(chunks), pool.workers)
        futures = {pool.submit(_split_chunk, source, a, b): i for i, (a, b) in enumerate(chunks)}
        parts: dict[int, SplitResult] = {}

        def done(future: Future[SplitResult]) -> None:
            parts[futures[future]] = future.result()
            progress(len(parts) / len(chunks), f"découpage : tronçon {len(parts)}/{len(chunks)}")

        collect(futures, cancel, done)
        split = _stitch(source, [parts[i] for i in range(len(chunks))], cancel)
    drop_short_sequences(split, source.config.scenes.min_sequence_frames)
    return split, len(chunks)


def process_video(
    path: Path,
    output_dir: Path,
    config: PipelineConfig,
    progress: ProgressCallback | None = None,
    cancel: CancellationToken | None = None,
    pool: WorkerPool | None = None,
) -> VideoOutcome:
    """Traite une vidéo ; les erreurs sont capturées dans le résultat (sauf l'annulation).

    ``pool`` permet de partager les processus entre plusieurs vidéos (lot) ; sans
    pool, un pool de ``runtime.num_workers`` processus est créé pour la vidéo.
    """
    start = time.perf_counter()
    outcome = VideoOutcome(path=Path(path))
    own_pool: WorkerPool | None = None
    try:
        config.validate()
        if pool is None:
            own_pool = pool = WorkerPool(resolve_num_workers(config.runtime.num_workers),
                                         config.runtime.log_level)
        outcome.workers = pool.workers
        out_dir = output_dir / outcome.path.stem
        frame_index = scan_frame_index(outcome.path)
        source = _VideoSource(outcome.path, frame_index, config)
        with source.reader(cancel) as probe:
            info = probe.info
            if probe.frame_index is None:
                frame_index = None
                source = _VideoSource(outcome.path, None, config)
        outcome.info = info
        if frame_index is None:
            logger.warning("%s : index des frames indisponible, traitement séquentiel",
                           outcome.path.name)

        split, outcome.chunks = _run_split(source, info, pool, cancel,
                                           _scaled(progress, 0.0, _SPLIT_SHARE))
        outcome.split = split
        if not split.sequences:
            raise ValueError("Aucune séquence exploitable dans la vidéo")
        report = _scaled(progress, _SPLIT_SHARE, 1.0 - _SPLIT_SHARE)
        if frame_index is None:
            outcome.sequences = _reconstruct_streaming(source, split, out_dir, info, cancel,
                                                       report)
        else:
            n = len(split.sequences)
            results: dict[int, SequenceOutcome] = {}
            if pool.inline:
                for k, item in enumerate(split.sequences):
                    results[k] = _reconstruct_sequence(source, k, item, out_dir, info, cancel,
                                                       _scaled(report, k / n, 1.0 / n))
            else:
                futures = {pool.submit(_reconstruct_sequence, source, k, item, out_dir, info): k
                           for k, item in enumerate(split.sequences)}

                def done(future: Future[SequenceOutcome]) -> None:
                    results[futures[future]] = future.result()
                    report(len(results) / n, f"séquences reconstruites : {len(results)}/{n}")

                collect(futures, cancel, done)
            outcome.sequences = [results[k] for k in range(n)]
        write_json(out_dir / f"{outcome.path.stem}{SEQUENCES_SUFFIX}", outcome.to_dict())
        if progress is not None:
            progress(1.0, "terminé")
    except OperationCancelled:
        if pool is not None:
            pool.cancel_all()
        raise
    except Exception as exc:  # une vidéo en échec ne doit jamais interrompre le lot
        outcome.error = f"{type(exc).__name__}: {exc}"
        outcome.traceback = traceback.format_exc()
        logger.error("Échec du traitement de %s : %s\n%s", outcome.path, outcome.error,
                     outcome.traceback)
    finally:
        if own_pool is not None:
            own_pool.close()
    outcome.elapsed_s = time.perf_counter() - start
    return outcome
