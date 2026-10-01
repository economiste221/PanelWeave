"""Orchestration du traitement d'une vidéo (sans dépendance à l'interface).

Trois passes de décodage linéaires (aucune vidéo n'est gardée en mémoire) :

1. **Découpage et recalage** (images réduites) : séquences, transitions écartées,
   transformations de chaque frame vers le repère canonique de sa séquence.
2. **Segmentation** (images réduites) : emprise du panel de chaque séquence,
   estimée dans son repère canonique puis projetée dans chaque frame.
3. **Fusion** (résolution native) : mosaïque de chaque séquence, puis export.

Une erreur sur une séquence est capturée et n'empêche pas les autres ; une
erreur sur la vidéo est capturée dans le résultat (seule l'annulation remonte).
Les callbacks de progression reçoivent ``(fraction ∈ [0, 1], message)``.
"""

from __future__ import annotations

import bisect
import logging
import time
import traceback
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from panelrecon.core.config import PipelineConfig
from panelrecon.core.export import (
    ExportedFiles,
    export_sequence,
    sequence_basename,
    write_json,
)
from panelrecon.core.models import (
    CancellationToken,
    FrameObs,
    OperationCancelled,
    Sequence,
    VideoInfo,
)
from panelrecon.core.mosaic import build_mosaic
from panelrecon.core.scene_split import SplitResult, split_and_register
from panelrecon.core.segmentation import PanelRegion, PanelRegionEstimator
from panelrecon.core.video_io import VideoReader

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]
SEQUENCES_SUFFIX = "_sequences.json"


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
        }


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


def _segment(
    reader: VideoReader,
    split: SplitResult,
    config: PipelineConfig,
    cancel: CancellationToken | None,
    progress: ProgressCallback,
    total: int,
) -> dict[int, PanelRegion]:
    """Passe 2 : emprise du panel de chaque séquence."""
    if config.segmentation.method == "none" or not split.sequences:
        return {}
    starts = [s.sequence.start_idx for s in split.sequences]
    estimators: dict[int, PanelRegionEstimator] = {}
    for frame in reader.frames():
        if cancel is not None:
            cancel.raise_if_cancelled()
        progress((frame.index + 1) / total, f"segmentation {frame.index}")
        k = bisect.bisect_right(starts, frame.index) - 1
        if k < 0 or frame.index > split.sequences[k].sequence.end_idx:
            continue
        estimator = estimators.get(k)
        if estimator is None:
            registration = split.sequences[k].registration
            estimator = PanelRegionEstimator(config, registration.transforms,
                                             registration.frame_sizes, frame.proxy_factor)
            estimators[k] = estimator
        estimator.add(frame)
    regions: dict[int, PanelRegion] = {}
    for k, estimator in estimators.items():
        regions[k] = estimator.estimate()
        if not regions[k].segmented:
            logger.warning("Séquence %d : panel non segmenté (frame entière)", k)
    return regions


def process_video(
    path: Path,
    output_dir: Path,
    config: PipelineConfig,
    progress: ProgressCallback | None = None,
    cancel: CancellationToken | None = None,
) -> VideoOutcome:
    """Traite une vidéo ; les erreurs sont capturées dans le résultat (sauf l'annulation)."""
    start = time.perf_counter()
    outcome = VideoOutcome(path=Path(path))
    try:
        config.validate()
        out_dir = output_dir / outcome.path.stem
        with VideoReader(outcome.path, config.video, config.preprocess, cancel=cancel) as reader:
            info = reader.info
            outcome.info = info
            total = max(1, info.frame_count or 1)
            report_split = _scaled(progress, 0.0, 0.4)

            def split_frames() -> Iterator[FrameObs]:
                for frame in reader.frames():
                    report_split((frame.index + 1) / total, f"découpage {frame.index}")
                    yield frame

            split = split_and_register(split_frames(), config, cancel)
            outcome.split = split
            if not split.sequences:
                raise ValueError("Aucune séquence exploitable dans la vidéo")
            regions = _segment(reader, split, config, cancel, _scaled(progress, 0.4, 0.15), total)

            stream = _SharedStream(reader.frames(compute_proxy=False))
            n = len(split.sequences)
            for k, item in enumerate(split.sequences):
                if cancel is not None:
                    cancel.raise_if_cancelled()
                sequence, registration = item.sequence, item.registration
                seq_outcome = SequenceOutcome(k, sequence, len(registration.transforms))
                outcome.sequences.append(seq_outcome)
                region = regions.get(k)
                frames = stream.take(sequence.start_idx, sequence.end_idx)
                try:
                    mosaic = build_mosaic(
                        frames, registration, config,
                        panel_masks=None if region is None else region.mask,
                        cancel=cancel,
                        progress=_scaled(progress, 0.55 + 0.45 * k / n, 0.45 / n),
                        clip=None if region is None else region.bounds(),
                    )
                    extra = {"segmentation": None if region is None else {
                        "segmented": region.segmented,
                        "panel_polygon_canonical": region.polygon.tolist(),
                    }}
                    seq_outcome.files = export_sequence(
                        mosaic, registration, out_dir, sequence_basename(outcome.path.stem, k,
                                                                         sequence),
                        config, info, extra,
                    )
                    if region is not None and not region.segmented:
                        seq_outcome.warnings.append("panel non segmenté : frame entière utilisée")
                except OperationCancelled:
                    raise
                except Exception as exc:  # une séquence en échec n'arrête pas les autres
                    seq_outcome.error = f"{type(exc).__name__}: {exc}"
                    logger.error("Séquence %d de %s en échec : %s\n%s", k, outcome.path.name,
                                 seq_outcome.error, traceback.format_exc())
                    for _ in frames:  # consomme le reste de la séquence
                        pass
        write_json(out_dir / f"{outcome.path.stem}{SEQUENCES_SUFFIX}", outcome.to_dict())
        if progress is not None:
            progress(1.0, "terminé")
    except OperationCancelled:
        raise
    except Exception as exc:  # une vidéo en échec ne doit jamais interrompre le lot
        outcome.error = f"{type(exc).__name__}: {exc}"
        outcome.traceback = traceback.format_exc()
        logger.error("Échec du traitement de %s : %s\n%s", outcome.path, outcome.error,
                     outcome.traceback)
    outcome.elapsed_s = time.perf_counter() - start
    return outcome
