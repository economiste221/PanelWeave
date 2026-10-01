"""Orchestration du traitement d'une vidéo (sans dépendance à l'interface).

État actuel (phase 4) : la vidéo entière est traitée comme **une** séquence.
Passe 1 : décodage et recalage (images réduites, en flux). Passe 2 : nouveau
décodage des frames recalées à la résolution native et fusion. Puis export.
Si le recalage est interrompu (changement de panel), seules les frames
recalées avant l'interruption sont fusionnées et l'interruption est rapportée ;
le découpage en séquences (phase 5) remplacera ce comportement.

Les callbacks de progression reçoivent ``(fraction ∈ [0, 1], message)`` ;
l'annulation est coopérative (vérifiée entre les frames et entre les tuiles).
"""

from __future__ import annotations

import logging
import time
import traceback
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from panelrecon.core.config import PipelineConfig
from panelrecon.core.export import ExportedFiles, export_sequence, sequence_basename
from panelrecon.core.models import (
    CancellationToken,
    FrameObs,
    OperationCancelled,
    Sequence,
    VideoInfo,
)
from panelrecon.core.mosaic import build_mosaic
from panelrecon.core.registration import PanelMaskProvider, register_frames
from panelrecon.core.video_io import VideoReader

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]


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
    error: str | None = None
    traceback: str | None = None
    elapsed_s: float = 0.0

    @property
    def ok(self) -> bool:
        return self.error is None and all(s.error is None for s in self.sequences)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "info": None if self.info is None else self.info.to_dict(),
            "sequences": [s.to_dict() for s in self.sequences],
            "error": self.error,
            "elapsed_s": round(self.elapsed_s, 3),
        }


def _scaled(progress: ProgressCallback | None, start: float, span: float) -> ProgressCallback:
    def report(fraction: float, message: str) -> None:
        if progress is not None:
            progress(start + span * min(1.0, max(0.0, fraction)), message)

    return report


def process_video(
    path: Path,
    output_dir: Path,
    config: PipelineConfig,
    progress: ProgressCallback | None = None,
    cancel: CancellationToken | None = None,
    panel_masks: PanelMaskProvider | None = None,
) -> VideoOutcome:
    """Traite une vidéo ; les erreurs sont capturées dans le résultat (sauf l'annulation)."""
    start = time.perf_counter()
    outcome = VideoOutcome(path=Path(path))
    report = _scaled(progress, 0.0, 1.0)
    try:
        config.validate()
        with VideoReader(outcome.path, config.video, config.preprocess, cancel=cancel) as reader:
            info = reader.info
            outcome.info = info
            total = max(1, info.frame_count or 1)

            def registration_frames() -> Iterator[FrameObs]:
                for frame in reader.frames():
                    report(0.45 * min(1.0, (frame.index + 1) / total), f"recalage {frame.index}")
                    yield frame

            registration = register_frames(registration_frames(), config, panel_masks, cancel)
            indices = registration.registered_indices
            if len(indices) < 1:
                raise ValueError("Aucune frame recalée")
            sequence = Sequence(indices[0], indices[-1], tuple(
                i for i in registration.excluded if indices[0] <= i <= indices[-1]))
            seq_outcome = SequenceOutcome(0, sequence, len(indices))
            if registration.interrupted:
                message = (f"Recalage interrompu ({registration.interruption_reason}) : seules les "
                           f"frames {indices[0]}–{indices[-1]} sont fusionnées")
                seq_outcome.warnings.append(message)
                logger.warning("%s : %s", outcome.path.name, message)
            native_frames = reader.frames(
                start_index=indices[0], stop_index=indices[-1] + 1, compute_proxy=False
            )
            mosaic = build_mosaic(native_frames, registration, config, panel_masks, cancel,
                                  _scaled(progress, 0.45, 0.5))
        out_dir = output_dir / outcome.path.stem
        basename = sequence_basename(outcome.path.stem, 0, mosaic.sequence)
        seq_outcome.files = export_sequence(mosaic, registration, out_dir, basename, config, info)
        outcome.sequences.append(seq_outcome)
        report(1.0, "terminé")
    except OperationCancelled:
        raise
    except Exception as exc:  # une vidéo en échec ne doit jamais interrompre le lot
        outcome.error = f"{type(exc).__name__}: {exc}"
        outcome.traceback = traceback.format_exc()
        logger.error("Échec du traitement de %s : %s\n%s", outcome.path, outcome.error,
                     outcome.traceback)
    outcome.elapsed_s = time.perf_counter() - start
    return outcome
