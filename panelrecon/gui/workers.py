"""Travaux en arrière-plan de l'interface (aucun calcul dans le thread GUI).

* :class:`BatchWorker` (``QObject`` déplacé dans un ``QThread``) traite une liste
  de vidéos avec un pool de processus partagé (comme la CLI) et publie la
  progression, les résultats et la fin par signaux. L'annulation est coopérative
  (drapeau vérifié entre les frames, propagé aux processus).
* :class:`FramePreviewTask` (``QRunnable``) décode une frame isolée d'une vidéo
  pour l'aperçu.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PyQt5.QtCore import QObject, QRunnable, pyqtSignal, pyqtSlot

from panelrecon.core.config import PipelineConfig
from panelrecon.core.hardware import resolve_num_workers
from panelrecon.core.models import CancellationToken, OperationCancelled
from panelrecon.core.parallel import WorkerPool
from panelrecon.core.pipeline import VideoOutcome, process_video
from panelrecon.core.video_io import FrameIndex, VideoReader, scan_frame_index

logger = logging.getLogger(__name__)

_PROGRESS_INTERVAL_S = 0.1


class BatchWorker(QObject):
    """Traite des vidéos ``(ligne, chemin)`` ; à placer dans un ``QThread``."""

    video_started = pyqtSignal(int)
    video_progress = pyqtSignal(int, float, str)
    video_finished = pyqtSignal(int, object)  # VideoOutcome
    video_cancelled = pyqtSignal(int)
    finished = pyqtSignal()
    failed = pyqtSignal(str)

    def __init__(self, jobs: list[tuple[int, Path]], output_dir: Path,
                 config: PipelineConfig) -> None:
        super().__init__()
        self._jobs = list(jobs)
        self._output_dir = output_dir
        self._config = config
        self._cancel = CancellationToken()
        self._pool: WorkerPool | None = None
        self._lock = threading.Lock()

    @property
    def cancelled(self) -> bool:
        return self._cancel.cancelled

    def cancel(self) -> None:
        """Demande l'arrêt (appelable depuis le thread GUI)."""
        self._cancel.cancel()
        with self._lock:
            if self._pool is not None:
                self._pool.cancel_all()

    @pyqtSlot()
    def run(self) -> None:
        try:
            workers = resolve_num_workers(self._config.runtime.num_workers)
            with WorkerPool(workers, self._config.runtime.log_level) as pool:
                with self._lock:
                    self._pool = pool
                if self._cancel.cancelled:
                    pool.cancel_all()
                concurrent = 1 if pool.inline else max(1, min(len(self._jobs), pool.workers))
                with ThreadPoolExecutor(max_workers=concurrent,
                                        thread_name_prefix="gui-video") as threads:
                    futures = [threads.submit(self._run_one, row, path, pool)
                               for row, path in self._jobs]
                    for future in futures:
                        future.result()
        except Exception as exc:  # erreur inattendue du lot : signalée, jamais de plantage
            logger.exception("Erreur du traitement par lot")
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        finally:
            with self._lock:
                self._pool = None
            self.finished.emit()

    def _run_one(self, row: int, path: Path, pool: WorkerPool) -> None:
        if self._cancel.cancelled:
            self.video_cancelled.emit(row)
            return
        self.video_started.emit(row)
        last = [0.0]

        def progress(fraction: float, message: str) -> None:
            now = time.monotonic()
            if now - last[0] >= _PROGRESS_INTERVAL_S or fraction >= 1.0:
                last[0] = now
                self.video_progress.emit(row, fraction, message)

        try:
            outcome: VideoOutcome = process_video(path, self._output_dir, self._config,
                                                  progress=progress, cancel=self._cancel,
                                                  pool=pool)
        except OperationCancelled:
            self.video_cancelled.emit(row)
            return
        self.video_finished.emit(row, outcome)


class _PreviewSignals(QObject):
    ready = pyqtSignal(int, object, int)  # requête, image BGR, indice de frame
    failed = pyqtSignal(int, str)


class FramePreviewTask(QRunnable):
    """Décode la frame la plus proche d'un instant donné (accès direct)."""

    def __init__(self, request: int, path: Path, time_s: float, config: PipelineConfig,
                 frame_index: FrameIndex | None = None) -> None:
        super().__init__()
        self.request = request
        self.path = path
        self.time_s = time_s
        self.config = config
        self.frame_index = frame_index
        self.signals = _PreviewSignals()

    def run(self) -> None:
        try:
            index = self.frame_index or scan_frame_index(self.path)
            cfg = self.config
            video = type(cfg.video)(**{**cfg.video.__dict__, "max_fps": 0.0, "frame_step": 1})
            with VideoReader(self.path, video, cfg.preprocess, frame_index=index) as reader:
                fps = reader.info.fps or 25.0
                target = max(0, int(round(self.time_s * fps)))
                if index is not None:
                    target = min(target, len(index) - 1)
                frame = next(iter(reader.frames(target, target + 1, compute_proxy=False)), None)
                if frame is None:
                    frame = next(iter(reader.frames(compute_proxy=False)), None)
            if frame is None:
                raise ValueError("aucune frame décodable")
            self.signals.ready.emit(self.request, frame.image, frame.index)
        except Exception as exc:  # aperçu impossible : signalé à l'interface
            self.signals.failed.emit(self.request, f"{type(exc).__name__}: {exc}")
