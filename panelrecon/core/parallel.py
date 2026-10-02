"""Exécution parallèle : pool de processus (méthode « spawn ») et repli en ligne.

* Les processus sont toujours créés par ``spawn`` (seule méthode sûre sur macOS,
  identique sur toutes les plateformes) ; les fonctions exécutées doivent donc
  être définies au niveau d'un module et leurs arguments être sérialisables.
* Chaque processus limite ses threads internes (OpenCV, fusion par tuiles) à sa
  part des cœurs : ``processus × threads ≤ cœurs``, pas de sursouscription.
* L'annulation passe par un ``Event`` partagé, vérifié par les tâches entre les
  frames (:func:`worker_cancel_token`).
* Avec un seul worker, les tâches s'exécutent dans le processus appelant
  (:class:`InlineExecutor`) : mêmes fonctions, aucun coût de démarrage.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
from collections.abc import Callable, Iterable
from concurrent.futures import FIRST_COMPLETED, Executor, Future, ProcessPoolExecutor, wait
from typing import Any, TypeVar

import cv2

from panelrecon.core.hardware import performance_core_count
from panelrecon.core.models import CancellationToken, OperationCancelled

logger = logging.getLogger(__name__)

T = TypeVar("T")

_WORKER_CANCEL: CancellationToken | None = None
_WORKER_THREADS: int = 0  # 0 : processus principal (pas de limite propre au worker)
_LOG_FORMAT = "%(asctime)s %(levelname)s [%(processName)s] %(name)s: %(message)s"


def _init_worker(cancel_flag: Any, threads: int, log_level: str) -> None:
    """Initialisation d'un processus de travail (exécutée une fois par processus)."""
    global _WORKER_CANCEL, _WORKER_THREADS
    _WORKER_CANCEL = CancellationToken(cancel_flag)
    _WORKER_THREADS = max(1, threads)
    cv2.setNumThreads(_WORKER_THREADS)
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=log_level, format=_LOG_FORMAT)
    root.setLevel(log_level)


def worker_cancel_token(fallback: CancellationToken | None = None) -> CancellationToken | None:
    """Jeton d'annulation de la tâche courante : celui du pool dans un processus de
    travail, ``fallback`` dans le processus principal."""
    return _WORKER_CANCEL if _WORKER_CANCEL is not None else fallback


def worker_threads(default: int) -> int:
    """Threads internes autorisés pour la tâche courante."""
    return _WORKER_THREADS if _WORKER_THREADS > 0 else default


class InlineExecutor(Executor):
    """Exécute chaque tâche immédiatement dans le processus appelant."""

    def submit(self, fn: Callable[..., T], /, *args: Any, **kwargs: Any) -> Future[T]:
        future: Future[T] = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as exc:  # transmis tel quel à l'appelant via result()
            future.set_exception(exc)
        return future


class WorkerPool:
    """Pool de processus « spawn » partageable entre plusieurs vidéos.

    Exemple ::

        with WorkerPool(4, "INFO") as pool:
            future = pool.submit(fonction_de_module, argument)
    """

    def __init__(self, workers: int, log_level: str = "INFO",
                 threads_per_worker: int | None = None) -> None:
        if workers < 1:
            raise ValueError(f"Nombre de workers invalide : {workers}")
        self.workers = workers
        context = multiprocessing.get_context("spawn")
        self._cancel_flag = context.Event()
        threads = threads_per_worker or max(1, performance_core_count() // workers)
        self._executor: Executor
        if workers == 1:
            self._executor = InlineExecutor()
        else:
            self._executor = ProcessPoolExecutor(
                max_workers=workers, mp_context=context, initializer=_init_worker,
                initargs=(self._cancel_flag, threads, log_level),
            )
        logger.debug("Pool de %d worker(s), %d thread(s) chacun (pid %d)", workers, threads,
                     os.getpid())

    @property
    def inline(self) -> bool:
        return isinstance(self._executor, InlineExecutor)

    def submit(self, fn: Callable[..., T], /, *args: Any) -> Future[T]:
        return self._executor.submit(fn, *args)

    def cancel_all(self) -> None:
        """Demande l'arrêt des tâches en cours (vérifié entre les frames)."""
        self._cancel_flag.set()

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)

    def __enter__(self) -> WorkerPool:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def collect(
    futures: Iterable[Future[T]],
    cancel: CancellationToken | None = None,
    on_done: Callable[[Future[T]], None] | None = None,
    poll_s: float = 0.25,
) -> None:
    """Attend des tâches, dans l'ordre de leur achèvement.

    ``on_done`` est appelée dans le thread appelant pour chaque tâche terminée.
    Si ``cancel`` est déclenché, les tâches non commencées sont annulées et
    :class:`OperationCancelled` est levée (les tâches en cours vérifient le
    drapeau du pool, que l'appelant doit lever via :meth:`WorkerPool.cancel_all`).
    """
    pending = set(futures)
    while pending:
        if cancel is not None and cancel.cancelled:
            for future in pending:
                future.cancel()
            raise OperationCancelled("Traitement annulé")
        done, pending = wait(pending, timeout=poll_s, return_when=FIRST_COMPLETED)
        for future in done:
            if on_done is not None:
                on_done(future)
