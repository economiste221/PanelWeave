"""Modèle de la file d'attente des vidéos (``QTableView``)."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from PyQt5.QtCore import QAbstractTableModel, QModelIndex, QObject, Qt
from PyQt5.QtGui import QBrush, QColor

from panelrecon.core.pipeline import VideoOutcome
from panelrecon.gui.style import VERDICT_COLORS


class Status(str, Enum):
    WAITING = "En attente"
    RUNNING = "En cours"
    DONE = "Terminé"
    FAILED = "Échec"
    CANCELLED = "Annulé"


@dataclass
class QueueItem:
    path: Path
    duration_s: float | None = None
    status: Status = Status.WAITING
    progress: float = 0.0
    message: str = ""
    n_sequences: int | None = None
    verdicts: Counter[str] = field(default_factory=Counter)
    output_dir: Path | None = None
    error: str | None = None
    elapsed_s: float | None = None

    def verdict_summary(self) -> str:
        if not self.verdicts:
            return ""
        return "  ".join(f"{name} {count}" for name, count in sorted(self.verdicts.items()))

    def worst_verdict(self) -> str | None:
        for name in ("ÉCHEC", "À VÉRIFIER", "OK"):
            if self.verdicts.get(name):
                return name
        return None


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "?"
    total = int(round(seconds))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


class VideoQueueModel(QAbstractTableModel):
    """Une ligne par vidéo : nom, durée, statut, progression, séquences, verdicts."""

    COLUMNS = ("Vidéo", "Durée", "Statut", "Progression", "Séquences", "Verdicts", "Temps")

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._items: list[QueueItem] = []

    # ------------------------------------------------------------------ accès
    @property
    def items(self) -> list[QueueItem]:
        return list(self._items)

    def item(self, row: int) -> QueueItem:
        return self._items[row]

    def paths(self) -> set[Path]:
        return {item.path for item in self._items}

    def add(self, path: Path, duration_s: float | None) -> bool:
        """Ajoute une vidéo (ignorée si déjà présente) ; renvoie ``True`` si ajoutée."""
        path = path.resolve()
        if path in self.paths():
            return False
        row = len(self._items)
        self.beginInsertRows(QModelIndex(), row, row)
        self._items.append(QueueItem(path, duration_s))
        self.endInsertRows()
        return True

    def remove_rows(self, rows: list[int]) -> None:
        for row in sorted(set(rows), reverse=True):
            if 0 <= row < len(self._items) and self._items[row].status != Status.RUNNING:
                self.beginRemoveRows(QModelIndex(), row, row)
                del self._items[row]
                self.endRemoveRows()

    def pending_rows(self) -> list[int]:
        return [i for i, item in enumerate(self._items)
                if item.status in (Status.WAITING, Status.FAILED, Status.CANCELLED)]

    # ---------------------------------------------------------- mises à jour
    def _changed(self, row: int) -> None:
        self.dataChanged.emit(self.index(row, 0), self.index(row, len(self.COLUMNS) - 1))

    def set_running(self, row: int) -> None:
        item = self._items[row]
        item.status, item.progress, item.message, item.error = Status.RUNNING, 0.0, "", None
        item.verdicts.clear()
        item.n_sequences = None
        self._changed(row)

    def set_progress(self, row: int, fraction: float, message: str) -> None:
        item = self._items[row]
        item.progress, item.message = fraction, message
        self.dataChanged.emit(self.index(row, 3), self.index(row, 3))

    def set_outcome(self, row: int, outcome: VideoOutcome, output_dir: Path) -> None:
        item = self._items[row]
        item.elapsed_s = outcome.elapsed_s
        item.output_dir = output_dir / outcome.path.stem
        item.n_sequences = len(outcome.sequences)
        item.verdicts = Counter(
            s.verdict.value if s.verdict is not None else "ÉCHEC" for s in outcome.sequences)
        item.error = outcome.error
        item.status = Status.DONE if outcome.error is None else Status.FAILED
        item.progress = 1.0 if outcome.error is None else item.progress
        if outcome.info is not None:
            item.duration_s = outcome.info.duration_s
        self._changed(row)

    def set_status(self, row: int, status: Status, message: str = "") -> None:
        item = self._items[row]
        item.status, item.message = status, message
        self._changed(row)

    # ------------------------------------------------------------- Qt (modèle)
    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802, B008
        return 0 if parent.isValid() else len(self._items)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802, B008
        return 0 if parent.isValid() else len(self.COLUMNS)

    def headerData(self, section: int, orientation: Qt.Orientation,  # noqa: N802
                   role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self.COLUMNS[section]
        return None

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid():
            return None
        item = self._items[index.row()]
        column = index.column()
        if role == Qt.ItemDataRole.DisplayRole:
            if column == 0:
                return item.path.name
            if column == 1:
                return _format_duration(item.duration_s)
            if column == 2:
                return item.status.value
            if column == 3:
                return f"{100.0 * item.progress:.0f} %"
            if column == 4:
                return "" if item.n_sequences is None else str(item.n_sequences)
            if column == 5:
                return item.verdict_summary()
            if column == 6:
                return "" if item.elapsed_s is None else _format_duration(item.elapsed_s)
        if role == Qt.ItemDataRole.UserRole and column == 3:
            return item.progress
        if role == Qt.ItemDataRole.ToolTipRole:
            details = [str(item.path)]
            if item.message:
                details.append(item.message)
            if item.error:
                details.append(item.error)
            return "\n".join(details)
        if role == Qt.ItemDataRole.ForegroundRole and column == 5:
            worst = item.worst_verdict()
            if worst is not None:
                return QBrush(QColor(VERDICT_COLORS[worst]))
        if role == Qt.ItemDataRole.ForegroundRole and column == 2 and item.status == Status.FAILED:
            return QBrush(QColor(VERDICT_COLORS["ÉCHEC"]))
        return None
