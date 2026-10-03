"""Console de journal alimentée par un ``logging.Handler`` relié à un signal Qt."""

from __future__ import annotations

import logging

from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QPlainTextEdit, QWidget

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s : %(message)s"


class _Emitter(QObject):
    message = pyqtSignal(str)


class QtLogHandler(logging.Handler):
    """Transmet les enregistrements de journal (de n'importe quel thread) au thread GUI."""

    def __init__(self) -> None:
        super().__init__()
        self.emitter = _Emitter()
        self.setFormatter(logging.Formatter(_FORMAT, "%H:%M:%S"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.emitter.message.emit(self.format(record))
        except RuntimeError:  # widget détruit pendant la fermeture
            pass


class LogConsole(QPlainTextEdit):
    """Journal en lecture seule, borné en nombre de lignes."""

    def __init__(self, max_lines: int = 5000, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMaximumBlockCount(max_lines)
        font = QFont("Menlo")
        font.setStyleHint(QFont.Monospace)
        font.setPointSize(11)
        self.setFont(font)
        self.handler = QtLogHandler()
        self.handler.emitter.message.connect(self.appendPlainText)

    def attach(self, logger: logging.Logger, level: int) -> None:
        self.handler.setLevel(level)
        logger.addHandler(self.handler)

    def detach(self, logger: logging.Logger) -> None:
        logger.removeHandler(self.handler)
