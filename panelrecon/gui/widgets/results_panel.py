"""Résultats d'une vidéo : liste des panels reconstruits (vignette, verdict) et
visionneuse (panel, carte de couverture ou frame de la vidéo)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from PyQt5.QtCore import QSize, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QIcon, QPixmap
from PyQt5.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from panelrecon.core.pipeline import SEQUENCES_SUFFIX
from panelrecon.gui.style import VERDICT_COLORS
from panelrecon.gui.widgets.image_viewer import ImageViewer

logger = logging.getLogger(__name__)

VIEW_PANEL = "Panel reconstruit"
VIEW_COVERAGE = "Carte de couverture"
VIEW_FRAME = "Frame de la vidéo"
_THUMB = 96


def load_sequences(video_dir: Path, stem: str) -> list[dict[str, Any]]:
    """Séquences exportées d'une vidéo (``<dossier>/<vidéo>_sequences.json``)."""
    path = video_dir / f"{stem}{SEQUENCES_SUFFIX}"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.debug("Résultats illisibles %s : %s", path, exc)
        return []
    sequences = data.get("sequences", [])
    return [s for s in sequences if isinstance(s, dict)]


class ResultsPanel(QWidget):
    """Panneau des résultats ; ``frame_requested(temps_s)`` demande un aperçu vidéo."""

    frame_requested = pyqtSignal(float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.viewer = ImageViewer()
        self.list = QListWidget()
        self.list.setIconSize(QSize(_THUMB, _THUMB))
        self.list.setMinimumWidth(220)
        self.mode = QComboBox()
        self.mode.addItems([VIEW_PANEL, VIEW_COVERAGE, VIEW_FRAME])
        self.info = QLabel("Aucune vidéo sélectionnée")
        self.info.setWordWrap(True)
        self.fit_button = QPushButton("Ajuster")
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.setEnabled(False)
        self.slider.setToolTip("Instant de la frame affichée (aperçu sans traitement)")
        self._duration_s = 0.0

        top = QHBoxLayout()
        top.addWidget(self.mode)
        top.addWidget(self.fit_button)
        top.addStretch(1)
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addLayout(top)
        right_layout.addWidget(self.viewer, 1)
        right_layout.addWidget(self.slider)
        right_layout.addWidget(self.info)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.list)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

        self.list.currentItemChanged.connect(lambda *_: self._show_current())
        self.mode.currentTextChanged.connect(lambda *_: self._on_mode())
        self.fit_button.clicked.connect(self.viewer.fit)
        self.slider.sliderReleased.connect(self._request_frame)

    # ----------------------------------------------------------------- données
    def set_video(self, video_dir: Path | None, stem: str, duration_s: float | None) -> None:
        """Affiche les résultats de la vidéo (``video_dir`` = dossier de sortie)."""
        self.list.clear()
        self._duration_s = duration_s or 0.0
        self.slider.setEnabled(self._duration_s > 0)
        self.slider.setRange(0, int(self._duration_s * 10))
        sequences = load_sequences(video_dir, stem) if video_dir is not None else []
        for seq in sequences:
            files = seq.get("files") or {}
            verdict = seq.get("verdict") or ("ÉCHEC" if seq.get("error") else "")
            span = seq.get("sequence", {})
            text = (f"Panel {seq.get('number', 0) + 1}  —  {verdict}\n"
                    f"frames {span.get('start_idx', '?')}–{span.get('end_idx', '?')}")
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, seq)
            image = files.get("image")
            if image:
                pixmap = QPixmap(str(image))
                if not pixmap.isNull():
                    item.setIcon(QIcon(pixmap.scaled(_THUMB, _THUMB, Qt.AspectRatioMode.KeepAspectRatio,
                                                     Qt.TransformationMode.SmoothTransformation)))
            if verdict in VERDICT_COLORS:
                item.setForeground(QColor(VERDICT_COLORS[verdict]))
            self.list.addItem(item)
        if sequences:
            self.info.setText(f"{len(sequences)} panel(s) reconstruit(s)")
            self.list.setCurrentRow(0)
        else:
            self.info.setText("Pas encore de résultat pour cette vidéo")
            self.viewer.clear()
            if self.mode.currentText() == VIEW_FRAME:
                self._request_frame()

    def show_frame(self, image: Any, index: int) -> None:
        """Affiche une frame décodée (aperçu)."""
        if self.mode.currentText() != VIEW_FRAME:
            self.mode.setCurrentText(VIEW_FRAME)
        self.viewer.set_image(image)
        self.info.setText(f"Frame {index}")

    # ---------------------------------------------------------------- internes
    def _on_mode(self) -> None:
        if self.mode.currentText() == VIEW_FRAME:
            self._request_frame()
        else:
            self._show_current()

    def _request_frame(self) -> None:
        if self._duration_s > 0 or self.slider.isEnabled():
            self.frame_requested.emit(self.slider.value() / 10.0)

    def _show_current(self) -> None:
        item = self.list.currentItem()
        if item is None or self.mode.currentText() == VIEW_FRAME:
            return
        seq: dict[str, Any] = item.data(Qt.ItemDataRole.UserRole)
        files = seq.get("files") or {}
        key = "image" if self.mode.currentText() == VIEW_PANEL else "coverage_color"
        path = files.get(key)
        if path and self.viewer.load(Path(path)):
            reasons = seq.get("quality_reasons") or []
            self.info.setText(
                f"{Path(path).name}\nVerdict : {seq.get('verdict') or '?'}"
                + ("" if not reasons else "\n" + "\n".join(f"• {r}" for r in reasons)))
        else:
            self.viewer.clear()
            self.info.setText(seq.get("error") or "Fichier indisponible")
