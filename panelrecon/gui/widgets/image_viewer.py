"""Visionneuse d'images : zoom à la molette, déplacement à la souris, damier sous la
transparence (les pixels jamais observés du panel ont alpha = 0)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PyQt5.QtCore import QRectF, Qt
from PyQt5.QtGui import QBrush, QColor, QImage, QPainter, QPixmap, QWheelEvent
from PyQt5.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QGraphicsView, QWidget

from panelrecon.core.models import ImageU8


def _checkerboard(cell: int = 12) -> QBrush:
    pixmap = QPixmap(2 * cell, 2 * cell)
    pixmap.fill(QColor("#3a3a3a"))
    painter = QPainter(pixmap)
    painter.fillRect(0, 0, cell, cell, QColor("#2a2a2a"))
    painter.fillRect(cell, cell, cell, cell, QColor("#2a2a2a"))
    painter.end()
    return QBrush(pixmap)


def qimage_from_array(image: ImageU8) -> QImage:
    """Copie d'une image OpenCV (gris, BGR ou BGRA, uint8) en ``QImage``."""
    array = np.ascontiguousarray(image)
    h, w = array.shape[:2]
    if array.ndim == 2:
        return QImage(array.tobytes(), w, h, w, QImage.Format_Grayscale8).copy()
    if array.shape[2] == 3:
        rgb = np.ascontiguousarray(array[..., ::-1])
        return QImage(rgb.tobytes(), w, h, 3 * w, QImage.Format_RGB888).copy()
    rgba = np.ascontiguousarray(array[..., [2, 1, 0, 3]])
    return QImage(rgba.tobytes(), w, h, 4 * w, QImage.Format_RGBA8888).copy()


class ImageViewer(QGraphicsView):
    """Affiche une image ; molette = zoom autour du curseur, glisser = déplacer."""

    ZOOM_STEP = 1.25

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self._item = QGraphicsPixmapItem()
        self._item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self._scene.addItem(self._item)
        self.setScene(self._scene)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setRenderHints(QPainter.SmoothPixmapTransform | QPainter.Antialiasing)
        self.setBackgroundBrush(_checkerboard())
        self._has_image = False

    @property
    def has_image(self) -> bool:
        return self._has_image

    def clear(self) -> None:
        self._item.setPixmap(QPixmap())
        self._has_image = False

    def set_pixmap(self, pixmap: QPixmap, fit: bool = True) -> None:
        self._item.setPixmap(pixmap)
        self._scene.setSceneRect(QRectF(pixmap.rect()))
        self._has_image = not pixmap.isNull()
        if fit:
            self.fit()

    def set_image(self, image: ImageU8, fit: bool = True) -> None:
        self.set_pixmap(QPixmap.fromImage(qimage_from_array(image)), fit)

    def load(self, path: Path, fit: bool = True) -> bool:
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            self.clear()
            return False
        self.set_pixmap(pixmap, fit)
        return True

    def fit(self) -> None:
        if self._has_image:
            self.fitInView(self._item, Qt.AspectRatioMode.KeepAspectRatio)

    def zoom(self, factor: float) -> None:
        self.scale(factor, factor)

    def wheelEvent(self, event: QWheelEvent | None) -> None:  # noqa: N802 - API Qt
        if event is None or not self._has_image:
            return
        self.zoom(self.ZOOM_STEP if event.angleDelta().y() > 0 else 1.0 / self.ZOOM_STEP)
        event.accept()
