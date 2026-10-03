"""Tests de l'interface PyQt5 (sans écran) : éditeur de paramètres, file d'attente,
visionneuse, et traitement réel d'une vidéo sans bloquer le thread GUI."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PyQt5.QtCore import QSettings, QTimer
from PyQt5.QtWidgets import QLineEdit, QSpinBox
from pytestqt.qtbot import QtBot

from panelrecon.core.config import ConfigError, PipelineConfig
from panelrecon.gui.main_window import MainWindow
from panelrecon.gui.widgets.config_editor import ConfigEditor
from panelrecon.gui.widgets.image_viewer import ImageViewer, qimage_from_array
from panelrecon.gui.widgets.queue_model import Status, VideoQueueModel
from panelrecon.tests.conftest import SyntheticCache


def _window(qtbot: QtBot, tmp_path: Path, workers: int = 1) -> MainWindow:
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    window = MainWindow(settings)
    qtbot.addWidget(window)
    window._set_output(tmp_path / "out")
    spin = window.editor.widget_for("runtime", "num_workers")
    assert isinstance(spin, QSpinBox)
    spin.setValue(workers)
    return window


def test_config_editor_roundtrip(qtbot: QtBot) -> None:
    config = PipelineConfig()
    config.mosaic.max_observations = 7
    config.motion.detector = "orb"
    config.preprocess.exclusion_zones = ((0.0, 0.8, 1.0, 1.0),)
    editor = ConfigEditor(config)
    qtbot.addWidget(editor)
    assert editor.config().to_dict() == config.to_dict()
    editor.set_config(PipelineConfig())
    assert editor.config().to_dict() == PipelineConfig().to_dict()
    zones = editor.widget_for("preprocess", "exclusion_zones")
    assert isinstance(zones, QLineEdit)
    zones.setText("[[0, 0")
    with pytest.raises(ConfigError):
        editor.config()
    zones.setText("[[0.5, 0.5, 0.2, 0.2]]")  # rectangle inversé : refusé par la validation
    with pytest.raises(ConfigError):
        editor.config()


def test_queue_model(tmp_path: Path) -> None:
    model = VideoQueueModel()
    video = tmp_path / "a.mp4"
    video.write_bytes(b"")
    assert model.add(video, 75.0) and not model.add(video, 75.0)
    assert model.rowCount() == 1 and model.pending_rows() == [0]
    assert model.data(model.index(0, 1)) == "1:15"
    model.set_running(0)
    assert model.item(0).status == Status.RUNNING and model.pending_rows() == []
    model.remove_rows([0])  # une vidéo en cours n'est pas retirée
    assert model.rowCount() == 1
    model.set_status(0, Status.CANCELLED)
    model.remove_rows([0])
    assert model.rowCount() == 0


def test_image_viewer(qtbot: QtBot) -> None:
    viewer = ImageViewer()
    qtbot.addWidget(viewer)
    bgra = np.zeros((20, 30, 4), np.uint8)
    bgra[..., 2] = 255  # rouge
    bgra[:, :10, 3] = 255
    image = qimage_from_array(bgra)
    assert (image.width(), image.height()) == (30, 20)
    assert image.pixelColor(0, 0).red() == 255 and image.pixelColor(20, 0).alpha() == 0
    assert qimage_from_array(np.zeros((4, 6), np.uint8)).width() == 6
    viewer.set_image(bgra)
    assert viewer.has_image
    viewer.zoom(2.0)
    viewer.clear()
    assert not viewer.has_image


@pytest.mark.parametrize("workers", [1, 2])
def test_process_video_from_gui_keeps_ui_responsive(
    qtbot: QtBot, tmp_path: Path, synthetic: SyntheticCache, workers: int
) -> None:
    """Traitement réel depuis l'interface, en ligne (1) ou avec des processus (2)."""
    gt = synthetic.get("pan_horizontal")
    window = _window(qtbot, tmp_path, workers)
    assert window.add_paths([gt.video_path]) == 1
    assert window.add_paths([gt.video_path]) == 0  # pas de doublon
    ticks: list[int] = []
    timer = QTimer()
    timer.timeout.connect(lambda: ticks.append(1))
    timer.start(20)
    assert window.start()
    assert window.running and not window.start_action.isEnabled()
    qtbot.waitUntil(lambda: not window.running, timeout=180_000)
    timer.stop()
    # La boucle d'événements a continué de tourner pendant le traitement.
    assert len(ticks) >= 5
    item = window.model.item(0)
    assert item.status == Status.DONE and item.n_sequences == 1
    assert item.verdicts["OK"] == 1
    assert window.results.list.count() == 1
    assert window.results.viewer.has_image
    assert (tmp_path / "out" / "pan_horizontal").is_dir()


def test_cancel_from_gui(qtbot: QtBot, tmp_path: Path, synthetic: SyntheticCache) -> None:
    window = _window(qtbot, tmp_path)
    window.add_paths([synthetic.get("pan_vertical").video_path,
                      synthetic.get("zoom_in").video_path])
    assert window.start()
    window.cancel()
    qtbot.waitUntil(lambda: not window.running, timeout=180_000)
    statuses = {window.model.item(r).status for r in range(window.model.rowCount())}
    assert Status.RUNNING not in statuses and Status.CANCELLED in statuses
    assert window.start_action.isEnabled()


def test_frame_preview(qtbot: QtBot, tmp_path: Path, synthetic: SyntheticCache) -> None:
    window = _window(qtbot, tmp_path)
    window.add_paths([synthetic.get("short").video_path])
    window.table.selectRow(0)
    with qtbot.waitSignal(window.results.frame_requested, timeout=5000):
        window.results.mode.setCurrentText("Frame de la vidéo")
    qtbot.waitUntil(lambda: window.results.viewer.has_image, timeout=30_000)
    assert window.results.info.text().startswith("Frame")
