"""Fenêtre principale : file d'attente, résultats, paramètres, journal, progression."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from PyQt5.QtCore import QSettings, QStandardPaths, Qt, QThread, QThreadPool, QUrl
from PyQt5.QtGui import QCloseEvent, QDesktopServices, QDragEnterEvent, QDropEvent, QKeySequence
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QAction,
    QDockWidget,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTableView,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from panelrecon import __version__
from panelrecon.core.config import ConfigError, PipelineConfig
from panelrecon.core.pipeline import VideoOutcome
from panelrecon.core.video_io import VideoError, discover_videos, probe_video
from panelrecon.gui.widgets.config_editor import ConfigEditor
from panelrecon.gui.widgets.log_console import LogConsole
from panelrecon.gui.widgets.queue_model import Status, VideoQueueModel
from panelrecon.gui.widgets.results_panel import ResultsPanel
from panelrecon.gui.workers import BatchWorker, FramePreviewTask

logger = logging.getLogger(__name__)

SETTINGS_CONFIG = "config_json"
SETTINGS_OUTPUT = "output_dir"
SETTINGS_GEOMETRY = "geometry"
SETTINGS_STATE = "window_state"


def default_output_dir() -> Path:
    base = QStandardPaths.writableLocation(QStandardPaths.PicturesLocation) or str(Path.home())
    return Path(base) / "PanelRecon"


class MainWindow(QMainWindow):
    """Fenêtre de l'application ; tout calcul s'exécute hors du thread GUI."""

    def __init__(self, settings: QSettings | None = None) -> None:
        super().__init__()
        self.setWindowTitle(f"PanelRecon {__version__}")
        self.setAcceptDrops(True)
        self.resize(1400, 900)
        self.settings = settings or QSettings()
        self.model = VideoQueueModel(self)
        self.batch_thread: QThread | None = None
        self.worker: BatchWorker | None = None
        self._preview_request = 0
        self._running_rows: set[int] = set()

        # ------------------------------------------------------------- file
        self.table = QTableView()
        self.table.setModel(self.model)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setAlternatingRowColors(True)
        vertical = self.table.verticalHeader()
        assert vertical is not None
        vertical.setVisible(False)
        header = self.table.horizontalHeader()
        assert header is not None
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        for column in range(1, self.model.columnCount()):
            header.setSectionResizeMode(column, QHeaderView.ResizeToContents)
        selection = self.table.selectionModel()
        assert selection is not None
        selection.currentRowChanged.connect(lambda *_: self._show_selected())

        self.results = ResultsPanel()
        self.results.frame_requested.connect(self._request_preview)

        self.drop_hint = QLabel("Glissez-déposez des vidéos ou des dossiers ici")
        self.drop_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.output_label = QLabel()
        self.output_button = QPushButton("Dossier de sortie…")
        self.output_button.clicked.connect(self._choose_output)
        output_row = QHBoxLayout()
        output_row.addWidget(self.output_label, 1)
        output_row.addWidget(self.output_button)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(self.drop_hint)
        left_layout.addWidget(self.table, 1)
        left_layout.addLayout(output_row)
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(left)
        splitter.addWidget(self.results)
        splitter.setStretchFactor(1, 2)
        self.setCentralWidget(splitter)

        # --------------------------------------------------------- docks
        self.editor = ConfigEditor(self._load_config())
        params = QWidget()
        params_layout = QVBoxLayout(params)
        params_layout.addWidget(self.editor, 1)
        buttons = QHBoxLayout()
        for text, slot in (("Charger un profil…", self._load_profile),
                           ("Enregistrer le profil…", self._save_profile),
                           ("Valeurs par défaut", self._reset_config)):
            button = QPushButton(text)
            button.clicked.connect(slot)
            buttons.addWidget(button)
        params_layout.addLayout(buttons)
        self.params_dock = QDockWidget("Paramètres", self)
        self.params_dock.setObjectName("params_dock")
        self.params_dock.setWidget(params)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.params_dock)

        self.console = LogConsole()
        self.console.attach(logging.getLogger("panelrecon"), logging.INFO)
        self.log_dock = QDockWidget("Journal", self)
        self.log_dock.setObjectName("log_dock")
        self.log_dock.setWidget(self.console)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.log_dock)

        # ------------------------------------------------------ actions
        toolbar = QToolBar("Actions")
        toolbar.setObjectName("actions")
        self.addToolBar(toolbar)
        self.add_files_action = self._action(toolbar, "Ajouter des vidéos…", self._add_files,
                                             QKeySequence.Open)
        self.add_folder_action = self._action(toolbar, "Ajouter un dossier…", self._add_folder)
        self.remove_action = self._action(toolbar, "Retirer", self._remove_selected,
                                          QKeySequence.Delete)
        toolbar.addSeparator()
        self.start_action = self._action(toolbar, "Lancer", self.start)
        self.cancel_action = self._action(toolbar, "Annuler", self.cancel)
        toolbar.addSeparator()
        self.open_output_action = self._action(toolbar, "Ouvrir le dossier de sortie",
                                               self._open_output)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setMaximumWidth(360)
        self.status_label = QLabel("Prêt")
        status_bar = self.statusBar()
        assert status_bar is not None
        status_bar.addWidget(self.status_label, 1)
        status_bar.addPermanentWidget(self.progress)

        self._set_output(Path(str(self.settings.value(SETTINGS_OUTPUT,
                                                      str(default_output_dir())))))
        geometry = self.settings.value(SETTINGS_GEOMETRY)
        if geometry is not None:
            self.restoreGeometry(geometry)
        state = self.settings.value(SETTINGS_STATE)
        if state is not None:
            self.restoreState(state)
        self._update_actions()

    # ------------------------------------------------------------- utilitaires
    def _action(self, toolbar: QToolBar, text: str, slot: Callable[[], object],
                shortcut: QKeySequence.StandardKey | None = None) -> QAction:
        action = QAction(text, self)
        if shortcut is not None:
            action.setShortcut(QKeySequence(shortcut))
        action.triggered.connect(lambda _checked=False: None if slot() else None)
        toolbar.addAction(action)
        return action

    @property
    def running(self) -> bool:
        return self.batch_thread is not None

    @property
    def output_dir(self) -> Path:
        return self._output_dir

    def _set_output(self, path: Path) -> None:
        self._output_dir = path
        self.output_label.setText(f"Sortie : {path}")
        self.settings.setValue(SETTINGS_OUTPUT, str(path))

    def _load_config(self) -> PipelineConfig:
        text = self.settings.value(SETTINGS_CONFIG)
        if isinstance(text, str) and text:
            try:
                return PipelineConfig.from_json(text)
            except ConfigError as exc:
                logger.warning("Paramètres enregistrés ignorés : %s", exc)
        return PipelineConfig()

    def _update_actions(self) -> None:
        running = self.running
        has_rows = self.model.rowCount() > 0
        self.start_action.setEnabled(not running and bool(self.model.pending_rows()))
        self.cancel_action.setEnabled(running)
        self.remove_action.setEnabled(not running and has_rows)
        self.drop_hint.setVisible(not has_rows)
        params = self.params_dock.widget()
        if params is not None:
            params.setEnabled(not running)

    # ------------------------------------------------------------ file d'attente
    def add_paths(self, paths: list[Path]) -> int:
        """Ajoute les vidéos trouvées dans ``paths`` (fichiers ou dossiers)."""
        try:
            config = self.editor.config()
        except ConfigError:
            config = PipelineConfig()
        added = 0
        for video in discover_videos(paths, config.video.extensions, config.video.recursive):
            try:
                duration = probe_video(video, config.video).duration_s
            except VideoError as exc:
                logger.warning("%s : %s", video.name, exc)
                duration = None
            added += self.model.add(video, duration)
        if added:
            logger.info("%d vidéo(s) ajoutée(s)", added)
            if self.table.currentIndex().row() < 0:
                self.table.selectRow(0)
        self._update_actions()
        return added

    def _add_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self, "Ajouter des vidéos", str(Path.home()),
            "Vidéos (*.mp4 *.mkv *.webm *.mov *.MP4 *.MKV *.WEBM *.MOV)")
        self.add_paths([Path(f) for f in files])

    def _add_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Ajouter un dossier", str(Path.home()))
        if folder:
            self.add_paths([Path(folder)])

    def _remove_selected(self) -> None:
        selection = self.table.selectionModel()
        rows = [] if selection is None else [index.row() for index in selection.selectedRows()]
        self.model.remove_rows(rows)
        self._update_actions()

    def _choose_output(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Dossier de sortie", str(self.output_dir))
        if folder:
            self._set_output(Path(folder))

    def _open_output(self) -> None:
        row = self.table.currentIndex().row()
        target = self.output_dir
        if row >= 0 and self.model.item(row).output_dir is not None:
            target = self.model.item(row).output_dir or target
        target.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))

    # ------------------------------------------------------- glisser-déposer
    def dragEnterEvent(self, event: QDragEnterEvent | None) -> None:  # noqa: N802 - API Qt
        if event is not None and (mime := event.mimeData()) is not None and mime.hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent | None) -> None:  # noqa: N802 - API Qt
        mime = None if event is None else event.mimeData()
        if event is None or mime is None:
            return
        paths = [Path(url.toLocalFile()) for url in mime.urls() if url.isLocalFile()]
        if paths:
            self.add_paths(paths)
            event.acceptProposedAction()

    # ------------------------------------------------------------- profils
    def _load_profile(self) -> None:
        name, _ = QFileDialog.getOpenFileName(self, "Charger un profil", str(Path.home()),
                                              "Profils (*.json)")
        if not name:
            return
        try:
            self.editor.set_config(PipelineConfig.load(Path(name)))
        except ConfigError as exc:
            QMessageBox.warning(self, "Profil invalide", str(exc))

    def _save_profile(self) -> None:
        name, _ = QFileDialog.getSaveFileName(self, "Enregistrer le profil", str(Path.home()),
                                              "Profils (*.json)")
        if not name:
            return
        try:
            self.editor.config().save(Path(name))
        except ConfigError as exc:
            QMessageBox.warning(self, "Paramètres invalides", str(exc))

    def _reset_config(self) -> None:
        self.editor.set_config(PipelineConfig())

    # ------------------------------------------------------------ traitement
    def start(self) -> bool:
        """Lance le traitement des vidéos en attente ; ``False`` si rien n'est lancé."""
        if self.running:
            return False
        try:
            config = self.editor.config()
        except ConfigError as exc:
            QMessageBox.warning(self, "Paramètres invalides", str(exc))
            return False
        rows = self.model.pending_rows()
        if not rows:
            return False
        self.settings.setValue(SETTINGS_CONFIG, config.to_json())
        jobs = [(row, self.model.item(row).path) for row in rows]
        for row in rows:
            self.model.set_status(row, Status.WAITING)
        self._running_rows = set(rows)
        self.batch_thread = QThread(self)
        self.worker = BatchWorker(jobs, self.output_dir, config)
        self.worker.moveToThread(self.batch_thread)
        self.batch_thread.started.connect(self.worker.run)
        self.worker.video_started.connect(self._on_started)
        self.worker.video_progress.connect(self._on_progress)
        self.worker.video_finished.connect(self._on_finished)
        self.worker.video_cancelled.connect(self._on_cancelled)
        self.worker.failed.connect(self._on_failed)
        self.worker.finished.connect(self.batch_thread.quit)
        self.batch_thread.finished.connect(self._on_batch_done)
        self.progress.setValue(0)
        self.status_label.setText(f"Traitement de {len(jobs)} vidéo(s)…")
        self.batch_thread.start()
        self._update_actions()
        return True

    def cancel(self) -> None:
        if self.worker is not None:
            self.status_label.setText("Annulation en cours…")
            self.worker.cancel()

    def _on_started(self, row: int) -> None:
        self.model.set_running(row)
        self._refresh_global_progress()

    def _on_progress(self, row: int, fraction: float, message: str) -> None:
        self.model.set_progress(row, fraction, message)
        self.status_label.setText(f"{self.model.item(row).path.name} : {message}")
        self._refresh_global_progress()

    def _on_finished(self, row: int, outcome: VideoOutcome) -> None:
        self.model.set_outcome(row, outcome, self.output_dir)
        self._refresh_global_progress()
        if row == self.table.currentIndex().row():
            self._show_selected()

    def _on_cancelled(self, row: int) -> None:
        self.model.set_status(row, Status.CANCELLED)

    def _on_failed(self, message: str) -> None:
        QMessageBox.critical(self, "Erreur du traitement", message)

    def _on_batch_done(self) -> None:
        for row in self._running_rows:
            if row < self.model.rowCount() and self.model.item(row).status == Status.RUNNING:
                self.model.set_status(row, Status.CANCELLED)
        cancelled = self.worker is not None and self.worker.cancelled
        if self.batch_thread is not None:
            self.batch_thread.deleteLater()
        if self.worker is not None:
            self.worker.deleteLater()
        self.batch_thread, self.worker = None, None
        self._running_rows = set()
        self.status_label.setText("Traitement annulé" if cancelled else "Traitement terminé")
        self._refresh_global_progress()
        self._update_actions()

    def _refresh_global_progress(self) -> None:
        rows = self._running_rows or set(range(self.model.rowCount()))
        if not rows:
            self.progress.setValue(0)
            return
        total = sum(self.model.item(r).progress for r in rows if r < self.model.rowCount())
        self.progress.setValue(int(1000 * total / len(rows)))

    # --------------------------------------------------------------- résultats
    def _show_selected(self) -> None:
        row = self.table.currentIndex().row()
        if row < 0:
            return
        item = self.model.item(row)
        video_dir = item.output_dir or (self.output_dir / item.path.stem)
        self.results.set_video(video_dir if video_dir.is_dir() else None, item.path.stem,
                               item.duration_s)

    def _request_preview(self, time_s: float) -> None:
        row = self.table.currentIndex().row()
        if row < 0:
            return
        try:
            config = self.editor.config()
        except ConfigError:
            config = PipelineConfig()
        self._preview_request += 1
        task = FramePreviewTask(self._preview_request, self.model.item(row).path, time_s, config)
        task.signals.ready.connect(self._on_preview)
        task.signals.failed.connect(lambda _r, msg: logger.warning("Aperçu impossible : %s", msg))
        pool = QThreadPool.globalInstance()
        if pool is not None:
            pool.start(task)

    def _on_preview(self, request: int, image: object, index: int) -> None:
        if request == self._preview_request:  # ignore les réponses périmées
            self.results.show_frame(image, index)

    # --------------------------------------------------------------- fermeture
    def closeEvent(self, event: QCloseEvent | None) -> None:  # noqa: N802 - API Qt
        if event is None:
            return
        if self.running:
            answer = QMessageBox.question(self, "Traitement en cours",
                                          "Annuler le traitement et quitter ?")
            if answer != QMessageBox.Yes:
                event.ignore()
                return
            self.cancel()
            if self.batch_thread is not None:
                self.batch_thread.wait()
        try:
            self.settings.setValue(SETTINGS_CONFIG, self.editor.config().to_json())
        except ConfigError:
            pass
        self.settings.setValue(SETTINGS_GEOMETRY, self.saveGeometry())
        self.settings.setValue(SETTINGS_STATE, self.saveState())
        self.console.detach(logging.getLogger("panelrecon"))
        event.accept()
