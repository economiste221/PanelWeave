"""Thème sombre (QSS) et couleurs des verdicts."""

from __future__ import annotations

from panelrecon.core.models import Verdict

VERDICT_COLORS: dict[str, str] = {
    Verdict.OK.value: "#4caf50",
    Verdict.TO_REVIEW.value: "#ffb300",
    Verdict.FAILED.value: "#e53935",
}

DARK_QSS = """
QWidget { background-color: #1e1f22; color: #dcdcdc; font-size: 13px; }
QMainWindow::separator { background: #2b2d31; width: 4px; height: 4px; }
QToolBar { background: #2b2d31; border: none; spacing: 6px; padding: 4px; }
QToolButton, QPushButton {
    background: #3a3d43; border: 1px solid #4a4d55; border-radius: 4px; padding: 5px 10px;
}
QToolButton:hover, QPushButton:hover { background: #474b52; }
QToolButton:pressed, QPushButton:pressed { background: #2f6fd0; }
QToolButton:disabled, QPushButton:disabled { color: #777; background: #2b2d31; }
QTableView, QListWidget, QPlainTextEdit, QTreeView {
    background: #17181a; alternate-background-color: #1c1d20; border: 1px solid #333;
    selection-background-color: #2f6fd0; selection-color: white;
}
QHeaderView::section { background: #2b2d31; color: #c8c8c8; border: none; padding: 4px; }
QProgressBar { border: 1px solid #444; border-radius: 4px; text-align: center; background: #17181a; }
QProgressBar::chunk { background-color: #2f6fd0; border-radius: 3px; }
QDockWidget::title { background: #2b2d31; padding: 4px; }
QTabWidget::pane { border: 1px solid #333; }
QTabBar::tab { background: #2b2d31; padding: 5px 10px; border: 1px solid #333; }
QTabBar::tab:selected { background: #3a3d43; }
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background: #17181a; border: 1px solid #444; border-radius: 3px; padding: 2px 4px;
}
QGraphicsView { border: 1px solid #333; }
QStatusBar { background: #2b2d31; }
"""
