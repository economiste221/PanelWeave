"""Point d'entrée de l'application graphique.

``main()`` doit s'exécuter dans le thread principal. ``freeze_support()`` est
appelé en premier : dans l'application empaquetée (.app), les processus de
calcul (méthode « spawn ») relancent le même exécutable.
"""

from __future__ import annotations

import logging
import multiprocessing
import sys
from collections.abc import Sequence
from logging.handlers import RotatingFileHandler
from pathlib import Path

APP_NAME = "PanelRecon"
ORG_NAME = "PanelWeave"
LOG_FILENAME = "panelrecon-gui.log"


def _configure_logging(log_dir: Path) -> None:
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s [%(threadName)s] %(name)s: %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    root.addHandler(stream)
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(log_dir / LOG_FILENAME, maxBytes=5_000_000, backupCount=3,
                                      encoding="utf-8")
        handler.setFormatter(formatter)
        root.addHandler(handler)
    except OSError as exc:
        root.warning("Journal sur disque indisponible (%s)", exc)


def main(argv: Sequence[str] | None = None) -> int:
    multiprocessing.freeze_support()
    from PyQt5.QtCore import QCoreApplication, QStandardPaths, Qt
    from PyQt5.QtWidgets import QApplication

    from panelrecon.gui.main_window import MainWindow
    from panelrecon.gui.style import DARK_QSS

    # Attributs à fixer avant la création de l'application (écrans Retina).
    QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
    QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)
    QCoreApplication.setOrganizationName(ORG_NAME)
    QCoreApplication.setApplicationName(APP_NAME)
    app = QApplication(list(argv) if argv is not None else sys.argv)
    app.setStyleSheet(DARK_QSS)
    log_dir = Path(QStandardPaths.writableLocation(QStandardPaths.AppDataLocation) or Path.home())
    _configure_logging(log_dir)
    window = MainWindow()
    window.show()
    extra = [Path(a) for a in (argv if argv is not None else sys.argv)[1:] if Path(a).exists()]
    if extra:
        window.add_paths(extra)
    return int(app.exec_())


if __name__ == "__main__":
    sys.exit(main())
