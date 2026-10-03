"""Point d'entrée de l'application empaquetée (PyInstaller).

* sans argument (ou avec des chemins de vidéos) : interface graphique ;
* ``--cli ...`` : traitement en ligne de commande (mêmes options que
  ``python -m panelrecon.cli``), utile pour un lot sans interface.

``freeze_support()`` doit être appelé avant tout : les processus de calcul
(« spawn ») relancent ce même exécutable.
"""

import multiprocessing
import sys

if __name__ == "__main__":
    multiprocessing.freeze_support()
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        from panelrecon.cli import main as cli_main

        sys.exit(cli_main(sys.argv[2:]))
    from panelrecon.gui.app import main as gui_main

    sys.exit(gui_main())
