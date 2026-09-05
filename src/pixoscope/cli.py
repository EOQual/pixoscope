"""Point d'entrée en ligne de commande : ``pixoscope [fichier1] [fichier2]``."""

from __future__ import annotations

import sys

from loguru import logger


def main() -> None:
    """Lance l'application Pixoscope.

    Sans argument, ouvre une fenêtre vide (utiliser "Ouvrir…"). Avec un
    chemin de fichier, l'ouvre directement. Avec deux chemins, ouvre le
    mode comparaison avec une image par panneau.
    """
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    from PySide6.QtWidgets import QApplication

    from pixoscope.ui.main_window import MainWindow

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()

    paths = [arg for arg in sys.argv[1:] if not arg.startswith("-")]
    window.open_paths(paths)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
