"""Petits utilitaires Qt partagés entre les widgets de ``pixoscope.ui``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import QThreadPool
from PySide6.QtGui import QImage

_ICONS_DIR = Path(__file__).parent / "icons"


def array_to_qimage(array: np.ndarray) -> QImage:
    """Convertit un tableau ``uint8`` ``(h, w)``/``(h, w, 3)`` en ``QImage``.

    Parameters
    ----------
    array : numpy.ndarray

    Returns
    -------
    PySide6.QtGui.QImage
        Copie interne des données (indépendante du tableau numpy source,
        qui peut être libéré après l'appel).
    """
    array = np.ascontiguousarray(array)
    h, w = array.shape[:2]
    if array.ndim == 2:
        image = QImage(array.data, w, h, w, QImage.Format.Format_Grayscale8)
    else:
        image = QImage(array.data, w, h, w * 3, QImage.Format.Format_RGB888)
    return image.copy()


def icon_path(name: str) -> str:
    """Chemin absolu d'une icône embarquée dans ``pixoscope.ui.icons``.

    Parameters
    ----------
    name : str
        Nom de fichier (ex. ``"home.svg"``).

    Returns
    -------
    str
    """
    return str(_ICONS_DIR / name)

#: Nombre maximal de tâches de fond simultanées (lecture de tuile, calcul
#: de statistiques, algorithme de rehaussement...). Laisse toujours au
#: moins un coeur disponible pour le thread UI.
_MAX_THREADS = max(2, (QThreadPool.globalInstance().maxThreadCount() or 4) - 1)

_pool_configured = False


def global_thread_pool() -> QThreadPool:
    """Retourne le pool de threads global, configuré au premier appel.

    Returns
    -------
    PySide6.QtCore.QThreadPool
    """
    global _pool_configured
    pool = QThreadPool.globalInstance()
    if not _pool_configured:
        pool.setMaxThreadCount(_MAX_THREADS)
        _pool_configured = True
    return pool
