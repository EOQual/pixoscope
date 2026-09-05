"""Backend ``ImageBackend``/``ImageHandle`` pour le format ``.lum``.

Expose un seul niveau natif (pas de pyramide dans le format lui-même) —
:mod:`pixoscope.io.backend_registry` construit automatiquement un cache
de pyramide par-dessus (voir :mod:`pixoscope.io.pyramid_builder`) si
l'image dépasse le seuil de taille, exactement comme pour un PNG/JPEG.

Fenêtrage : seule la plage de lignes demandée est lue depuis le disque
(``LumReader.read_rows``) ; le découpage en colonnes se fait ensuite en
mémoire sur cette plage déjà réduite — moins coûteux qu'une lecture
complète, mais pas un vrai fenêtrage 2D natif (limite du format, voir
``REFERENCE_TECHNIQUE.md``).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from pixoscope.io.backend_base import ImageBackend, ImageHandle, PyramidLevelInfo
from pixoscope.plugins.lum.lum_object import LumReader


class LumHandle(ImageHandle):
    """``ImageHandle`` pour un fichier ``.lum`` (toujours mono-bande)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._reader = LumReader(path)
        self.height = self._reader.n_rows
        self.width = self._reader.n_cols
        self.n_bands = 1
        self.dtype = self._reader.dtype
        self.band_names = None
        self.levels = [PyramidLevelInfo(0, self.width, self.height)]

    def read_region(
        self, level: int, x: int, y: int, w: int, h: int, bands: Sequence[int] | None = None
    ) -> np.ndarray:
        if level != 0:
            raise ValueError("LumHandle n'expose que le niveau 0 (voir PyramidCacheHandle pour le reste)")
        rows = self._reader.read_rows(y, h)
        return rows[:, x : x + w]

    def read_overview(self, level: int | None = None) -> np.ndarray:
        return self._reader.read_all()

    def close(self) -> None:
        pass  # LumReader ouvre/ferme le fichier à chaque lecture, rien à garder ouvert.


class LumBackend(ImageBackend):
    """Backend pour le format ``.lum`` (extra ``pixoscope[lum]``)."""

    name = "lum"

    def can_open(self, path: Path) -> bool:
        return path.suffix.lower() == ".lum"

    def open(self, path: Path) -> ImageHandle:
        return LumHandle(path)
