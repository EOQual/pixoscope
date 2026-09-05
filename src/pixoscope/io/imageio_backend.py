"""Backend générique via ``imageio`` — PNG/JPEG/BMP/GIF et repli universel.

Limite assumée et documentée (voir ``REFERENCE_TECHNIQUE.md``) : ces
formats ne supportent pas de lecture fenêtrée native. La première lecture
décode donc l'image entière, mise en cache en mémoire dans le handle ;
les lectures suivantes ne font que découper ce tableau déjà décodé. Pour
les images de cette catégorie dépassant une taille conséquente,
``pixoscope.io.pyramid_builder`` construit une pyramide en cache disque
dès l'ouverture, pour que ce coût de décodage complet ne soit payé
qu'une seule fois.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import imageio.v3 as iio
import numpy as np

from pixoscope.io.backend_base import ImageBackend, ImageHandle, PyramidLevelInfo

#: Extensions couvertes par ce backend. Sert de filtre rapide dans
#: :meth:`ImageIOBackend.can_open` — imageio lui-même sait ouvrir bien
#: plus de formats, mais on ne revendique ici que ceux qui n'ont pas de
#: backend plus performant dédié (TIFF -> tifffile, .lum -> plugin).
_SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".ppm", ".pgm"}


def _downsample_stride(array: np.ndarray, factor: int) -> np.ndarray:
    """Sous-échantillonne un tableau par un pas fixe (décimation simple).

    Utilisé uniquement pour produire un aperçu basse résolution
    (statistiques, vignette) sans dépendance de rééchantillonnage tierce
    — pas destiné à un rendu final de qualité photo.

    Parameters
    ----------
    array : numpy.ndarray
        Tableau ``(h, w)`` ou ``(h, w, n)``.
    factor : int
        Pas de décimation (>= 1).

    Returns
    -------
    numpy.ndarray
    """
    if factor <= 1:
        return array
    return array[::factor, ::factor, ...]


class ImageIOHandle(ImageHandle):
    """``ImageHandle`` pour un fichier décodé intégralement par ``imageio``."""

    def __init__(self, path: Path) -> None:
        self.path = path
        array = iio.imread(path)
        if array.ndim == 2:
            array = array[:, :, np.newaxis]
        self._array = array  # (H, W, bands), décodé une seule fois
        self.height, self.width, self.n_bands = array.shape
        self.dtype = array.dtype
        self.band_names = None

        # Niveau 0 uniquement, plus un ou deux niveaux réduits pour les
        # statistiques/vignette sans repasser par un décodage.
        self.levels: list[PyramidLevelInfo] = [PyramidLevelInfo(0, self.width, self.height)]
        factor = 1
        level_index = 0
        while min(self.width // factor, self.height // factor) > 1024:
            factor *= 4
            level_index += 1
            self.levels.append(
                PyramidLevelInfo(level_index, max(1, self.width // factor), max(1, self.height // factor))
            )
        self._level_factors = {info.level: 4**info.level for info in self.levels}

    def read_region(
        self, level: int, x: int, y: int, w: int, h: int, bands: Sequence[int] | None = None
    ) -> np.ndarray:
        factor = self._level_factors[level]
        source = self._array if factor == 1 else _downsample_stride(self._array, factor)
        region = source[y : y + h, x : x + w, :]
        if bands is not None:
            region = region[..., bands]
        return region[..., 0] if region.shape[-1] == 1 else region

    def read_overview(self, level: int | None = None) -> np.ndarray:
        target = self.levels[-1].level if level is None else level
        factor = self._level_factors[target]
        source = _downsample_stride(self._array, factor)
        return source[..., 0] if source.shape[-1] == 1 else source

    def close(self) -> None:
        # Rien à libérer explicitement : le tableau décodé est laissé au
        # ramasse-miettes avec le handle.
        self._array = None


class ImageIOBackend(ImageBackend):
    """Backend de repli générique, basé sur ``imageio``."""

    name = "imageio"

    def can_open(self, path: Path) -> bool:
        return path.suffix.lower() in _SUPPORTED_SUFFIXES

    def open(self, path: Path) -> ImageHandle:
        return ImageIOHandle(path)
