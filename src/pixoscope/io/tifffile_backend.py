"""Backend TIFF/BigTIFF via ``tifffile`` + ``zarr`` — lecture fenêtrée.

C'est le backend central de pixoscope pour les grosses images : au lieu
d'un ``ReadAsArray()`` complet (qui charge l'image entière quelle que
soit la taille de la fenêtre affichée), on ouvre le fichier via
``tifffile.imread(path, aszarr=True)``, qui expose un store zarr
paresseux — une fenêtre lue via
``read_region`` ne décode que les tuiles/bandes du fichier réellement
recouvertes par la requête.

Pyramide : si le fichier est un TIFF pyramidal (SubIFDs, OME-TIFF
multi-résolution...), le store zarr expose un groupe avec un tableau par
niveau ; sinon un seul niveau (pleine résolution) est disponible et
``pixoscope.io.pyramid_builder`` prend le relais pour construire un cache.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import tifffile
import zarr

from pixoscope.io.backend_base import ImageBackend, ImageHandle, PyramidLevelInfo

_SUPPORTED_SUFFIXES = {".tif", ".tiff"}

#: Taille maximale plausible pour un axe "bandes" (au-delà, c'est presque
#: certainement un axe spatial). Voir _infer_band_axis.
_MAX_PLAUSIBLE_BAND_COUNT = 32


def _infer_band_axis(shape: tuple[int, ...]) -> int | None:
    """Devine l'indice de l'axe "bande" d'un tableau 3D à partir de sa forme.

    Parameters
    ----------
    shape : tuple of int
        Forme du tableau zarr exposé par tifffile.

    Returns
    -------
    int or None
        ``None`` si le tableau est 2D (pas d'axe bande) ou si aucun axe
        ne ressemble à un axe de bandes.

    Notes
    -----
    On se base sur la taille des axes plutôt que sur la chaîne
    ``TiffPageSeries.axes`` : cette dernière ne correspond pas toujours,
    en pratique, à l'ordre réel des axes du tableau retourné par
    ``tifffile.imread(..., aszarr=True)`` (observé notamment sur des
    fichiers écrits sans tag ``ExtraSamples`` explicite). L'heuristique
    retenue — le plus petit des 3 axes, s'il est nettement plus petit
    que les deux autres et reste sous :data:`_MAX_PLAUSIBLE_BAND_COUNT`
    — est robuste aux dispositions "bandes en premier" (``C, H, W``) et
    "bandes en dernier" (``H, W, C``), les deux cas courants.

    Limite connue : un TIFF multi-dimensionnel avec un axe non-bande de
    petite taille (pile Z peu profonde, série temporelle courte) serait
    mal interprété comme un axe de bandes en v1.
    """
    if len(shape) != 3:
        return None
    axis = int(np.argmin(shape))
    smallest = shape[axis]
    others = [n for i, n in enumerate(shape) if i != axis]
    if smallest <= _MAX_PLAUSIBLE_BAND_COUNT and smallest < min(others):
        return axis
    return None


class TiffFileHandle(ImageHandle):
    """``ImageHandle`` pour un fichier TIFF/BigTIFF via tifffile+zarr."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._tiff = tifffile.TiffFile(str(path))

        store = tifffile.imread(str(path), aszarr=True, series=0)
        opened = zarr.open(store, mode="r")
        self._store = store

        if isinstance(opened, zarr.Group):
            # Groupe multi-résolution : les clés sont conventionnellement
            # "0", "1", ... du plus résolu au moins résolu.
            level_keys = sorted(opened.array_keys(), key=lambda k: int(k))
            self._level_arrays: list[zarr.Array] = [opened[k] for k in level_keys]
        else:
            self._level_arrays = [opened]

        first = self._level_arrays[0]
        self._band_axis = _infer_band_axis(first.shape)
        self.height, self.width = self._spatial_shape(first.shape)
        self.dtype = first.dtype
        self.n_bands = first.shape[self._band_axis] if self._band_axis is not None else 1
        self.band_names = None

        self.levels: list[PyramidLevelInfo] = []
        for i, arr in enumerate(self._level_arrays):
            h, w = self._spatial_shape(arr.shape)
            self.levels.append(PyramidLevelInfo(i, w, h))

    def _spatial_shape(self, shape: tuple[int, ...]) -> tuple[int, int]:
        """Extrait ``(height, width)`` d'une forme de tableau tifffile."""
        if len(shape) == 2:
            return shape[0], shape[1]
        spatial = [n for axis, n in enumerate(shape) if axis != self._band_axis]
        return spatial[0], spatial[1]

    def _slice_region(self, arr: zarr.Array, x: int, y: int, w: int, h: int, bands: Sequence[int] | None) -> np.ndarray:
        if self._band_axis is None:
            region = arr[y : y + h, x : x + w]
            return region
        if self._band_axis == arr.ndim - 1:
            region = arr[y : y + h, x : x + w, :]
            return region if bands is None else region[..., bands]
        # Axe bande en tête (ex. "SYX" -> lecture puis transposition en HWC
        # pour rester homogène avec le reste de pixoscope).
        region = arr[:, y : y + h, x : x + w]
        region = np.moveaxis(region, 0, -1)
        return region if bands is None else region[..., bands]

    def read_region(
        self, level: int, x: int, y: int, w: int, h: int, bands: Sequence[int] | None = None
    ) -> np.ndarray:
        arr = self._level_arrays[level]
        level_h, level_w = self._spatial_shape(arr.shape)
        w = min(w, level_w - x)
        h = min(h, level_h - y)
        if w <= 0 or h <= 0:
            n = self.n_bands if bands is None else len(bands)
            shape = (max(h, 0), max(w, 0)) if self._band_axis is None else (max(h, 0), max(w, 0), n)
            return np.zeros(shape, dtype=self.dtype)
        region = self._slice_region(arr, x, y, w, h, bands)
        return np.asarray(region)

    def read_overview(self, level: int | None = None) -> np.ndarray:
        target = self.levels[-1].level if level is None else level
        arr = self._level_arrays[target]
        h, w = self._spatial_shape(arr.shape)
        return self.read_region(target, 0, 0, w, h)

    def close(self) -> None:
        self._tiff.close()


class TiffFileBackend(ImageBackend):
    """Backend TIFF/BigTIFF via tifffile — fenêtrage et pyramide natifs."""

    name = "tifffile"

    def can_open(self, path: Path) -> bool:
        return path.suffix.lower() in _SUPPORTED_SUFFIXES

    def open(self, path: Path) -> ImageHandle:
        return TiffFileHandle(path)
