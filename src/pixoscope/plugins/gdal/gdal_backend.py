"""Backend GDAL optionnel — extra ``pixoscope[gdal]``.

Jamais requis pour l'usage courant de pixoscope (voir
``REFERENCE_TECHNIQUE.md`` §2 pour la justification). Utile dans deux
cas où GDAL apporte un gain réel par rapport à ``tifffile``/``imageio`` :

- lire des overviews déjà construites dans un GeoTIFF/COG (aucun calcul
  client, juste la lecture du niveau stocké) ;
- ouvrir des formats que ``tifffile``/``imageio`` ne savent pas fenêtrer
  nativement, notamment **JPEG2000**, dont le driver GDAL sait lire des
  fenêtres et des niveaux de résolution sans décoder l'image entière.

Ce module lève ``ImportError`` à l'import si ``osgeo.gdal`` n'est pas
installé — c'est ce que :mod:`pixoscope.io.backend_registry` utilise
pour savoir si ce backend est disponible.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from osgeo import gdal, gdal_array

from pixoscope.io.backend_base import ImageBackend, ImageHandle, PyramidLevelInfo

gdal.UseExceptions()

#: Extensions pour lesquelles GDAL apporte un gain documenté par rapport
#: au socle tifffile/imageio (voir le docstring du module).
_PREFERRED_SUFFIXES = {".jp2", ".j2k", ".ecw", ".sid"}


class GdalHandle(ImageHandle):
    """``ImageHandle`` basé sur un ``gdal.Dataset``.

    Convention de niveaux : le niveau 0 est la pleine résolution (lue
    directement sur le dataset) ; le niveau ``i >= 1`` correspond à
    l'overview d'indice ``i - 1`` de GDAL (``band.GetOverview(i - 1)``).
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._ds = gdal.Open(str(path), gdal.GA_ReadOnly)
        if self._ds is None:
            raise ValueError(f"GDAL n'a pas pu ouvrir [{path}]")

        self.width = self._ds.RasterXSize
        self.height = self._ds.RasterYSize
        self.n_bands = self._ds.RasterCount
        first_band = self._ds.GetRasterBand(1)
        self.dtype = np.dtype(gdal_array.GDALTypeCodeToNumericTypeCode(first_band.DataType))
        self.band_names = None

        self.levels = [PyramidLevelInfo(0, self.width, self.height)]
        for i in range(first_band.GetOverviewCount()):
            overview = first_band.GetOverview(i)
            self.levels.append(PyramidLevelInfo(i + 1, overview.XSize, overview.YSize))

    def _band_array(self, band_index: int, level: int, x: int, y: int, w: int, h: int) -> np.ndarray:
        band = self._ds.GetRasterBand(band_index + 1)
        if level > 0:
            band = band.GetOverview(level - 1)
        return band.ReadAsArray(x, y, w, h)

    def read_region(
        self, level: int, x: int, y: int, w: int, h: int, bands: Sequence[int] | None = None
    ) -> np.ndarray:
        band_indices = list(bands) if bands is not None else list(range(self.n_bands))
        planes = [self._band_array(b, level, x, y, w, h) for b in band_indices]
        if len(planes) == 1:
            return planes[0]
        return np.stack(planes, axis=-1)

    def read_overview(self, level: int | None = None) -> np.ndarray:
        if len(self.levels) > 1:
            target = self.levels[-1].level if level is None else level
            info = self.levels[target]
            return self.read_region(target, 0, 0, info.width, info.height)

        # Pas d'overview stockée : on laisse GDAL rééchantillonner à la
        # volée en demandant un buffer de sortie réduit — évite de
        # matérialiser la pleine résolution juste pour un aperçu.
        max_side = 1024
        scale = min(1.0, max_side / max(self.width, self.height))
        buf_w, buf_h = max(1, int(self.width * scale)), max(1, int(self.height * scale))
        planes = [
            self._ds.GetRasterBand(b + 1).ReadAsArray(
                0, 0, self.width, self.height, buf_xsize=buf_w, buf_ysize=buf_h
            )
            for b in range(self.n_bands)
        ]
        return planes[0] if len(planes) == 1 else np.stack(planes, axis=-1)

    def close(self) -> None:
        self._ds = None


class GdalBackend(ImageBackend):
    """Backend GDAL — sollicité pour JPEG2000/ECW/MrSID ou sur demande explicite.

    Pour les formats déjà couverts par le socle (TIFF, PNG, JPEG),
    :func:`pixoscope.io.backend_registry.open_image` essaie ``tifffile``
    et ``imageio`` en priorité ; ce backend n'est atteint que pour les
    formats qu'ils ne gèrent pas, ou si ``PIXOSCOPE_BACKEND=gdal`` force
    son usage.
    """

    name = "gdal"

    def can_open(self, path: Path) -> bool:
        if path.suffix.lower() in _PREFERRED_SUFFIXES:
            return True
        # Repli : GDAL sait ouvrir énormément de formats — utile comme
        # dernier recours pour un fichier qu'aucun autre backend ne
        # revendique, plutôt que d'échouer complètement.
        return gdal.IdentifyDriver(str(path)) is not None

    def open(self, path: Path) -> ImageHandle:
        return GdalHandle(path)
