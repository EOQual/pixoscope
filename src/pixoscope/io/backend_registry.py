"""Sélection automatique du backend de lecture le plus adapté à un fichier.

Ordre de priorité par défaut : plugin ``.lum`` (si installé, extra
``pixoscope[lum]``) > ``tifffile`` (TIFF/BigTIFF, fenêtrage + pyramide
natifs) > ``imageio`` (repli générique PNG/JPEG/...) > plugin GDAL (si
installé, extra ``pixoscope[gdal]`` — utile pour JPEG2000 ou les COG déjà
pyramidés). GDAL n'est jamais requis : voir ``REFERENCE_TECHNIQUE.md``.

La variable d'environnement ``PIXOSCOPE_BACKEND`` force un backend par
son ``name`` (ex. ``PIXOSCOPE_BACKEND=gdal``), utile pour comparer les
backends sur un même fichier (voir ``scripts/bench_open.py``).
"""

from __future__ import annotations

import os
from pathlib import Path

from loguru import logger

from pixoscope.io import pyramid_builder
from pixoscope.io.backend_base import ImageBackend, ImageHandle
from pixoscope.io.imageio_backend import ImageIOBackend
from pixoscope.io.tifffile_backend import TiffFileBackend


def _default_backends() -> list[ImageBackend]:
    """Construit la liste des backends disponibles, plugins optionnels inclus.

    Returns
    -------
    list of ImageBackend
        Ordonnée par priorité d'essai.
    """
    backends: list[ImageBackend] = []

    try:
        from pixoscope.plugins.lum.lum_backend import LumBackend

        backends.append(LumBackend())
    except ImportError:
        logger.trace("Plugin .lum non disponible (extra pixoscope[lum] non installé)")

    backends.append(TiffFileBackend())
    backends.append(ImageIOBackend())

    try:
        from pixoscope.plugins.gdal.gdal_backend import GdalBackend

        backends.append(GdalBackend())
    except ImportError:
        logger.trace("Backend GDAL non disponible (extra pixoscope[gdal] non installé)")

    return backends


def open_image(path: str | Path, *, auto_pyramid_cache: bool = True) -> ImageHandle:
    """Ouvre une image en sélectionnant automatiquement le backend adapté.

    Parameters
    ----------
    path : str or pathlib.Path
        Chemin du fichier à ouvrir.
    auto_pyramid_cache : bool, optional
        Si ``True`` (par défaut), construit automatiquement un cache de
        pyramide (voir :mod:`pixoscope.io.pyramid_builder`) pour les
        images dépassant le seuil de taille et n'exposant qu'un seul
        niveau natif.

    Returns
    -------
    ImageHandle

    Raises
    ------
    FileNotFoundError
        Si ``path`` n'existe pas.
    ValueError
        Si aucun backend disponible ne sait ouvrir ce fichier.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    forced_name = os.environ.get("PIXOSCOPE_BACKEND")
    backends = _default_backends()
    if forced_name:
        forced = [b for b in backends if b.name == forced_name]
        if not forced:
            raise ValueError(
                f"PIXOSCOPE_BACKEND={forced_name!r} demandé mais indisponible "
                f"(backends installés : {[b.name for b in backends]})"
            )
        backends = forced

    for backend in backends:
        if backend.can_open(path):
            logger.debug(f"Ouverture de [{path}] avec le backend [{backend.name}]")
            handle = backend.open(path)
            if auto_pyramid_cache and pyramid_builder.needs_pyramid_cache(handle):
                cache_dir = pyramid_builder.build_pyramid_cache(handle, path)
                handle = pyramid_builder.PyramidCacheHandle(handle, cache_dir)
            return handle

    raise ValueError(
        f"Aucun backend ne sait ouvrir [{path}] (extension {path.suffix!r}). "
        "Formats gérés par défaut : .tif/.tiff, .png/.jpg/.jpeg/.bmp/.gif/.webp. "
        "Voir les extras pixoscope[lum] et pixoscope[gdal] pour d'autres formats."
    )
