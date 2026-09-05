"""Construction et cache disque d'une pyramide pour un fichier non pyramidal.

Quand un ``ImageHandle`` n'expose qu'un seul niveau (fichier PNG/JPEG
décodé intégralement, ou TIFF simple résolution), chaque zoom arrière
recalculerait sans cela une décimation à la volée sur l'image pleine
résolution. Ce module construit une fois pour toutes des niveaux
basse résolution supplémentaires, mis en cache sur disque
(``platformdirs.user_cache_dir("pixoscope")``), et les expose via
:class:`PyramidCacheHandle` qui délègue le niveau 0 au handle d'origine
(aucune duplication du plein résolution) et sert les niveaux inférieurs
depuis le cache.

Le cache est nommé d'après un hash de ``(chemin absolu, mtime, taille)``
du fichier source : toute modification du fichier source change
automatiquement le chemin de cache, sans étape d'invalidation explicite.

Note de conception
-------------------
Le niveau 0 est construit par lecture en bandes horizontales
(``_STRIPE_ROWS`` lignes à la fois) plutôt qu'un chargement complet en
mémoire, pour borner le pic mémoire lors de la construction. Les niveaux
suivants sont dérivés du niveau juste au-dessus, en mémoire (déjà 4x plus
petit à chaque niveau, donc rapidement de taille raisonnable).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import platformdirs
import tifffile
from loguru import logger

from pixoscope.io.backend_base import ImageHandle, PyramidLevelInfo

_STRIPE_ROWS = 2048
#: En dessous de cette taille (plus grand côté), on ne construit pas de
#: pyramide supplémentaire : l'image tient déjà confortablement en mémoire.
_MIN_SIZE_FOR_PYRAMID = 2048
#: On arrête de générer des niveaux quand le plus grand côté descend
#: sous ce seuil.
_MIN_LEVEL_SIZE = 1024

_MANIFEST_NAME = "manifest.json"


def cache_dir_for(source: Path) -> Path:
    """Retourne le répertoire de cache dédié à ``source``.

    Le nom encode ``(chemin absolu, mtime, taille)`` : toute
    modification du fichier source pointe automatiquement vers un
    répertoire différent, ce qui sert d'invalidation implicite.

    Parameters
    ----------
    source : pathlib.Path

    Returns
    -------
    pathlib.Path
    """
    stat = source.stat()
    key = f"{source.resolve()}::{stat.st_mtime_ns}::{stat.st_size}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return Path(platformdirs.user_cache_dir("pixoscope")) / "pyramids" / digest


def _downsample2x(array: np.ndarray) -> np.ndarray:
    """Réduit un tableau ``(h, w, c)`` d'un facteur 2 par moyenne de blocs.

    Parameters
    ----------
    array : numpy.ndarray
        Tableau 3D ``(h, w, c)``. Les dimensions impaires sont tronquées
        d'une ligne/colonne avant réduction.

    Returns
    -------
    numpy.ndarray
        Tableau ``(h // 2, w // 2, c)``, dans le même dtype (arrondi
        pour les types entiers).
    """
    h, w, c = array.shape
    h2, w2 = h - (h % 2), w - (w % 2)
    trimmed = array[:h2, :w2, :]
    reshaped = trimmed.reshape(h2 // 2, 2, w2 // 2, 2, c).astype(np.float32)
    reduced = reshaped.mean(axis=(1, 3))
    if np.issubdtype(array.dtype, np.integer):
        reduced = np.round(reduced)
    return reduced.astype(array.dtype)


def _as_hwc(region: np.ndarray) -> np.ndarray:
    """Normalise une tuile ``(h, w)`` ou ``(h, w, c)`` vers ``(h, w, c)``."""
    return region[:, :, np.newaxis] if region.ndim == 2 else region


def needs_pyramid_cache(handle: ImageHandle) -> bool:
    """Indique si ``handle`` bénéficierait d'une pyramide mise en cache.

    Parameters
    ----------
    handle : ImageHandle

    Returns
    -------
    bool
        ``True`` si le handle n'expose qu'un seul niveau et que l'image
        dépasse :data:`_MIN_SIZE_FOR_PYRAMID`.
    """
    return len(handle.levels) == 1 and max(handle.width, handle.height) > _MIN_SIZE_FOR_PYRAMID


def build_pyramid_cache(handle: ImageHandle, source: Path) -> Path:
    """Construit (si absent) le cache de niveaux basse résolution.

    Parameters
    ----------
    handle : ImageHandle
        Handle du fichier source, positionné sur son unique niveau 0.
    source : pathlib.Path
        Chemin du fichier source (sert à nommer le cache).

    Returns
    -------
    pathlib.Path
        Répertoire de cache, contenant ``manifest.json`` et un fichier
        TIFF par niveau supplémentaire (``level_1.tif``, ``level_2.tif``, ...).
    """
    cache_dir = cache_dir_for(source)
    manifest_path = cache_dir / _MANIFEST_NAME
    if manifest_path.exists():
        return cache_dir

    logger.info(f"Construction du cache pyramide pour [{source}] dans [{cache_dir}]")
    cache_dir.mkdir(parents=True, exist_ok=True)

    n_bands = handle.n_bands
    prev_reader: Callable[[int, int, int, int], np.ndarray]
    prev_reader = lambda x, y, w, h: _as_hwc(handle.read_region(0, x, y, w, h))  # noqa: E731
    prev_w, prev_h = handle.width, handle.height

    levels_meta: list[dict[str, object]] = []
    level_index = 0
    while max(prev_w, prev_h) > _MIN_LEVEL_SIZE:
        level_index += 1
        new_w, new_h = prev_w // 2, prev_h // 2
        if new_w < 1 or new_h < 1:
            break

        level_path = cache_dir / f"level_{level_index}.tif"
        memmap = tifffile.memmap(
            str(level_path),
            shape=(new_h, new_w, n_bands),
            dtype=handle.dtype,
            photometric="minisblack",
        )
        # Bandes lues par tranches horizontales pour borner le pic mémoire :
        # 2 lignes source produisent 1 ligne du niveau réduit.
        stripe = _STRIPE_ROWS - (_STRIPE_ROWS % 2)
        for y0 in range(0, new_h * 2, stripe):
            src_h = min(stripe, prev_h - y0)
            if src_h <= 0:
                break
            chunk = prev_reader(0, y0, prev_w, src_h)
            reduced = _downsample2x(chunk)
            out_y0 = y0 // 2
            memmap[out_y0 : out_y0 + reduced.shape[0], :, :] = reduced
        memmap.flush()

        levels_meta.append({"index": level_index, "width": new_w, "height": new_h, "path": level_path.name})

        # Le niveau suivant part de celui qu'on vient d'écrire (relu depuis
        # le memmap, déjà nettement plus petit).
        # mypy ne parvient pas à inférer le type de ce lambda à cause du
        # paramètre par défaut `_m=memmap` (idiome de capture précoce en
        # boucle) ; le comportement est correct (voir prev_reader ci-dessus).
        prev_reader = lambda x, y, w, h, _m=memmap: _m[y : y + h, x : x + w, :]  # type: ignore[misc]  # noqa: E731
        prev_w, prev_h = new_w, new_h

    manifest = {
        "source": str(source.resolve()),
        "n_bands": n_bands,
        "dtype": str(handle.dtype),
        "levels": levels_meta,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info(f"Cache pyramide construit : {len(levels_meta)} niveau(x) supplémentaire(s)")
    return cache_dir


class PyramidCacheHandle(ImageHandle):
    """``ImageHandle`` combinant un handle source et des niveaux en cache.

    Le niveau 0 est toujours servi par le handle d'origine (pas de
    duplication de la pleine résolution) ; les niveaux suivants sont lus
    depuis les fichiers TIFF de cache, en lecture seule via
    ``tifffile.memmap``.

    Parameters
    ----------
    inner : ImageHandle
        Handle du fichier source (niveau 0).
    cache_dir : pathlib.Path
        Répertoire produit par :func:`build_pyramid_cache`.
    """

    def __init__(self, inner: ImageHandle, cache_dir: Path) -> None:
        self._inner = inner
        self.path = inner.path
        self.height, self.width = inner.height, inner.width
        self.n_bands = inner.n_bands
        self.dtype = inner.dtype
        self.band_names = inner.band_names

        manifest = json.loads((cache_dir / _MANIFEST_NAME).read_text())
        self._cached_arrays: dict[int, np.ndarray] = {}
        self.levels = [PyramidLevelInfo(0, self.width, self.height)]
        for level in manifest["levels"]:
            idx = int(level["index"])
            level_path = cache_dir / str(level["path"])
            self._cached_arrays[idx] = tifffile.memmap(str(level_path), mode="r")
            self.levels.append(PyramidLevelInfo(idx, int(level["width"]), int(level["height"])))

    def read_region(
        self, level: int, x: int, y: int, w: int, h: int, bands: Sequence[int] | None = None
    ) -> np.ndarray:
        if level == 0:
            return self._inner.read_region(0, x, y, w, h, bands=bands)
        arr = _as_hwc(self._cached_arrays[level])
        region = arr[y : y + h, x : x + w, :]
        if bands is not None:
            region = region[..., bands]
        return region[..., 0] if region.shape[-1] == 1 else np.asarray(region)

    def read_overview(self, level: int | None = None) -> np.ndarray:
        target = self.levels[-1].level if level is None else level
        w, h = self.levels[target].width, self.levels[target].height
        return self.read_region(target, 0, 0, w, h)

    def close(self) -> None:
        self._inner.close()
