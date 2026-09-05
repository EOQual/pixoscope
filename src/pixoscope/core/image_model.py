"""Modèle de données d'une image ouverte : bandes, mapping RVB, rendu.

``ImageDataset`` est la seule chose que ``pixoscope.ui`` manipule pour
afficher une image — jamais un ``ImageHandle`` de
:mod:`pixoscope.io.backend_base` directement. Il ajoute par-dessus :

- l'assignation de bandes arbitraires aux plans R/V/B (ou niveau de
  gris) affichés, indépendante des bandes physiques du fichier ;
- l'étirement de contraste par bande (``display_range``), appliqué au
  moment du rendu d'une tuile, jamais sur l'image entière ;
- un cache de statistiques par bande, alimenté de façon asynchrone par
  l'IHM (voir ``pixoscope.ui.load_worker``) sans jamais bloquer l'accès
  aux pixels eux-mêmes.

Aucune dépendance Qt dans ce module.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pixoscope.core.stats import BandStats, auto_stretch_range
from pixoscope.io.backend_base import ImageBackend, ImageHandle, PyramidLevelInfo

#: Bornes d'affichage par défaut selon le type natif des pixels, utilisées
#: tant qu'aucune statistique n'a encore été calculée pour la bande.
_DEFAULT_RANGE_BY_DTYPE: dict[str, tuple[float, float]] = {
    "uint8": (0.0, 255.0),
    "uint16": (0.0, 65535.0),
    "int16": (-32768.0, 32767.0),
}


@dataclass(frozen=True)
class BandInfo:
    """Métadonnées d'une bande physique de l'image.

    Attributes
    ----------
    index : int
        Indice de la bande (0-based) dans le fichier source.
    name : str
        Nom lisible (ex. ``"Bande 3"`` ou un nom fourni par le fichier).
    dtype : numpy.dtype
        Type natif des pixels de cette bande.
    """

    index: int
    name: str
    dtype: np.dtype


@dataclass(frozen=True)
class ChannelMapping:
    """Assignation des bandes physiques aux plans d'affichage.

    Deux modes mutuellement exclusifs :

    - Niveaux de gris : seul ``gray`` est renseigné.
    - Couleur : ``red``/``green``/``blue`` sont renseignés (chacun peut
      pointer vers la même bande physique, ou être ``None`` pour
      afficher un plan éteint).

    Attributes
    ----------
    red, green, blue : int or None
        Indices de bande assignés aux plans rouge/vert/bleu.
    gray : int or None
        Indice de bande assigné au mode niveaux de gris.
    """

    red: int | None = None
    green: int | None = None
    blue: int | None = None
    gray: int | None = None

    @property
    def is_grayscale(self) -> bool:
        """``True`` si le mapping courant est en niveaux de gris."""
        return self.gray is not None

    def band_indices(self) -> list[int]:
        """Retourne les indices de bandes physiques réellement utilisés.

        Sert à ne lire depuis le disque que les bandes nécessaires au
        rendu courant (voir :meth:`ImageDataset.read_display_tile`).

        Returns
        -------
        list of int
        """
        if self.is_grayscale:
            return [self.gray] if self.gray is not None else []
        return sorted({b for b in (self.red, self.green, self.blue) if b is not None})


def default_channel_mapping(n_bands: int) -> ChannelMapping:
    """Construit le mapping par défaut à l'ouverture d'une image.

    Parameters
    ----------
    n_bands : int
        Nombre de bandes physiques de l'image.

    Returns
    -------
    ChannelMapping
        Niveaux de gris sur la bande 0 si l'image est mono-bande ;
        R=0, V=1, B=2 si l'image a au moins 3 bandes ; niveaux de gris
        sur la bande 0 pour le cas dégénéré à 2 bandes (pas de
        convention RVB naturelle).
    """
    if n_bands >= 3:
        return ChannelMapping(red=0, green=1, blue=2)
    return ChannelMapping(gray=0)


def _stretch_to_uint8(band: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Étire une bande vers ``uint8`` selon ``[vmin, vmax]``.

    Le calcul intermédiaire se fait en ``float32`` mais uniquement sur
    la tuile passée en argument — jamais sur l'image entière.

    Parameters
    ----------
    band : numpy.ndarray
        Tuile 2D d'une seule bande, dans son dtype natif.
    vmin, vmax : float
        Bornes de l'étirement linéaire.

    Returns
    -------
    numpy.ndarray
        Tuile ``uint8`` de même forme que ``band``.
    """
    if vmax <= vmin:
        vmax = vmin + 1.0
    scaled = (band.astype(np.float32) - vmin) * (255.0 / (vmax - vmin))
    return np.clip(scaled, 0.0, 255.0).astype(np.uint8)


class ImageDataset:
    """Image ouverte, avec mapping de bandes et étirement d'affichage.

    Parameters
    ----------
    handle : ImageHandle
        Handle de lecture bas niveau (voir
        :mod:`pixoscope.io.backend_base`).

    Attributes
    ----------
    handle : ImageHandle
    bands : list of BandInfo
    channel_mapping : ChannelMapping
    """

    def __init__(self, handle: ImageHandle) -> None:
        self.handle = handle
        self.bands: list[BandInfo] = [
            BandInfo(
                index=i,
                name=(handle.band_names[i] if handle.band_names else f"Bande {i + 1}"),
                dtype=handle.dtype,
            )
            for i in range(handle.n_bands)
        ]
        self.channel_mapping: ChannelMapping = default_channel_mapping(handle.n_bands)
        self._display_range: dict[int, tuple[float, float]] = {
            band.index: _DEFAULT_RANGE_BY_DTYPE.get(str(band.dtype), (0.0, 1.0))
            for band in self.bands
            if str(band.dtype) in _DEFAULT_RANGE_BY_DTYPE
        }
        self._manual_range: set[int] = set()
        self.stats_cache: dict[int, BandStats] = {}
        #: Algorithme de rehaussement optionnel (voir
        #: :mod:`pixoscope.processing`), appliqué après composition
        #: RVB/gris et étirement — jamais sur les données brutes.
        self.processing_key: str | None = None
        self.processing_params: dict[str, object] = {}

    @classmethod
    def open(cls, path: str | Path, backend: ImageBackend | None = None) -> ImageDataset:
        """Ouvre un fichier image et retourne l'``ImageDataset`` associé.

        Parameters
        ----------
        path : str or pathlib.Path
            Chemin du fichier à ouvrir.
        backend : ImageBackend, optional
            Backend à utiliser explicitement. Par défaut, sélection
            automatique via
            :func:`pixoscope.io.backend_registry.open_image`.

        Returns
        -------
        ImageDataset
        """
        from pixoscope.io.backend_registry import open_image

        handle = backend.open(Path(path)) if backend is not None else open_image(Path(path))
        return cls(handle)

    @property
    def shape(self) -> tuple[int, int]:
        """Dimensions ``(height, width)`` en pleine résolution."""
        return self.handle.height, self.handle.width

    @property
    def levels(self) -> list[PyramidLevelInfo]:
        """Niveaux de pyramide disponibles (voir :mod:`pixoscope.core.pyramid`)."""
        return self.handle.levels

    def set_channel_mapping(self, mapping: ChannelMapping) -> None:
        """Change le mapping de bandes utilisé pour le rendu.

        N'entraîne aucune relecture disque à l'appel — seul un futur
        appel à :meth:`read_display_tile` lira, le cas échéant, une
        bande pas encore rencontrée.

        Parameters
        ----------
        mapping : ChannelMapping
        """
        self.channel_mapping = mapping

    def set_display_range(self, band_index: int, vmin: float, vmax: float) -> None:
        """Fixe manuellement l'intervalle d'étirement d'une bande.

        Parameters
        ----------
        band_index : int
        vmin, vmax : float
        """
        self._display_range[band_index] = (vmin, vmax)
        self._manual_range.add(band_index)

    def display_range(self, band_index: int) -> tuple[float, float]:
        """Retourne l'intervalle d'étirement courant d'une bande.

        Parameters
        ----------
        band_index : int

        Returns
        -------
        tuple of float
            ``(vmin, vmax)``. Si aucune statistique ni valeur manuelle
            n'est encore disponible, retombe sur l'intervalle par défaut
            du type de la bande (voir ``_DEFAULT_RANGE_BY_DTYPE``), ou
            ``(0.0, 1.0)`` pour un type flottant sans référence connue.
        """
        return self._display_range.get(band_index, (0.0, 1.0))

    def set_band_stats(self, band_index: int, stats: BandStats) -> None:
        """Enregistre des statistiques et met à jour l'étirement auto.

        N'écrase pas un intervalle fixé manuellement par l'utilisateur
        via :meth:`set_display_range`.

        Parameters
        ----------
        band_index : int
        stats : pixoscope.core.stats.BandStats
        """
        self.stats_cache[band_index] = stats
        if band_index not in self._manual_range:
            self._display_range[band_index] = auto_stretch_range(stats)

    def set_processing(self, key: str | None, **params: object) -> None:
        """Active (ou désactive) un algorithme de rehaussement post-affichage.

        Parameters
        ----------
        key : str or None
            Clé de :class:`pixoscope.processing.registry.ProcessingRegistry`,
            ou ``None`` pour revenir à l'affichage sans rehaussement
            supplémentaire.
        **params
            Paramètres transmis à l'algorithme.
        """
        self.processing_key = key
        self.processing_params = params

    def read_display_tile(self, level: int, x: int, y: int, w: int, h: int) -> np.ndarray:
        """Lit et compose une tuile prête à afficher (RVB ou gris, uint8).

        Seules les bandes physiques réellement utilisées par le mapping
        courant sont lues depuis le disque.

        Parameters
        ----------
        level : int
            Niveau de pyramide à lire.
        x, y, w, h : int
            Fenêtre à lire, en coordonnées du niveau demandé.

        Returns
        -------
        numpy.ndarray
            ``(h, w)`` en niveaux de gris, ``(h, w, 3)`` en couleur.
        """
        mapping = self.channel_mapping
        needed = mapping.band_indices()
        if not needed:
            return np.zeros((h, w), dtype=np.uint8)

        raw = self.handle.read_region(level, x, y, w, h, bands=needed)
        # read_region retourne (h, w) si une seule bande, (h, w, n) sinon.
        if raw.ndim == 2:
            by_band = {needed[0]: raw}
        else:
            by_band = {b: raw[..., i] for i, b in enumerate(needed)}

        def _plane(band_index: int | None) -> np.ndarray:
            if band_index is None or band_index not in by_band:
                return np.zeros((h, w), dtype=np.uint8)
            vmin, vmax = self.display_range(band_index)
            return _stretch_to_uint8(by_band[band_index], vmin, vmax)

        if mapping.is_grayscale:
            result = _plane(mapping.gray)
        else:
            result = np.stack([_plane(mapping.red), _plane(mapping.green), _plane(mapping.blue)], axis=-1)

        if self.processing_key:
            from pixoscope.processing.registry import ProcessingRegistry

            result = ProcessingRegistry.apply(self.processing_key, result, **self.processing_params)
        return result

    def close(self) -> None:
        """Ferme le handle sous-jacent."""
        self.handle.close()

    def __enter__(self) -> ImageDataset:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
