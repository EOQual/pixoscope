"""Interface commune à tous les backends de lecture d'image.

Tout le reste de pixoscope (modèle de données, rendu, statistiques) ne
parle qu'à :class:`ImageHandle` — jamais directement à ``tifffile``,
``imageio`` ou GDAL. C'est le point d'extension unique pour ajouter un
nouveau format ou un nouveau moteur de lecture (voir
``pixoscope.plugins`` pour des exemples de backends optionnels).

Convention de niveaux de pyramide
----------------------------------
Le niveau ``0`` est toujours la résolution native (pleine résolution).
Les niveaux suivants sont des résolutions décroissantes (typiquement un
facteur 2 par niveau), ``levels[-1]`` étant la plus basse résolution
disponible. Un backend qui ne sait pas fournir de pyramide expose une
liste ``levels`` à un seul élément.
"""

from __future__ import annotations

import abc
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

import numpy as np


class PyramidLevelInfo(NamedTuple):
    """Métadonnées d'un niveau de pyramide.

    Parameters
    ----------
    level : int
        Indice du niveau (0 = pleine résolution). Nommé ``level`` plutôt
        que ``index`` pour ne pas masquer ``tuple.index()``, dont
        `NamedTuple` hérite.
    width : int
        Largeur du niveau, en pixels.
    height : int
        Hauteur du niveau, en pixels.
    """

    level: int
    width: int
    height: int


class ImageHandle(abc.ABC):
    """Représente une image ouverte, indépendamment de son format.

    Un ``ImageHandle`` ne charge aucune donnée pixel à la construction :
    seules les métadonnées (dimensions, bandes, type, pyramide) sont
    lues. Les pixels ne sont lus qu'à la demande, via
    :meth:`read_region`, et uniquement pour la fenêtre demandée.

    Attributes
    ----------
    path : pathlib.Path
        Chemin du fichier source.
    height : int
        Hauteur de l'image en pleine résolution (niveau 0), en pixels.
    width : int
        Largeur de l'image en pleine résolution (niveau 0), en pixels.
    n_bands : int
        Nombre de bandes/canaux de l'image.
    dtype : numpy.dtype
        Type des pixels natif du fichier.
    band_names : list of str or None
        Noms des bandes s'ils sont connus (métadonnées du fichier),
        sinon ``None``.
    levels : list of PyramidLevelInfo
        Niveaux de pyramide disponibles, triés du plus résolu (index 0)
        au moins résolu.
    """

    path: Path
    height: int
    width: int
    n_bands: int
    dtype: np.dtype
    band_names: list[str] | None
    levels: list[PyramidLevelInfo]

    @abc.abstractmethod
    def read_region(
        self,
        level: int,
        x: int,
        y: int,
        w: int,
        h: int,
        bands: Sequence[int] | None = None,
    ) -> np.ndarray:
        """Lit une fenêtre rectangulaire de l'image à un niveau donné.

        Parameters
        ----------
        level : int
            Indice du niveau de pyramide à lire (0 = pleine résolution).
        x, y : int
            Coin supérieur gauche de la fenêtre, en pixels, dans le
            repère du niveau demandé.
        w, h : int
            Largeur et hauteur de la fenêtre, en pixels.
        bands : sequence of int, optional
            Indices des bandes à lire (0-based). ``None`` lit toutes les
            bandes. Seules les bandes demandées doivent être décodées
            depuis le disque quand le format le permet.

        Returns
        -------
        numpy.ndarray
            Tableau de forme ``(h, w)`` si une seule bande est demandée
            sur une image mono-bande, ``(h, w, len(bands))`` sinon.
        """

    @abc.abstractmethod
    def read_overview(self, level: int | None = None) -> np.ndarray:
        """Lit intégralement un niveau de pyramide basse résolution.

        Utilisé pour les vignettes, le calcul de statistiques et la vue
        d'ensemble — jamais pour la résolution native d'une grosse image.

        Parameters
        ----------
        level : int, optional
            Niveau à lire intégralement. Par défaut, le niveau le moins
            résolu disponible (``levels[-1]``).

        Returns
        -------
        numpy.ndarray
            Tableau ``(h, w)`` ou ``(h, w, n_bands)`` selon le nombre de
            bandes de l'image.
        """

    @abc.abstractmethod
    def close(self) -> None:
        """Libère les ressources associées (descripteurs de fichier...)."""

    def __enter__(self) -> ImageHandle:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


class ImageBackend(abc.ABC):
    """Fabrique d'``ImageHandle`` pour une famille de formats.

    Attributes
    ----------
    name : str
        Identifiant court du backend (ex. ``"tifffile"``, ``"imageio"``),
        utilisé par ``PIXOSCOPE_BACKEND`` pour forcer un choix.
    """

    name: str

    @abc.abstractmethod
    def can_open(self, path: Path) -> bool:
        """Indique si ce backend sait a priori ouvrir ``path``.

        Une vérification légère (extension, éventuellement signature de
        fichier) — ne doit jamais décoder l'image entière.

        Parameters
        ----------
        path : pathlib.Path
            Chemin du fichier à tester.

        Returns
        -------
        bool
        """

    @abc.abstractmethod
    def open(self, path: Path) -> ImageHandle:
        """Ouvre ``path`` et retourne un :class:`ImageHandle`.

        Parameters
        ----------
        path : pathlib.Path
            Chemin du fichier à ouvrir.

        Returns
        -------
        ImageHandle
        """
