"""Registre des algorithmes de rehaussement dynamique.

Registre à décorateurs : chaque module de :mod:`pixoscope.processing`
s'enregistre lui-même avec :func:`register`, en déclarant ses
dépendances optionnelles (``requires``). :meth:`ProcessingRegistry.available`
ne retourne que les algorithmes réellement utilisables dans
l'environnement courant — un algorithme dont une dépendance de l'extra
``enhance`` manque est simplement absent du menu de l'IHM plutôt que de
faire planter tout le paquet à l'import.

Contrat commun à tous les algorithmes enregistrés
--------------------------------------------------
``f(image: np.ndarray, **params) -> np.ndarray``, où ``image`` est une
tuile déjà étirée en ``uint8`` (``(h, w)`` niveaux de gris ou
``(h, w, 3)`` RVB — voir ``ImageDataset.read_display_tile``) et le
résultat a la même forme et le même dtype. Ce choix évite la
prolifération de branches par dtype : le rehaussement s'applique
toujours après l'étirement d'affichage, jamais sur les données brutes.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

ProcessingFunction = Callable[..., np.ndarray]


@dataclass(frozen=True)
class AlgorithmSpec:
    """Décrit un algorithme de rehaussement enregistré.

    Attributes
    ----------
    key : str
        Identifiant stable (utilisé en code, ex. dans les tests).
    label : str
        Libellé affiché dans l'IHM.
    func : callable
        Fonction ``(image, **params) -> image``.
    requires : tuple of str
        Modules Python devant être importables pour que l'algorithme
        soit disponible (ex. ``("cv2",)``). Vide pour le socle
        (numpy seul).
    grayscale_only : bool
        ``True`` si l'algorithme ne traite que des images en niveaux de
        gris (2D).
    slow_on_large_images : bool
        ``True`` si le coût de l'algorithme croît significativement
        avec la taille de l'image (ex. système linéaire creux de taille
        ``n_pixels``) — l'IHM peut s'en servir pour avertir avant de
        lancer le calcul sur une grosse tuile.
    """

    key: str
    label: str
    func: ProcessingFunction
    requires: tuple[str, ...] = ()
    grayscale_only: bool = False
    slow_on_large_images: bool = False


class ProcessingRegistry:
    """Registre global des algorithmes de rehaussement dynamique."""

    _algorithms: dict[str, AlgorithmSpec] = {}

    @classmethod
    def register(
        cls,
        key: str,
        label: str,
        *,
        requires: tuple[str, ...] = (),
        grayscale_only: bool = False,
        slow_on_large_images: bool = False,
    ) -> Callable[[ProcessingFunction], ProcessingFunction]:
        """Décorateur enregistrant une fonction comme algorithme disponible.

        Parameters
        ----------
        key : str
        label : str
        requires : tuple of str, optional
        grayscale_only : bool, optional
        slow_on_large_images : bool, optional

        Returns
        -------
        callable
            Décorateur à appliquer à la fonction de traitement.
        """

        def _decorator(func: ProcessingFunction) -> ProcessingFunction:
            cls._algorithms[key] = AlgorithmSpec(
                key=key,
                label=label,
                func=func,
                requires=requires,
                grayscale_only=grayscale_only,
                slow_on_large_images=slow_on_large_images,
            )
            return func

        return _decorator

    @classmethod
    def is_available(cls, spec: AlgorithmSpec) -> bool:
        """Indique si toutes les dépendances optionnelles de ``spec`` sont importables.

        Parameters
        ----------
        spec : AlgorithmSpec

        Returns
        -------
        bool
        """
        for module_name in spec.requires:
            try:
                importlib.import_module(module_name)
            except ImportError:
                return False
        return True

    @classmethod
    def available(cls) -> list[AlgorithmSpec]:
        """Liste les algorithmes utilisables dans l'environnement courant.

        Returns
        -------
        list of AlgorithmSpec
        """
        return [spec for spec in cls._algorithms.values() if cls.is_available(spec)]

    @classmethod
    def all_specs(cls) -> list[AlgorithmSpec]:
        """Liste tous les algorithmes enregistrés, disponibles ou non.

        Returns
        -------
        list of AlgorithmSpec
        """
        return list(cls._algorithms.values())

    @classmethod
    def get(cls, key: str) -> AlgorithmSpec:
        """Récupère un algorithme par sa clé.

        Parameters
        ----------
        key : str

        Returns
        -------
        AlgorithmSpec

        Raises
        ------
        KeyError
            Si ``key`` n'est pas enregistrée.
        ValueError
            Si l'algorithme est enregistré mais qu'une dépendance
            optionnelle manque.
        """
        spec = cls._algorithms[key]
        if not cls.is_available(spec):
            missing = [m for m in spec.requires if not _is_importable(m)]
            raise ValueError(
                f"Algorithme [{spec.label}] indisponible : dépendance(s) manquante(s) {missing}. "
                "Installez l'extra pixoscope[enhance]."
            )
        return spec

    @classmethod
    def apply(cls, key: str, image: np.ndarray, **params: object) -> np.ndarray:
        """Applique un algorithme enregistré à une image.

        Parameters
        ----------
        key : str
        image : numpy.ndarray
        **params
            Paramètres transmis à la fonction de traitement.

        Returns
        -------
        numpy.ndarray
        """
        spec = cls.get(key)
        if spec.grayscale_only and image.ndim != 2:
            raise ValueError(f"L'algorithme [{spec.label}] ne traite que des images en niveaux de gris")
        return spec.func(image, **params)


def _is_importable(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False
