"""Calcul de statistiques et d'histogramme échantillonnés.

Ces fonctions ne doivent **jamais** être appelées sur l'image pleine
résolution — l'appelant
(``pixoscope.core.image_model.ImageDataset``) leur passe systématiquement
un niveau de pyramide basse résolution (voir
:func:`pixoscope.core.pyramid.level_for_zoom` et
``ImageHandle.read_overview``). Ce module reste pur numpy, sans I/O ni
Qt, pour rester testable et réutilisable côté ``ui`` (thread de calcul)
sans dépendance croisée.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

#: Percentiles calculés par défaut, utilisés notamment pour un étirement
#: de contraste robuste aux valeurs aberrantes (voir
#: ``ImageDataset.auto_stretch``).
DEFAULT_PERCENTILES: tuple[int, ...] = (1, 2, 25, 50, 75, 98, 99)

#: Nombre de classes de l'histogramme calculé.
DEFAULT_HISTOGRAM_BINS = 256


@dataclass(frozen=True)
class BandStats:
    """Statistiques d'une bande, calculées sur un échantillon de l'image.

    Attributes
    ----------
    minimum, maximum, mean, std : float
        Statistiques usuelles.
    percentiles : dict of int to float
        Valeurs aux percentiles de :data:`DEFAULT_PERCENTILES` (clé =
        percentile entier, ex. ``98``).
    histogram_counts : numpy.ndarray
        Effectifs de l'histogramme (longueur ``DEFAULT_HISTOGRAM_BINS``).
    histogram_edges : numpy.ndarray
        Bornes des classes de l'histogramme (longueur
        ``DEFAULT_HISTOGRAM_BINS + 1``).
    sample_shape : tuple of int
        Forme du tableau réellement échantillonné pour ce calcul —
        permet à l'IHM d'afficher "statistiques calculées sur un
        aperçu 1024x1024" plutôt que de laisser croire à un calcul
        exhaustif.
    """

    minimum: float
    maximum: float
    mean: float
    std: float
    percentiles: dict[int, float] = field(default_factory=dict)
    histogram_counts: np.ndarray = field(default_factory=lambda: np.zeros(0))
    histogram_edges: np.ndarray = field(default_factory=lambda: np.zeros(0))
    sample_shape: tuple[int, ...] = ()


def compute_band_stats(
    sample: np.ndarray,
    *,
    percentiles: Sequence[int] = DEFAULT_PERCENTILES,
    bins: int = DEFAULT_HISTOGRAM_BINS,
) -> BandStats:
    """Calcule les statistiques d'une bande à partir d'un échantillon 2D.

    Parameters
    ----------
    sample : numpy.ndarray
        Tableau 2D (une seule bande, déjà extraite). Doit être un
        niveau de pyramide sous-échantillonné pour une grosse image —
        cette fonction ne fait elle-même aucun sous-échantillonnage.
    percentiles : sequence of int, optional
        Percentiles à calculer.
    bins : int, optional
        Nombre de classes de l'histogramme.

    Returns
    -------
    BandStats

    Notes
    -----
    Les valeurs ``NaN`` (nodata flottant courant) sont ignorées via les
    variantes ``nan*`` de numpy plutôt que de faire planter le calcul.
    """
    finite = sample[np.isfinite(sample)] if np.issubdtype(sample.dtype, np.floating) else sample.ravel()
    if finite.size == 0:
        return BandStats(minimum=0.0, maximum=0.0, mean=0.0, std=0.0, sample_shape=sample.shape)

    minimum = float(np.min(finite))
    maximum = float(np.max(finite))
    mean = float(np.mean(finite))
    std = float(np.std(finite))
    pct_values = np.percentile(finite, percentiles)
    pct = {int(p): float(v) for p, v in zip(percentiles, pct_values, strict=True)}

    counts, edges = np.histogram(finite, bins=bins, range=(minimum, maximum) if maximum > minimum else None)

    return BandStats(
        minimum=minimum,
        maximum=maximum,
        mean=mean,
        std=std,
        percentiles=pct,
        histogram_counts=counts,
        histogram_edges=edges,
        sample_shape=sample.shape,
    )


def auto_stretch_range(stats: BandStats, low: int = 2, high: int = 98) -> tuple[float, float]:
    """Propose un intervalle d'affichage robuste à partir des percentiles.

    Parameters
    ----------
    stats : BandStats
        Statistiques déjà calculées (doit contenir ``low`` et ``high``
        dans ``percentiles``, sinon on retombe sur min/max).
    low, high : int, optional
        Percentiles bas et haut à utiliser pour l'étirement.

    Returns
    -------
    tuple of float
        ``(vmin, vmax)`` à utiliser pour la LUT d'affichage.
    """
    vmin = stats.percentiles.get(low, stats.minimum)
    vmax = stats.percentiles.get(high, stats.maximum)
    if vmax <= vmin:
        vmin, vmax = stats.minimum, stats.maximum
    if vmax <= vmin:
        vmax = vmin + 1.0
    return vmin, vmax
