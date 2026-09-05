"""Socle d'ajustements — numpy uniquement, toujours disponible.

Transformations simples (linéaire, étirement de contraste, gamma, log,
sigmoïde) exprimées entièrement en LUT numpy sur des images déjà en
``uint8`` (voir la convention de :mod:`pixoscope.processing.registry`) —
aucune dépendance à OpenCV/PIL nécessaire.
"""

from __future__ import annotations

import numpy as np

from pixoscope.processing.registry import ProcessingRegistry


def _apply_lut(image: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Applique une LUT 256 valeurs à une image ``uint8`` (gris ou RVB)."""
    return lut[image]


@ProcessingRegistry.register("linear", "Linear")
def linear(image: np.ndarray) -> np.ndarray:
    """Ne fait rien : l'étirement linéaire est déjà appliqué en amont.

    Conservé comme option explicite dans le menu de rehaussement, pour
    repartir d'un état "neutre" en un clic.

    Parameters
    ----------
    image : numpy.ndarray

    Returns
    -------
    numpy.ndarray
        ``image``, inchangée.
    """
    return image


@ProcessingRegistry.register("contrast_stretch", "Contrast stretching (percentiles)")
def contrast_stretch(image: np.ndarray, low_percentile: float = 2.0, high_percentile: float = 98.0) -> np.ndarray:
    """Étire le contraste entre deux percentiles, robuste aux valeurs extrêmes.

    Parameters
    ----------
    image : numpy.ndarray
        Image ``uint8``, ``(h, w)`` ou ``(h, w, 3)``.
    low_percentile, high_percentile : float, optional
        Percentiles bas/haut de l'étirement.

    Returns
    -------
    numpy.ndarray
    """
    low, high = np.percentile(image, (low_percentile, high_percentile))
    if high <= low:
        return image
    scaled = (image.astype(np.float32) - low) * (255.0 / (high - low))
    return np.clip(scaled, 0, 255).astype(np.uint8)


@ProcessingRegistry.register("gamma", "Gamma correction")
def gamma_correction(image: np.ndarray, gamma: float | None = None) -> np.ndarray:
    """Correction gamma (loi de puissance), avec estimation automatique optionnelle.

    Parameters
    ----------
    image : numpy.ndarray
        Image ``uint8``.
    gamma : float, optional
        Exposant gamma. ``None`` (défaut) estime une valeur qui amène la
        luminance moyenne de l'image à 128, calculée directement en
        ``uint8``.

    Returns
    -------
    numpy.ndarray
    """
    if gamma is None:
        mean = float(np.mean(image))
        mean = max(mean, 1.0)  # évite log(0)
        gamma = np.log(0.5) / np.log(mean / 255.0)
    gamma = max(gamma, 1e-3)
    lut = ((np.arange(256, dtype=np.float32) / 255.0) ** (1.0 / gamma) * 255.0).clip(0, 255).astype(np.uint8)
    return _apply_lut(image, lut)


@ProcessingRegistry.register("log_correction", "Logarithmic correction")
def log_correction(image: np.ndarray, gain: float = 1.0) -> np.ndarray:
    """Correction logarithmique : ``out = gain * log2(1 + in / 255) * 255``.

    Parameters
    ----------
    image : numpy.ndarray
    gain : float, optional

    Returns
    -------
    numpy.ndarray
    """
    lut = (gain * np.log2(1.0 + np.arange(256, dtype=np.float32) / 255.0) * 255.0).clip(0, 255).astype(np.uint8)
    return _apply_lut(image, lut)


@ProcessingRegistry.register("sigmoid_correction", "Sigmoid correction")
def sigmoid_correction(image: np.ndarray, cutoff: float = 0.5, gain: float = 10.0) -> np.ndarray:
    """Correction sigmoïde : contraste en "S" centré sur ``cutoff``.

    Parameters
    ----------
    image : numpy.ndarray
    cutoff : float, optional
        Point d'inflexion, en fraction de la plage ``[0, 1]``.
    gain : float, optional
        Pente de la sigmoïde.

    Returns
    -------
    numpy.ndarray
    """
    x = np.arange(256, dtype=np.float32) / 255.0
    lut = (1.0 / (1.0 + np.exp(gain * (cutoff - x))) * 255.0).clip(0, 255).astype(np.uint8)
    return _apply_lut(image, lut)
