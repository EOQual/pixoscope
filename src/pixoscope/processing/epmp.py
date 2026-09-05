"""EPMP : Entropy-Preserving Mapping Prior.

Référence : Bo-Hao Chen, Yu-Ling Wu, Ling-Feng Shi, *A Fast Image
Contrast Enhancement Algorithm Using Entropy-Preserving Mapping Prior*.
https://github.com/bigmms/entropy-preserving-mapping-prior

``T = inv(A) @ b`` est résolu ici via ``np.linalg.solve(A, b)`` plutôt
que par inversion explicite de la matrice — même résultat mathématique,
sans construire l'inverse.

Le système résolu porte sur un vecteur de 256 valeurs (une par niveau
d'intensité) : son coût ne dépend donc **pas** de la taille de l'image
— contrairement à DHE ou LIME, cet algorithme n'est pas marqué comme
lent sur les grandes images.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from pixoscope.processing.registry import ProcessingRegistry

_THETA0 = 1.3945
_THETA1 = 9.1377
_THETA2 = 0.5


def _psf_to_otf(psf: Sequence[float], shape: tuple[int, int]) -> np.ndarray:
    """Transfert optique d'un filtre PSF, centré par décalage circulaire."""
    psf_row = np.asarray(psf, dtype=np.float64).reshape(1, -1)
    kernel = np.zeros(shape, dtype=np.float64)
    kh, kw = psf_row.shape
    kernel[:kh, :kw] = psf_row
    kernel = np.roll(kernel, -(kw // 2), axis=1)
    kernel = np.roll(kernel, -(kh // 2), axis=0)
    return np.fft.fft2(kernel)


def _histogram_equalization_curve(channel: np.ndarray) -> np.ndarray:
    """Courbe de mapping d'égalisation d'histogramme classique, normalisée ``[0, 1]``."""
    hist, _ = np.histogram(channel.flatten(), 256, (0, 256))
    cdf = hist.cumsum().astype(np.float64)
    cdf_min, cdf_max = cdf.min(), cdf.max()
    if cdf_max <= cdf_min:
        return np.arange(256, dtype=np.float64) / 255.0
    return (cdf - cdf_min) / (cdf_max - cdf_min)


def _identity_curve() -> np.ndarray:
    return np.arange(256, dtype=np.float64) / 255.0


def _mapping_curve(channel: np.ndarray, theta0: float, theta1: float, theta2: float) -> np.ndarray:
    """Résout la courbe de mapping régularisée par gradient (256 valeurs)."""
    t_he = _histogram_equalization_curve(channel)
    t_identity = _identity_curve()

    div_gradient = np.abs(_psf_to_otf([1, -1], (256, 256))) ** 2
    a = (theta0 + theta1) * np.eye(256) + theta2 * div_gradient
    rhs = theta0 * t_he + theta1 * t_identity

    mapping = np.linalg.solve(a, rhs).real
    m_min, m_max = mapping.min(), mapping.max()
    if m_max <= m_min:
        return t_identity
    return (mapping - m_min) / (m_max - m_min)


@ProcessingRegistry.register("epmp", "EPMP: Entropy-Preserving Mapping Prior", requires=("cv2",))
def epmp(image: np.ndarray, theta0: float = _THETA0, theta1: float = _THETA1, theta2: float = _THETA2) -> np.ndarray:
    """Rehaussement de contraste par courbe de mapping régularisée par gradient.

    Parameters
    ----------
    image : numpy.ndarray
        Image ``uint8`` RVB, ``(h, w, 3)``.
    theta0, theta1, theta2 : float, optional
        Poids du modèle (égalisation d'histogramme, identité, régularité).

    Returns
    -------
    numpy.ndarray
        Image ``uint8`` ``(h, w, 3)``.
    """
    import cv2

    if image.ndim != 3:
        raise ValueError("epmp ne traite que les images RVB (h, w, 3)")

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hue, sat, val = cv2.split(hsv)

    mapping = _mapping_curve(val, theta0, theta1, theta2)
    new_val = np.clip(mapping[val] * 255.0, 0, 255).astype(np.uint8)

    merged = cv2.merge((hue, sat, new_val))
    return cv2.cvtColor(merged, cv2.COLOR_HSV2RGB)
