"""Histogram Equalization — algorithme générique, implémenté via scikit-image.

Ne porte aucun code d'un dépôt tiers parfois cité en référence pour cet
algorithme (``AndyHuang1995/Image-Contrast-Enhancement``, sans licence —
voir ``THIRD_PARTY_LICENSES.md`` §2.2) : l'égalisation d'histogramme est
un algorithme de traitement d'image générique et non protégeable, et
cette implémentation appelle directement
``skimage.exposure.equalize_hist`` (licence BSD).

Nécessite l'extra ``pixoscope[enhance]`` (scikit-image).
"""

from __future__ import annotations

import numpy as np

from pixoscope.processing.registry import ProcessingRegistry


@ProcessingRegistry.register("he", "HE: Histogram Equalization", requires=("skimage",))
def histogram_equalization(image: np.ndarray) -> np.ndarray:
    """Égalisation d'histogramme, par canal pour une image couleur.

    Parameters
    ----------
    image : numpy.ndarray
        Image ``uint8``, ``(h, w)`` ou ``(h, w, 3)``.

    Returns
    -------
    numpy.ndarray
        Image ``uint8`` égalisée.
    """
    from skimage import exposure

    if image.ndim == 2:
        return np.clip(exposure.equalize_hist(image) * 255.0, 0, 255).astype(np.uint8)

    out = np.empty_like(image)
    for channel in range(image.shape[2]):
        out[:, :, channel] = np.clip(exposure.equalize_hist(image[:, :, channel]) * 255.0, 0, 255)
    return out


@ProcessingRegistry.register("adaptive_eq", "Adaptive equalization (CLAHE, skimage)", requires=("skimage",))
def adaptive_equalization(image: np.ndarray, clip_limit: float = 0.03) -> np.ndarray:
    """Égalisation adaptative (CLAHE via scikit-image), par canal.

    Parameters
    ----------
    image : numpy.ndarray
    clip_limit : float, optional

    Returns
    -------
    numpy.ndarray
    """
    from skimage import exposure

    normalized = image.astype(np.float32) / 255.0
    if image.ndim == 2:
        return np.clip(exposure.equalize_adapthist(normalized, clip_limit=clip_limit) * 255.0, 0, 255).astype(
            np.uint8
        )

    out = np.empty_like(image)
    for channel in range(image.shape[2]):
        out[:, :, channel] = np.clip(
            exposure.equalize_adapthist(normalized[:, :, channel], clip_limit=clip_limit) * 255.0, 0, 255
        )
    return out
