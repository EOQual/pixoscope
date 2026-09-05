"""SUACE : Speeded-Up Adaptive Contrast Enhancement.

Implémentation vectorisée (``np.where``/``np.clip``) de la fonction par
morceaux de l'algorithme : une version itérant pixel par pixel en Python
pur serait de l'ordre de plusieurs dizaines de secondes sur une image de
quelques mégapixels — inutilisable pour un usage interactif.

Référence : http://www.ravimal.com/2017/08/a-fast-simple-and-powerful-contrast.html

Nécessite l'extra ``pixoscope[enhance]`` (OpenCV, pour le flou gaussien).
"""

from __future__ import annotations

import numpy as np

from pixoscope.processing.registry import ProcessingRegistry


def _suace_channel(channel: np.ndarray, distance: float, sigma: float) -> np.ndarray:
    import cv2

    val = channel.astype(np.float32)
    smoothed = cv2.GaussianBlur(channel, (0, 0), sigma).astype(np.float32)
    half_distance = distance / 2.0

    adjuster = np.where(val - smoothed > distance, smoothed + (val - smoothed) * 0.5, smoothed)
    adjuster = np.maximum(adjuster, half_distance)

    b = np.minimum(adjuster + half_distance, 255.0)
    a = np.maximum(b - distance, 0.0)

    stretched = (val - a) / distance * 255.0
    out = np.where(val < a, 0.0, np.where(val > b, 255.0, stretched))
    return np.clip(out, 0, 255).astype(np.uint8)


@ProcessingRegistry.register("suace", "SUACE", requires=("cv2",))
def suace(image: np.ndarray, distance: float = 21.0, sigma: float = 4.625) -> np.ndarray:
    """Speeded-Up Adaptive Contrast Enhancement.

    Parameters
    ----------
    image : numpy.ndarray
        Image ``uint8``, ``(h, w)`` ou ``(h, w, 3)``.
    distance : float, optional
        Doit être strictement positif.
    sigma : float, optional
        Écart-type du flou gaussien d'estimation locale. Doit être
        strictement positif.

    Returns
    -------
    numpy.ndarray
    """
    if distance <= 0 or sigma <= 0:
        raise ValueError(f"distance [{distance}] et sigma [{sigma}] doivent être strictement positifs")

    if image.ndim == 2:
        return _suace_channel(image, distance, sigma)

    out = np.empty_like(image)
    for channel in range(image.shape[2]):
        out[:, :, channel] = _suace_channel(image[:, :, channel], distance, sigma)
    return out
