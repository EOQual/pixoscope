"""Retinex.

Référence : Tomoya Kamata (ISS Inc., 2011).

L'application par canal se fait par une simple boucle sur les 3 canaux
plutôt que via ``skimage.color.adapt_rgb``, pour ne pas ajouter de
dépendance à scikit-image à ce module (seul OpenCV est nécessaire).
"""

from __future__ import annotations

import numpy as np

from pixoscope.processing.registry import ProcessingRegistry


def _retinex_channel(channel: np.ndarray, prefilter: str) -> np.ndarray:
    import cv2

    g1 = (channel.astype(np.float32) + 1.0) / 256.0
    if prefilter == "gaussian":
        g2 = (cv2.GaussianBlur(channel, (201, 201), 100).astype(np.float32) + 1.0) / 257.0
    elif prefilter == "box":
        g2 = (cv2.blur(channel, (51, 51)).astype(np.float32) + 1.0) / 256.0
    else:
        raise ValueError(f"prefilter inconnu : {prefilter!r} (attendu 'gaussian' ou 'box')")

    g3 = np.log(g1 / g2)
    g3_min, g3_max = float(np.min(g3)), float(np.max(g3))
    if g3_max <= g3_min:
        return np.zeros_like(channel, dtype=np.uint8)
    g4 = (g3 - g3_min) / (g3_max - g3_min) * 255.0
    return np.round(g4).astype(np.uint8)


@ProcessingRegistry.register("retinex", "Retinex", requires=("cv2",))
def retinex(image: np.ndarray, prefilter: str = "gaussian") -> np.ndarray:
    """Rehaussement Retinex (rapport image / version floutée, en log).

    Parameters
    ----------
    image : numpy.ndarray
        Image ``uint8``, ``(h, w)`` ou ``(h, w, 3)``.
    prefilter : {"gaussian", "box"}, optional
        Filtre utilisé pour estimer l'illumination locale.

    Returns
    -------
    numpy.ndarray
    """
    if image.ndim == 2:
        return _retinex_channel(image, prefilter)

    out = np.empty_like(image)
    for channel in range(image.shape[2]):
        out[:, :, channel] = _retinex_channel(image[:, :, channel], prefilter)
    return out
