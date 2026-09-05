"""CLAHE (OpenCV, espace LAB).

Utilise ``COLOR_RGB2LAB``/``COLOR_LAB2RGB`` : contrairement à
``cv2.imread``, les images de pixoscope sont décodées en RVB (via
imageio/tifffile), jamais en BGR.

Nécessite l'extra ``pixoscope[enhance]`` (OpenCV).
"""

from __future__ import annotations

import numpy as np

from pixoscope.processing.registry import ProcessingRegistry


@ProcessingRegistry.register("clahe_lab", "CLAHE (OpenCV, espace LAB)", requires=("cv2",))
def clahe_lab(image: np.ndarray, clip_limit: float = 3.0, tile_grid_size: int = 8) -> np.ndarray:
    """Contrast Limited Adaptive Histogram Equalization.

    Sur une image en niveaux de gris, s'applique directement. Sur une
    image RVB, s'applique uniquement au canal de luminance L de l'espace
    LAB, pour ne pas perturber la teinte.

    Parameters
    ----------
    image : numpy.ndarray
        Image ``uint8``, ``(h, w)`` ou ``(h, w, 3)``.
    clip_limit : float, optional
    tile_grid_size : int, optional
        Taille (côté) de la grille de tuiles locales.

    Returns
    -------
    numpy.ndarray
    """
    import cv2

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))

    if image.ndim == 2:
        return clahe.apply(image)

    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    l_channel = clahe.apply(l_channel)
    merged = cv2.merge((l_channel, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
