"""IAGCWD : Improved Adaptive Gamma Correction with Weighting Distribution.

Référence : https://github.com/leowang7/iagcwd

Implémentation vectorisée : une unique indexation ``inverse_cdf[image]``
remplace la boucle ``for i in unique_intensity: ...`` habituelle pour ce
type d'algorithme, qui repasserait sur toute l'image à chaque niveau
d'intensité (jusqu'à 256 fois). Utilise ``COLOR_RGB2YCrCb``/
``COLOR_YCrCb2RGB`` : pixoscope décode toujours ses images en RVB
(jamais en BGR).

Nécessite l'extra ``pixoscope[enhance]`` (OpenCV).
"""

from __future__ import annotations

import numpy as np

from pixoscope.processing.registry import ProcessingRegistry


def _agcwd(channel: np.ndarray, alpha: float, truncated_cdf: bool) -> np.ndarray:
    hist, _ = np.histogram(channel.flatten(), 256, (0, 256))
    prob = hist / hist.sum()

    prob_min, prob_max = prob.min(), prob.max()
    if prob_max <= prob_min:
        return channel.copy()

    weighted = (prob - prob_min) / (prob_max - prob_min)
    weighted = np.where(weighted > 0, prob_max * (weighted**alpha), weighted)
    weighted = weighted / weighted.sum()
    cdf = weighted.cumsum()

    inverse_cdf = np.maximum(0.5, 1 - cdf) if truncated_cdf else 1 - cdf

    exponents = inverse_cdf[channel]
    normalized = channel.astype(np.float64) / 255.0
    powered = np.zeros_like(normalized)
    positive = normalized > 0
    np.power(normalized, exponents, where=positive, out=powered)
    out = np.round(255.0 * powered)
    return np.clip(out, 0, 255).astype(np.uint8)


@ProcessingRegistry.register("iagcwd", "IAGCWD (Improved Adaptive Gamma Correction)", requires=("cv2",))
def iagcwd(image: np.ndarray, bright_image: bool = False) -> np.ndarray:
    """Correction gamma adaptative pondérée par la distribution d'histogramme.

    Parameters
    ----------
    image : numpy.ndarray
        Image ``uint8``, ``(h, w)`` ou ``(h, w, 3)``.
    bright_image : bool, optional
        ``True`` pour une image plutôt sur-exposée (traite le négatif de
        la luminance), ``False`` pour une image sous-exposée (défaut).

    Returns
    -------
    numpy.ndarray
    """
    if image.ndim == 2:
        luminance = image
        if bright_image:
            return 255 - _agcwd(255 - luminance, alpha=0.25, truncated_cdf=False)
        return _agcwd(luminance, alpha=0.75, truncated_cdf=True)

    import cv2

    ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    y = ycrcb[:, :, 0]
    if bright_image:
        ycrcb[:, :, 0] = 255 - _agcwd(255 - y, alpha=0.25, truncated_cdf=False)
    else:
        ycrcb[:, :, 0] = _agcwd(y, alpha=0.75, truncated_cdf=True)
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
