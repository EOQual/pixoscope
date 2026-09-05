"""DHE : Dynamic Histogram Equalization.

.. warning::
    **Licence amont non établie.** La référence prise pour cet algorithme
    (https://github.com/AndyHuang1995/Image-Contrast-Enhancement) ne
    porte aucun fichier ``LICENSE`` — voir ``THIRD_PARTY_LICENSES.md``
    §2.2. Ce module n'est **pas** enregistré par défaut à l'import de
    :mod:`pixoscope.processing` ; il faut appeler explicitement
    :func:`pixoscope.processing.enable_experimental_unlicensed_algorithms`
    pour l'activer, en connaissance de cause.

La conversion RVB/HSV utilise ``skimage.color`` (convention ``[0, 1]``
cohérente de bout en bout) plutôt que ``matplotlib.colors`` — évite
d'introduire matplotlib comme dépendance de la couche de traitement
(voir la règle d'architecture "processing n'importe jamais
matplotlib/Qt").

**Limite connue, assumée telle quelle** : le calcul de la carte de
corrélation locale (:func:`_build_is_hist`) itère pixel par pixel en
Python (``np.corrcoef`` sur une fenêtre 5x5 à chaque pixel) — coût
prohibitif au-delà de quelques centaines de milliers de pixels. Cet
algorithme est marqué ``slow_on_large_images=True`` : l'IHM avertit
avant de le lancer sur une grande tuile.
"""

from __future__ import annotations

import numpy as np

from pixoscope.processing.registry import ProcessingRegistry


def _build_is_hist(image01: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Construit les histogrammes pondérés par gradient et par corrélation I/S.

    Parameters
    ----------
    image01 : numpy.ndarray
        Image RVB en float, valeurs dans ``[0, 1]``.

    Returns
    -------
    tuple of numpy.ndarray
        ``(hist_i, hist_s)``, chacun de forme ``(256, 1)``.
    """
    import scipy.signal
    from skimage.color import rgb2hsv

    height, width, _ = image01.shape
    padded = np.stack([np.pad(image01[:, :, c], 2, mode="edge") for c in range(3)], axis=-1)
    hsv = rgb2hsv(padded)
    intensity = np.round(hsv[:, :, 2] * 255).astype(np.float64)
    saturation = hsv[:, :, 1] * 255

    fh = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
    fv = fh.conj().T

    def _gradient_magnitude(channel: np.ndarray) -> np.ndarray:
        dh = scipy.signal.convolve2d(channel, np.rot90(fh, 2), mode="same")
        dv = scipy.signal.convolve2d(channel, np.rot90(fv, 2), mode="same")
        dh[dh == 0] = 1e-5
        dv[dv == 0] = 1e-5
        return np.sqrt(dh**2 + dv**2)

    grad_i = _gradient_magnitude(intensity)[2 : height + 2, 2 : width + 2]
    grad_s = _gradient_magnitude(saturation)[2 : height + 2, 2 : width + 2]

    i_int = intensity[2 : height + 2, 2 : width + 2].astype(np.uint8)

    # Corrélation locale I/S dans une fenêtre 5x5 autour de chaque pixel
    # -- voir la limite de performance documentée en tête de module.
    rho = np.zeros((height + 4, width + 4))
    for p in range(2, height + 2):
        for q in range(2, width + 2):
            window_i = intensity[p - 2 : p + 3, q - 2 : q + 3]
            window_s = saturation[p - 2 : p + 3, q - 2 : q + 3]
            corr = np.corrcoef(window_i.flatten("F"), window_s.flatten("F"))
            rho[p, q] = corr[0, 1]
    rho = np.abs(rho[2 : height + 2, 2 : width + 2])
    rho[np.isnan(rho)] = 0.0
    weighted_grad_s = (rho * grad_s).astype(np.float64)

    hist_i = np.zeros((256, 1))
    hist_s = np.zeros((256, 1))
    for level in range(255):
        mask = i_int == level
        hist_i[level + 1] = grad_i[mask].sum()
        hist_s[level + 1] = weighted_grad_s[mask].sum()
    return hist_i, hist_s


@ProcessingRegistry.register(
    "dhe",
    "DHE: Dynamic Histogram Equalization",
    requires=("scipy", "skimage"),
    slow_on_large_images=True,
)
def dhe(image: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Dynamic Histogram Equalization pondérée par corrélation intensité/saturation.

    Parameters
    ----------
    image : numpy.ndarray
        Image ``uint8`` RVB, ``(h, w, 3)`` (pas de variante niveaux de gris).
    alpha : float, optional
        Poids du terme "saturation" dans l'histogramme combiné (0 à 1).

    Returns
    -------
    numpy.ndarray
        Image ``uint8`` ``(h, w, 3)``.
    """
    if image.ndim != 3:
        raise ValueError("dhe ne traite que les images RVB (h, w, 3)")

    from skimage.color import hsv2rgb, rgb2hsv

    image01 = image.astype(np.float64) / 255.0
    hist_i, hist_s = _build_is_hist(image01)
    hist_c = alpha * hist_s + (1 - alpha) * hist_i
    hist_sum = hist_c.sum()
    hist_cum = hist_c.cumsum(axis=0)

    hsv = rgb2hsv(image01)
    hue, sat, intensity = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    i_int = np.round(intensity * 255).astype(np.uint8)

    mapping = (hist_cum / hist_sum * 255).ravel() if hist_sum > 0 else np.arange(256, dtype=np.float64)
    new_intensity = np.zeros_like(intensity)
    for level in range(255):
        new_intensity[i_int == level] = mapping[level + 1] / 255.0
    new_intensity[i_int == 255] = 1.0

    result = hsv2rgb(np.stack((hue, sat, new_intensity), axis=2))
    return np.clip(result * 255.0, 0, 255).astype(np.uint8)
