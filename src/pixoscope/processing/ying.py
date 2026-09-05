"""Ying (2017) : rehaussement de contraste par fusion d'expositions.

.. warning::
    **Licence amont non établie.** La référence prise pour cet algorithme
    (https://github.com/AndyHuang1995/Image-Contrast-Enhancement) ne
    porte aucun fichier ``LICENSE`` — voir ``THIRD_PARTY_LICENSES.md``
    §2.2. Ce module n'est **pas** enregistré par défaut à l'import de
    :mod:`pixoscope.processing` ; il faut appeler explicitement
    :func:`pixoscope.processing.enable_experimental_unlicensed_algorithms`
    pour l'activer, en connaissance de cause.

Référence : Zhenqiang Ying, Ge Li, Yurui
Ren, Ronggang Wang, Wenmin Wang, *A New Image Contrast Enhancement
Algorithm Using Exposure Fusion Framework*, CAIP 2017.
"""

from __future__ import annotations

import numpy as np

from pixoscope.processing.registry import ProcessingRegistry


def _compute_texture_weights(image: np.ndarray, sigma: int, sharpness: float) -> tuple[np.ndarray, np.ndarray]:
    import scipy.signal

    dt0_v = np.vstack((np.diff(image, n=1, axis=0), image[0, :] - image[-1, :]))
    dt0_h = np.vstack(
        (np.diff(image, n=1, axis=1).conj().T, image[:, 0].conj().T - image[:, -1].conj().T)
    ).conj().T

    gauker_h = scipy.signal.convolve2d(dt0_h, np.ones((1, sigma)), mode="same")
    gauker_v = scipy.signal.convolve2d(dt0_v, np.ones((sigma, 1)), mode="same")

    weight_h = 1 / (np.abs(gauker_h) * np.abs(dt0_h) + sharpness)
    weight_v = 1 / (np.abs(gauker_v) * np.abs(dt0_v) + sharpness)
    return weight_h, weight_v


def _solve_linear_equation(image: np.ndarray, wx: np.ndarray, wy: np.ndarray, lamda: float) -> np.ndarray:
    import scipy.sparse
    import scipy.sparse.linalg

    r, c = image.shape
    k = r * c
    dx = -lamda * wx.flatten("F")
    dy = -lamda * wy.flatten("F")
    tempx = np.roll(wx, 1, axis=1)
    tempy = np.roll(wy, 1, axis=0)
    dxa = -lamda * tempx.flatten("F")
    dya = -lamda * tempy.flatten("F")

    tmp = wx[:, -1]
    tempx = np.concatenate((tmp[:, None], np.zeros((r, c - 1))), axis=1)
    tmp = wy[-1, :]
    tempy = np.concatenate((tmp[None, :], np.zeros((r - 1, c))), axis=0)
    dxd1 = -lamda * tempx.flatten("F")
    dyd1 = -lamda * tempy.flatten("F")

    wx = wx.copy()
    wy = wy.copy()
    wx[:, -1] = 0
    wy[-1, :] = 0
    dxd2 = -lamda * wx.flatten("F")
    dyd2 = -lamda * wy.flatten("F")

    ax = scipy.sparse.spdiags(np.concatenate((dxd1[:, None], dxd2[:, None]), axis=1).T, np.array([-k + r, -r]), k, k)
    ay = scipy.sparse.spdiags(np.concatenate((dyd1[None, :], dyd2[None, :]), axis=0), np.array([-r + 1, -1]), k, k)
    diag = 1 - (dx + dy + dxa + dya)
    a = ((ax + ay) + (ax + ay).conj().T + scipy.sparse.spdiags(diag, 0, k, k)).T

    solved = scipy.sparse.linalg.spsolve(scipy.sparse.csr_matrix(a), image.flatten("F"))
    return np.reshape(solved, (r, c), order="F")


def _smooth(image: np.ndarray, lamda: float = 0.01, sigma: int = 3, sharpness: float = 0.001) -> np.ndarray:
    import cv2

    normalized = cv2.normalize(image.astype("float64"), None, 0.0, 1.0, cv2.NORM_MINMAX)
    wx, wy = _compute_texture_weights(normalized.copy(), sigma, sharpness)
    return _solve_linear_equation(normalized, wx, wy, lamda)


def _rgb_to_geometric_mean(image: np.ndarray) -> np.ndarray:
    import cv2

    if image.shape[2] != 3:
        return image
    normalized = cv2.normalize(image.astype("float64"), None, 0.0, 1.0, cv2.NORM_MINMAX)
    return np.abs(normalized[:, :, 0] * normalized[:, :, 1] * normalized[:, :, 2]) ** (1 / 3)


def _apply_k(image: np.ndarray, k: float, a: float = -0.3293, b: float = 1.1258) -> np.ndarray:
    beta = np.exp((1 - k**a) * b)
    gamma = k**a
    return (image**gamma) * beta


def _entropy(image: np.ndarray) -> float:
    scaled = np.clip(image * 255, 0, 255).astype(np.uint8)
    _, counts = np.unique(scaled, return_counts=True)
    probabilities = counts / counts.sum()
    return float(-np.sum(probabilities * np.log2(probabilities)))


def _max_entropy_enhance(
    image: np.ndarray, is_underexposed: np.ndarray, a: float = -0.3293, b: float = 1.1258
) -> np.ndarray:
    import cv2
    import scipy.optimize

    small = cv2.resize(image, (50, 50), interpolation=cv2.INTER_AREA)
    small = np.clip(small, 0, None).real
    gray_small = _rgb_to_geometric_mean(small)

    mask = cv2.resize(is_underexposed.astype(np.float32), (50, 50), interpolation=cv2.INTER_CUBIC)
    mask = (mask >= 0.5).astype(np.float64)
    sample = gray_small[mask == 1]

    if sample.size == 0:
        return image

    objective = lambda k: -_entropy(_apply_k(sample, k))  # noqa: E731
    optimal_k = scipy.optimize.fminbound(objective, 1, 7)
    return _apply_k(image, optimal_k, a, b) - 0.01


@ProcessingRegistry.register("ying", "Ying (2017) — fusion d'expositions", requires=("cv2", "scipy"))
def ying(image: np.ndarray, mu: float = 0.5, a: float = -0.3293, b: float = 1.1258) -> np.ndarray:
    """Rehaussement de contraste par fusion d'une version sous- et sur-exposée.

    Parameters
    ----------
    image : numpy.ndarray
        Image ``uint8`` RVB, ``(h, w, 3)``.
    mu : float, optional
        Poids de la carte d'illumination dans la fusion finale.
    a, b : float, optional
        Paramètres du modèle de caméra (voir l'article).

    Returns
    -------
    numpy.ndarray
        Image ``uint8`` ``(h, w, 3)``.
    """
    import cv2

    if image.ndim != 3:
        raise ValueError("ying ne traite que les images RVB (h, w, 3)")

    lamda = 0.5
    sigma = 5
    normalized = cv2.normalize(image.astype("float64"), None, 0.0, 1.0, cv2.NORM_MINMAX)

    max_channel = np.max(normalized, axis=2)
    scale = 0.5
    target_h, target_w = max(1, round(max_channel.shape[0] * scale)), max(1, round(max_channel.shape[1] * scale))
    resized = cv2.resize(max_channel, dsize=(target_w, target_h), interpolation=cv2.INTER_CUBIC)
    illumination = cv2.resize(
        _smooth(resized, lamda, sigma), (max_channel.shape[1], max_channel.shape[0]), interpolation=cv2.INTER_AREA
    )

    is_underexposed = illumination < 0.5
    enhanced = _max_entropy_enhance(normalized, is_underexposed)

    weight = illumination[:, :, None] ** mu
    result = normalized * weight + enhanced * (1 - weight)
    return np.clip(result * 255.0, 0, 255).astype(np.uint8)
