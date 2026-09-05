"""LIME et DUAL : rehaussement par estimation de carte d'illumination.

Implémente Guo et al. *LIME* et Zhang et al. *DUAL* — voir
https://github.com/pvnieo/Low-light-Image-Enhancement.

**Limite connue, assumée telle quelle** : la carte d'illumination est
raffinée en résolvant un système linéaire creux de taille
``n_pixels x n_pixels`` — le coût croît donc directement avec la taille
de l'image. Marqué ``slow_on_large_images=True`` (voir
:mod:`pixoscope.processing.registry`).
"""

from __future__ import annotations

import numpy as np

from pixoscope.processing.lime.utils import get_sparse_neighbor
from pixoscope.processing.registry import ProcessingRegistry


def _spatial_affinity_kernel(spatial_sigma: float, size: int = 15) -> np.ndarray:
    """Noyau gaussien d'affinité spatiale, de taille ``size x size``."""
    from scipy.spatial import distance

    kernel = np.zeros((size, size))
    center = (size // 2, size // 2)
    for i in range(size):
        for j in range(size):
            kernel[i, j] = np.exp(-0.5 * distance.euclidean((i, j), center) ** 2 / spatial_sigma**2)
    return kernel


def _smoothness_weights(illumination: np.ndarray, direction: int, kernel: np.ndarray, eps: float) -> np.ndarray:
    import cv2
    from scipy.ndimage import convolve

    gradient = cv2.Sobel(illumination, cv2.CV_64F, int(direction == 1), int(direction == 0), ksize=1)
    normalization = convolve(np.ones_like(illumination), kernel, mode="constant")
    normalization = normalization / (np.abs(convolve(gradient, kernel, mode="constant")) + eps)
    return normalization / (np.abs(gradient) + eps)


def _fuse_exposures(
    image: np.ndarray, under_exposed: np.ndarray, over_exposed: np.ndarray, bc: float, bs: float, be: float
) -> np.ndarray:
    import cv2

    merger = cv2.createMergeMertens(bc, bs, be)
    images = [np.clip(x * 255, 0, 255).astype("uint8") for x in (image, under_exposed, over_exposed)]
    return merger.process(images)


def _refine_illumination_map(
    illumination: np.ndarray, gamma: float, lamda: float, kernel: np.ndarray, eps: float
) -> np.ndarray:
    from scipy.sparse import csr_matrix, diags
    from scipy.sparse.linalg import spsolve

    wx = _smoothness_weights(illumination, direction=1, kernel=kernel, eps=eps)
    wy = _smoothness_weights(illumination, direction=0, kernel=kernel, eps=eps)

    n, m = illumination.shape
    flat = illumination.copy().flatten()

    row, column, data = [], [], []
    for p in range(n * m):
        diag = 0.0
        for q, (k, col, direction) in get_sparse_neighbor(p, n, m).items():
            weight = wx[k, col] if direction else wy[k, col]
            row.append(p)
            column.append(q)
            data.append(-weight)
            diag += weight
        row.append(p)
        column.append(p)
        data.append(diag)
    laplacian = csr_matrix((data, (row, column)), shape=(n * m, n * m))

    identity = diags([np.ones(n * m)], [0])
    system = identity + lamda * laplacian
    refined = spsolve(csr_matrix(system), flat, permc_spec=None, use_umfpack=True).reshape((n, m))

    return np.clip(refined, eps, 1) ** gamma


def _correct_underexposure(
    image: np.ndarray, gamma: float, lamda: float, kernel: np.ndarray, eps: float
) -> np.ndarray:
    illumination = np.max(image, axis=-1)
    refined = _refine_illumination_map(illumination, gamma, lamda, kernel, eps)
    refined_3d = np.repeat(refined[..., None], 3, axis=-1)
    return image / refined_3d


def enhance_image_exposure(
    image: np.ndarray,
    gamma: float = 0.6,
    lamda: float = 0.15,
    dual: bool = True,
    sigma: int = 3,
    bc: float = 1.0,
    bs: float = 1.0,
    be: float = 1.0,
    eps: float = 1e-3,
) -> np.ndarray:
    """Rehausse une image sous-exposée (LIME) ou sous/sur-exposée (DUAL).

    Parameters
    ----------
    image : numpy.ndarray
        Image ``uint8`` RVB, ``(h, w, 3)``.
    gamma : float, optional
        Exposant de correction gamma appliqué à la carte d'illumination.
    lamda : float, optional
        Poids du terme de régularité dans le raffinement de la carte.
    dual : bool, optional
        ``True`` : méthode DUAL (corrige aussi la sur-exposition, fusionne
        les deux versions). ``False`` : méthode LIME (sous-exposition
        seule).
    sigma : int, optional
        Écart-type spatial du noyau d'affinité.
    bc, bs, be : float, optional
        Poids de fusion (contraste, saturation, exposition — méthode
        DUAL uniquement).
    eps : float, optional
        Constante de stabilité numérique.

    Returns
    -------
    numpy.ndarray
        Image ``uint8`` ``(h, w, 3)``.
    """
    kernel = _spatial_affinity_kernel(sigma)
    normalized = image.astype(np.float64) / 255.0
    under_corrected = _correct_underexposure(normalized, gamma, lamda, kernel, eps)

    if dual:
        inverted = 1 - normalized
        over_corrected = 1 - _correct_underexposure(inverted, gamma, lamda, kernel, eps)
        corrected = _fuse_exposures(normalized, under_corrected, over_corrected, bc, bs, be)
    else:
        corrected = under_corrected

    return np.clip(corrected * 255, 0, 255).astype(np.uint8)


@ProcessingRegistry.register(
    "lime",
    "LIME: Low-Light Image Enhancement",
    requires=("cv2", "scipy"),
    slow_on_large_images=True,
)
def lime(image: np.ndarray, gamma: float = 0.6, lamda: float = 0.15, sigma: int = 3) -> np.ndarray:
    """Rehaussement LIME (sous-exposition seule). Voir :func:`enhance_image_exposure`."""
    if image.ndim != 3:
        raise ValueError("lime ne traite que les images RVB (h, w, 3)")
    return enhance_image_exposure(image, gamma=gamma, lamda=lamda, dual=False, sigma=sigma)


@ProcessingRegistry.register(
    "dual",
    "DUAL: Dual Illumination Estimation",
    requires=("cv2", "scipy"),
    slow_on_large_images=True,
)
def dual(image: np.ndarray, gamma: float = 0.6, lamda: float = 0.15, sigma: int = 3) -> np.ndarray:
    """Rehaussement DUAL (sous- et sur-exposition). Voir :func:`enhance_image_exposure`."""
    if image.ndim != 3:
        raise ValueError("dual ne traite que les images RVB (h, w, 3)")
    return enhance_image_exposure(image, gamma=gamma, lamda=lamda, dual=True, sigma=sigma)
