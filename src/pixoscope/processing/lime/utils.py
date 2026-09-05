"""Fonctions utilitaires pour LIME/DUAL — port de ``LowLightImageEnhancement/utils.py``."""

from __future__ import annotations


def get_sparse_neighbor(p: int, n: int, m: int) -> dict[int, tuple[int, int, int]]:
    """Retourne les 4-voisins de ``p`` dans une matrice creuse ``n x m`` aplatie.

    Parameters
    ----------
    p : int
        Indice du pixel dans la matrice creuse (image aplatie).
    n : int
        Nombre de lignes de l'image d'origine.
    m : int
        Nombre de colonnes de l'image d'origine.

    Returns
    -------
    dict of int to tuple
        Clés : indices des voisins dans la matrice creuse. Valeurs :
        ``(i, j, x)`` où ``i, j`` sont les coordonnées 2D du voisin et
        ``x`` la direction (0 = vertical, 1 = horizontal).
    """
    i, j = p // m, p % m
    neighbors: dict[int, tuple[int, int, int]] = {}
    if i - 1 >= 0:
        neighbors[(i - 1) * m + j] = (i - 1, j, 0)
    if i + 1 < n:
        neighbors[(i + 1) * m + j] = (i + 1, j, 0)
    if j - 1 >= 0:
        neighbors[i * m + j - 1] = (i, j - 1, 1)
    if j + 1 < m:
        neighbors[i * m + j + 1] = (i, j + 1, 1)
    return neighbors
