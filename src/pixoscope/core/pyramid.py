"""Sélection du niveau de pyramide à afficher selon le zoom courant.

Ne contient aucune I/O : les niveaux eux-mêmes viennent de
``ImageHandle.levels`` (voir :mod:`pixoscope.io.backend_base`). Ce
module se contente de choisir, pour un facteur de zoom donné, quel
niveau offre la meilleure résolution sans dépasser ce qui est
réellement affichable à l'écran.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from pixoscope.io.backend_base import PyramidLevelInfo


def level_for_zoom(levels: Sequence[PyramidLevelInfo], zoom: float) -> int:
    """Choisit le niveau de pyramide le plus adapté à un facteur de zoom.

    Parameters
    ----------
    levels : sequence of PyramidLevelInfo
        Niveaux disponibles, triés du plus résolu (index 0, pleine
        résolution) au moins résolu.
    zoom : float
        Facteur d'affichage courant, en pixels écran par pixel image
        **pleine résolution** (niveau 0). ``zoom == 1.0`` correspond à
        un affichage 1:1 ; ``zoom < 1`` signifie qu'on voit l'image
        dézoomée (plus de pixels image que de pixels écran).

    Returns
    -------
    int
        Indice du niveau à lire. Ne dépasse jamais un facteur de
        sous-échantillonnage supérieur à ce que ``zoom`` requiert, pour
        ne jamais afficher une image plus floue que nécessaire ; ne
        choisit jamais non plus un niveau plus résolu que le niveau 0.

    Notes
    -----
    Le niveau ``i`` est supposé être environ ``2**i`` fois plus petit
    que le niveau 0 (convention de pyramide standard, y compris pour
    les pyramides construites par
    :mod:`pixoscope.io.pyramid_builder`). Un backend dont les niveaux ne
    suivent pas cette convention (facteur non-2) reste géré correctement
    grâce au calcul direct sur ``levels[i].width`` plutôt que sur un
    facteur théorique.
    """
    if not levels:
        raise ValueError("La liste de niveaux ne doit pas être vide")
    if zoom >= 1.0 or len(levels) == 1:
        return 0

    full_width = levels[0].width
    # On veut le niveau le plus bas-résolu dont le sous-échantillonnage
    # reste <= 1/zoom (donc au moins aussi résolu que ce que l'écran peut
    # montrer), pour ne pas afficher une image plus floue que nécessaire.
    max_downsample = 1.0 / zoom
    best = 0
    for info in levels:
        downsample = full_width / info.width if info.width else math.inf
        if downsample <= max_downsample:
            best = info.level
        else:
            break
    return best
