#!/usr/bin/env python3
"""Génère une image TIFF synthétique pour tester Pixoscope sur de grosses images.

Utile en l'absence d'image de test suffisamment grosse sous la main : ce
script en construit une à la taille voulue, sans jamais matérialiser
l'image entière en mémoire (écriture par bandes horizontales via
``tifffile.memmap``, comme :mod:`pixoscope.io.pyramid_builder`).

Exemples
--------
.. code-block:: bash

    # ~6000x6000, 3 bandes uint16 (quelques centaines de Mo)
    python scripts/make_synthetic_bigtiff.py /tmp/test.tif --width 6000 --height 6000

    # BigTIFF réellement volumineux (plusieurs Go), mono-bande
    python scripts/make_synthetic_bigtiff.py /tmp/huge.tif --width 40000 --height 40000 --bands 1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tifffile

_STRIPE_ROWS = 2048


def generate_synthetic_tiff(
    path: Path,
    width: int,
    height: int,
    bands: int = 3,
    dtype: str = "uint16",
    seed: int = 0,
) -> None:
    """Écrit une image TIFF procédurale (dégradé + bruit), sans la charger en RAM.

    Parameters
    ----------
    path : pathlib.Path
        Fichier TIFF de sortie.
    width, height : int
        Dimensions de l'image.
    bands : int, optional
        Nombre de bandes (1 = niveaux de gris, 3 = RVB).
    dtype : str, optional
        Type numpy des pixels (ex. ``"uint8"``, ``"uint16"``).
    seed : int, optional
        Graine du bruit procédural, pour un contenu reproductible.
    """
    np_dtype = np.dtype(dtype)
    shape = (height, width) if bands == 1 else (height, width, bands)
    max_value = float(np.iinfo(np_dtype).max) if np.issubdtype(np_dtype, np.integer) else 1.0

    photometric = "rgb" if bands == 3 and np_dtype == np.uint8 else "minisblack"
    memmap = tifffile.memmap(str(path), shape=shape, dtype=np_dtype, photometric=photometric)
    rng = np.random.default_rng(seed)

    x_coords = np.arange(width).reshape(1, -1)
    for y0 in range(0, height, _STRIPE_ROWS):
        h = min(_STRIPE_ROWS, height - y0)
        y_coords = np.arange(y0, y0 + h).reshape(-1, 1)
        pattern = (np.sin(y_coords / 97.0) + np.cos(x_coords / 131.0)) * 0.5 + 0.5  # dans [0, 1]
        noise = rng.random((h, width)) * 0.1
        stripe = np.clip(pattern + noise, 0.0, 1.0) * max_value

        if bands == 1:
            memmap[y0 : y0 + h, :] = stripe.astype(np_dtype)
        else:
            for b in range(bands):
                shifted = np.roll(stripe, shift=b * 37, axis=1)
                memmap[y0 : y0 + h, :, b] = shifted.astype(np_dtype)

    memmap.flush()
    print(f"Écrit [{path}] : {width}x{height}, {bands} bande(s), {np_dtype}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Fichier TIFF de sortie")
    parser.add_argument("--width", type=int, default=8000)
    parser.add_argument("--height", type=int, default=8000)
    parser.add_argument("--bands", type=int, default=3, choices=(1, 2, 3, 4))
    parser.add_argument("--dtype", type=str, default="uint16")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    """Point d'entrée CLI."""
    args = _parse_args()
    generate_synthetic_tiff(args.output, args.width, args.height, args.bands, args.dtype, args.seed)


if __name__ == "__main__":
    main()
