#!/usr/bin/env python3
"""Compare le coût d'ouverture "naïf" (image entière) vs pixoscope.

Démontre le gain de la stratégie de lecture fenêtrée/pyramidale
(voir ``REFERENCE_TECHNIQUE.md`` §2) : temps jusqu'au premier affichage
et pic mémoire, pour lire une petite fenêtre d'un gros fichier.

Exemples
--------
.. code-block:: bash

    # Génère d'abord un fichier de test (voir make_synthetic_bigtiff.py)
    python scripts/make_synthetic_bigtiff.py /tmp/big.tif --width 20000 --height 20000

    python scripts/bench_open.py /tmp/big.tif --backend naive
    python scripts/bench_open.py /tmp/big.tif --backend pixoscope
    python scripts/bench_open.py /tmp/big.tif --backend pixoscope --pixoscope-backend gdal
"""

from __future__ import annotations

import argparse
import os
import tracemalloc
from pathlib import Path
from time import perf_counter


def _bench_naive(path: Path, window: int) -> None:
    """Lit l'image entière avant de découper la fenêtre demandée en mémoire.

    Base de comparaison "naïve" (équivalent, pour un TIFF, à un
    ``gdal.Dataset.ReadAsArray()`` sans fenêtrage) : lit l'image entière
    avec ``tifffile.imread``, puis découpe la fenêtre demandée.
    """
    import tifffile

    t0 = perf_counter()
    full = tifffile.imread(str(path))
    tile = full[:window, :window]
    elapsed = perf_counter() - t0
    print(f"[naif]      temps={elapsed:.3f}s  tuile={tile.shape}  image_complete={full.shape} ({full.nbytes / 1e6:.1f} Mo)")


def _bench_pixoscope(path: Path, window: int) -> None:
    from pixoscope.core.image_model import ImageDataset

    t0 = perf_counter()
    dataset = ImageDataset.open(path)
    tile = dataset.read_display_tile(0, 0, 0, window, window)
    elapsed = perf_counter() - t0
    print(f"[pixoscope] temps={elapsed:.3f}s  tuile={tile.shape}  niveaux_pyramide={len(dataset.levels)}")
    dataset.close()


def main() -> None:
    """Point d'entrée CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--backend", choices=("naive", "pixoscope"), default="pixoscope")
    parser.add_argument("--pixoscope-backend", default=None, help="Force PIXOSCOPE_BACKEND (ex. gdal, tifffile)")
    parser.add_argument("--window", type=int, default=512, help="Taille de la fenêtre lue, en pixels")
    args = parser.parse_args()

    if args.pixoscope_backend:
        os.environ["PIXOSCOPE_BACKEND"] = args.pixoscope_backend

    tracemalloc.start()
    if args.backend == "naive":
        _bench_naive(args.path, args.window)
    else:
        _bench_pixoscope(args.path, args.window)
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"pic mémoire (Python tracé) : {peak / 1e6:.1f} Mo")


if __name__ == "__main__":
    main()
