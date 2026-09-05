"""Fixtures partagées — génère toutes les images de test à la volée.

Volontairement autonome : ne dépend d'aucun fichier externe, pour que
les tests fonctionnent sur un simple `git clone` de Pixoscope (CI
incluse).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pytest
import tifffile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@dataclass(frozen=True)
class LumFixture:
    """Fichier ``.lum`` de test, avec les données attendues pour comparaison."""

    path: Path
    data: np.ndarray


@pytest.fixture
def small_rgb_array() -> np.ndarray:
    """Petit tableau RVB ``uint8`` synthétique, reproductible."""
    rng = np.random.default_rng(42)
    return (rng.random((48, 64, 3)) * 255).astype(np.uint8)


@pytest.fixture
def small_gray_array() -> np.ndarray:
    """Petit tableau niveaux de gris ``uint8`` synthétique, reproductible."""
    rng = np.random.default_rng(42)
    return (rng.random((48, 64)) * 255).astype(np.uint8)


@pytest.fixture
def png_path(tmp_path: Path, small_rgb_array: np.ndarray) -> Path:
    """Fichier PNG RVB de test."""
    path = tmp_path / "test.png"
    iio.imwrite(path, small_rgb_array)
    return path


@pytest.fixture
def pyramidal_tiff_path(tmp_path: Path) -> Path:
    """Fichier TIFF multi-bandes assez grand pour déclencher une pyramide en cache.

    2200 px de côté dépasse le seuil (2048) sans nécessiter d'écrire un
    fichier volumineux — le test reste rapide.
    """
    path = tmp_path / "big.tif"
    size = 2200
    rng = np.random.default_rng(7)
    data = (rng.random((size, size, 3)) * 255).astype(np.uint8)
    tifffile.imwrite(path, data, photometric="rgb")
    return path


@pytest.fixture
def lum_fixture(tmp_path: Path) -> LumFixture:
    """Fichier ``.lum`` de test (16 bits, little-endian) et ses données attendues."""
    import struct

    path = tmp_path / "test.lum"
    n_cols, n_rows = 16, 10
    row_offset = n_cols * 2
    rng = np.random.default_rng(3)
    data = (rng.random((n_rows, n_cols)) * 60000).astype(np.uint16)

    with open(path, "wb") as handle:
        handle.write(struct.pack("<I", n_cols))
        handle.write(struct.pack("<I", n_rows))
        handle.write(b"16LI")
        handle.write(b"\x00" * (row_offset - 12))
        handle.write(data.tobytes())

    return LumFixture(path=path, data=data)
