"""Tests des backends de lecture et de la sélection automatique."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pixoscope.io import open_image
from pixoscope.io.pyramid_builder import PyramidCacheHandle
from pixoscope.plugins.lum.lum_object import LumReader


def test_open_png_via_imageio(png_path: Path, small_rgb_array: np.ndarray) -> None:
    handle = open_image(png_path)
    try:
        assert (handle.height, handle.width) == small_rgb_array.shape[:2]
        assert handle.n_bands == 3
        region = handle.read_region(0, 2, 3, 10, 8)
        assert region.shape == (8, 10, 3)
        np.testing.assert_array_equal(region, small_rgb_array[3:11, 2:12])
    finally:
        handle.close()


def test_open_pyramidal_tiff_builds_cache(pyramidal_tiff_path: Path) -> None:
    handle = open_image(pyramidal_tiff_path)
    try:
        assert isinstance(handle, PyramidCacheHandle)
        assert len(handle.levels) > 1
        # Le niveau 0 doit rester délégué à la source, pixel pour pixel.
        top_left = handle.read_region(0, 0, 0, 4, 4)
        assert top_left.shape == (4, 4, 3)

        # Ré-ouverture : le cache doit être détecté et réutilisé (pas de
        # reconstruction, donc rapide et sans nouvel appel à tifffile.memmap
        # en écriture).
        handle2 = open_image(pyramidal_tiff_path)
        try:
            assert isinstance(handle2, PyramidCacheHandle)
            assert handle2.levels == handle.levels
        finally:
            handle2.close()
    finally:
        handle.close()


def test_lum_reader_round_trip(lum_fixture) -> None:
    reader = LumReader(lum_fixture.path)
    assert reader.n_cols == lum_fixture.data.shape[1]
    assert reader.n_rows == lum_fixture.data.shape[0]
    np.testing.assert_array_equal(reader.read_all(), lum_fixture.data)
    np.testing.assert_array_equal(reader.read_rows(2, 3), lum_fixture.data[2:5])


def test_open_lum_via_registry(lum_fixture) -> None:
    handle = open_image(lum_fixture.path)
    try:
        assert handle.n_bands == 1
        assert (handle.height, handle.width) == lum_fixture.data.shape
        region = handle.read_region(0, 1, 1, 5, 3)
        np.testing.assert_array_equal(region, lum_fixture.data[1:4, 1:6])
    finally:
        handle.close()
