"""Tests du modèle de canaux et de la composition RVB/gris."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pixoscope.core.image_model import ChannelMapping, ImageDataset, default_channel_mapping


def test_default_mapping_grayscale_for_single_band() -> None:
    mapping = default_channel_mapping(1)
    assert mapping.is_grayscale
    assert mapping.gray == 0


def test_default_mapping_rgb_for_three_or_more_bands() -> None:
    mapping = default_channel_mapping(5)
    assert not mapping.is_grayscale
    assert (mapping.red, mapping.green, mapping.blue) == (0, 1, 2)


def test_band_indices_grayscale() -> None:
    assert ChannelMapping(gray=2).band_indices() == [2]


def test_band_indices_rgb_deduplicates_and_sorts() -> None:
    assert ChannelMapping(red=3, green=1, blue=3).band_indices() == [1, 3]


def test_read_display_tile_uses_selected_bands(pyramidal_tiff_path: Path) -> None:
    dataset = ImageDataset.open(pyramidal_tiff_path)
    try:
        # Bascule sur une bande unique en niveaux de gris : la tuile doit
        # être 2D, pas 3D.
        dataset.set_channel_mapping(ChannelMapping(gray=1))
        tile = dataset.read_display_tile(0, 0, 0, 8, 8)
        assert tile.ndim == 2
        assert tile.dtype == np.uint8

        dataset.set_channel_mapping(ChannelMapping(red=2, green=0, blue=1))
        tile_rgb = dataset.read_display_tile(0, 0, 0, 8, 8)
        assert tile_rgb.shape == (8, 8, 3)
    finally:
        dataset.close()


def test_manual_display_range_survives_new_stats(pyramidal_tiff_path: Path) -> None:
    from pixoscope.core.stats import compute_band_stats

    dataset = ImageDataset.open(pyramidal_tiff_path)
    try:
        dataset.set_display_range(0, 10.0, 200.0)
        stats = compute_band_stats(np.arange(256, dtype=np.uint8).reshape(16, 16))
        dataset.set_band_stats(0, stats)
        assert dataset.display_range(0) == (10.0, 200.0)
    finally:
        dataset.close()
