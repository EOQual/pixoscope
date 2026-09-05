"""Tests des statistiques échantillonnées."""

from __future__ import annotations

import numpy as np

from pixoscope.core.stats import auto_stretch_range, compute_band_stats


def test_compute_band_stats_basic() -> None:
    array = np.arange(256, dtype=np.uint8).reshape(16, 16)
    stats = compute_band_stats(array)
    assert stats.minimum == 0.0
    assert stats.maximum == 255.0
    assert stats.sample_shape == (16, 16)
    assert stats.histogram_counts.sum() == array.size


def test_compute_band_stats_ignores_nan_for_float() -> None:
    array = np.full((4, 4), np.nan, dtype=np.float32)
    array[0, 0] = 1.0
    array[0, 1] = 3.0
    stats = compute_band_stats(array)
    assert stats.minimum == 1.0
    assert stats.maximum == 3.0


def test_compute_band_stats_empty_after_nan_filtering() -> None:
    array = np.full((3, 3), np.nan, dtype=np.float32)
    stats = compute_band_stats(array)
    assert stats.minimum == stats.maximum == 0.0


def test_auto_stretch_range_uses_percentiles() -> None:
    array = np.arange(100, dtype=np.float64).reshape(10, 10)
    stats = compute_band_stats(array)
    vmin, vmax = auto_stretch_range(stats, low=2, high=98)
    assert stats.minimum <= vmin < vmax <= stats.maximum
