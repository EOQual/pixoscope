"""Tests de sélection du niveau de pyramide."""

from __future__ import annotations

import pytest

from pixoscope.core.pyramid import level_for_zoom
from pixoscope.io.backend_base import PyramidLevelInfo

_LEVELS = [
    PyramidLevelInfo(0, 6000, 6000),
    PyramidLevelInfo(1, 3000, 3000),
    PyramidLevelInfo(2, 1500, 1500),
    PyramidLevelInfo(3, 750, 750),
]


@pytest.mark.parametrize(
    ("zoom", "expected_level"),
    [
        (2.0, 0),
        (1.0, 0),
        (0.5, 1),
        (0.24, 2),
        (0.1, 3),
        (0.001, 3),  # ne descend jamais sous le niveau le moins résolu
    ],
)
def test_level_for_zoom(zoom: float, expected_level: int) -> None:
    assert level_for_zoom(_LEVELS, zoom) == expected_level


def test_level_for_zoom_single_level() -> None:
    assert level_for_zoom([PyramidLevelInfo(0, 100, 100)], 0.01) == 0


def test_level_for_zoom_rejects_empty_list() -> None:
    with pytest.raises(ValueError):
        level_for_zoom([], 1.0)
