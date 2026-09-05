"""Tests de fumée de l'IHM — instanciation et ouverture de fichier.

Exécutés avec ``QT_QPA_PLATFORM=offscreen`` (voir ``conftest.py``) : pas
de rendu visuel réel, seulement une vérification que les composants
s'assemblent et que le chargement asynchrone aboutit.
"""

from __future__ import annotations

from pathlib import Path

from pixoscope.ui.main_window import MainWindow


def test_main_window_opens_single_image(qtbot, png_path: Path) -> None:
    window = MainWindow()
    qtbot.addWidget(window)

    window.open_paths([str(png_path)])
    qtbot.waitUntil(lambda: window._panes[0].dataset is not None, timeout=5000)

    dataset = window._panes[0].dataset
    assert dataset is not None
    assert dataset.shape == (48, 64)

    qtbot.waitUntil(lambda: len(dataset.stats_cache) == len(dataset.bands), timeout=5000)


def test_main_window_compare_mode(qtbot, png_path: Path, pyramidal_tiff_path: Path) -> None:
    window = MainWindow()
    qtbot.addWidget(window)

    window.open_paths([str(png_path), str(pyramidal_tiff_path)])
    assert len(window._panes) == 2
    assert window._compare_action.isChecked()

    qtbot.waitUntil(lambda: window._panes[0].dataset is not None, timeout=5000)
    qtbot.waitUntil(lambda: window._panes[1].dataset is not None, timeout=5000)

    window._link_action.setChecked(True)
    assert set(window._link_unsubscribers) == {"pane-0", "pane-1"}

    window._compare_action.setChecked(False)
    assert len(window._panes) == 1


def test_channel_mapping_change_triggers_refresh(qtbot, pyramidal_tiff_path: Path) -> None:
    from pixoscope.core.image_model import ChannelMapping

    window = MainWindow()
    qtbot.addWidget(window)

    window.open_paths([str(pyramidal_tiff_path)])
    qtbot.waitUntil(lambda: window._panes[0].dataset is not None, timeout=5000)

    dataset = window._panes[0].dataset
    window._on_mapping_changed(ChannelMapping(gray=0))
    assert dataset.channel_mapping.is_grayscale
