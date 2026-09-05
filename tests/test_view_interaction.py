"""Tests des interactions de la vue : molette/zoom, zoom rectangle, panneau.

Fait suite à un retour utilisateur (macOS) : le zoom molette ne
fonctionnait pas correctement sur trackpad, il n'y avait pas d'outil de
zoom par sélection de zone, et le panneau latéral ne pouvait pas être
replié pour laisser toute la place à l'image. Un second retour a ensuite
signalé un zoom trop sensible sur trackpad (sans réglage possible) et
l'absence de choix d'interpolateur.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import QGraphicsView

from pixoscope.ui.graphics_view import InterpolationMode
from pixoscope.ui.main_window import MainWindow


def _make_wheel_event(view, delta_y: int, modifiers: Qt.KeyboardModifier) -> QWheelEvent:
    center = QPointF(view.viewport().rect().center())
    return QWheelEvent(
        center,
        view.mapToGlobal(center.toPoint()).toPointF(),
        QPoint(0, 0),  # pixelDelta
        QPoint(0, delta_y),  # angleDelta
        Qt.MouseButton.NoButton,
        modifiers,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )


@pytest.fixture
def opened_window(qtbot, pyramidal_tiff_path: Path) -> MainWindow:
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    window.open_paths([str(pyramidal_tiff_path)])
    qtbot.waitUntil(lambda: window._panes[0].dataset is not None, timeout=5000)
    dataset = window._panes[0].dataset
    # Attendre la fin du calcul de statistiques en arrière-plan avant de
    # rendre la main au test, pour ne pas laisser un worker en vol
    # toucher une fenêtre déjà détruite au test suivant.
    qtbot.waitUntil(lambda: len(dataset.stats_cache) == len(dataset.bands), timeout=5000)
    return window


def test_wheel_zooms_in_and_out(opened_window: MainWindow) -> None:
    """Convention reprise de QGIS : la molette seule zoome (pas de modificateur requis).

    Le modificateur Ctrl a été abandonné : sur macOS, Ctrl+molette/pincement
    est souvent intercepté par la fonction d'accessibilité "Zoom" du
    système avant d'atteindre l'application.
    """
    view = opened_window._panes[0].view
    scale_before = view.transform().m11()

    view.wheelEvent(_make_wheel_event(view, delta_y=120, modifiers=Qt.KeyboardModifier.NoModifier))
    scale_after_in = view.transform().m11()
    assert scale_after_in > scale_before

    view.wheelEvent(_make_wheel_event(view, delta_y=-120, modifiers=Qt.KeyboardModifier.NoModifier))
    scale_after_out = view.transform().m11()
    assert scale_after_out < scale_after_in


def test_wheel_zoom_falls_back_to_pixel_delta(opened_window: MainWindow) -> None:
    """Trackpad macOS : angleDelta peut être nul, seul pixelDelta est renseigné."""
    view = opened_window._panes[0].view
    scale_before = view.transform().m11()

    event = QWheelEvent(
        QPointF(view.viewport().rect().center()),
        view.mapToGlobal(view.viewport().rect().center()).toPointF(),
        QPoint(0, 120),  # pixelDelta seul renseigné
        QPoint(0, 0),  # angleDelta nul
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    view.wheelEvent(event)
    assert view.transform().m11() > scale_before


def test_zoom_rect_tool_toggles_drag_mode(opened_window: MainWindow) -> None:
    view = opened_window._panes[0].view
    assert view.dragMode() == QGraphicsView.DragMode.ScrollHandDrag

    opened_window._zoom_rect_action.setChecked(True)
    assert view.dragMode() == QGraphicsView.DragMode.RubberBandDrag

    opened_window._pan_action.setChecked(True)
    assert view.dragMode() == QGraphicsView.DragMode.ScrollHandDrag


def test_zoom_rect_mode_applies_to_new_panes(opened_window: MainWindow) -> None:
    opened_window._zoom_rect_action.setChecked(True)
    opened_window._compare_action.setChecked(True)  # ajoute un second panneau
    assert len(opened_window._panes) == 2
    assert opened_window._panes[1].view.dragMode() == QGraphicsView.DragMode.RubberBandDrag


def test_rubber_band_selection_zooms_to_rect(qtbot, opened_window: MainWindow) -> None:
    """Glisser-déposer réel en mode "zoom rectangle" (pas d'appel direct à une méthode privée)."""
    view = opened_window._panes[0].view
    view.fit_to_view()
    scale_before = view.transform().m11()

    opened_window._zoom_rect_action.setChecked(True)
    viewport = view.viewport()
    start = viewport.rect().center() - QPoint(60, 60)
    end = viewport.rect().center() + QPoint(60, 60)

    midpoint = QPoint((start.x() + end.x()) // 2, (start.y() + end.y()) // 2)
    qtbot.mousePress(viewport, Qt.MouseButton.LeftButton, pos=start)
    qtbot.mouseMove(viewport, pos=midpoint)
    qtbot.mouseMove(viewport, pos=end)
    qtbot.mouseRelease(viewport, Qt.MouseButton.LeftButton, pos=end)

    assert view.transform().m11() > scale_before


def test_rubber_band_ignores_a_simple_click(qtbot, opened_window: MainWindow) -> None:
    """Un simple clic (sans glisser) ne doit pas déclencher de zoom."""
    view = opened_window._panes[0].view
    view.fit_to_view()
    scale_before = view.transform().m11()

    opened_window._zoom_rect_action.setChecked(True)
    viewport = view.viewport()
    pos = viewport.rect().center()

    qtbot.mousePress(viewport, Qt.MouseButton.LeftButton, pos=pos)
    qtbot.mouseRelease(viewport, Qt.MouseButton.LeftButton, pos=pos)

    assert view.transform().m11() == pytest.approx(scale_before)


def test_panel_toggle_hides_and_shows_side_tabs(opened_window: MainWindow) -> None:
    assert opened_window._side_tabs.isVisible()
    opened_window._panel_action.setChecked(False)
    assert not opened_window._side_tabs.isVisible()
    opened_window._panel_action.setChecked(True)
    assert opened_window._side_tabs.isVisible()


def test_navigator_hidden_when_full_image_visible(qtbot, opened_window: MainWindow) -> None:
    pane = opened_window._panes[0]
    pane.view.fit_to_view()
    qtbot.waitUntil(lambda: not pane.view.navigator.isVisible(), timeout=2000)


def test_navigator_shown_when_zoomed_in(qtbot, opened_window: MainWindow) -> None:
    pane = opened_window._panes[0]
    pane.view.fit_to_view()
    pane.view.scale(4.0, 4.0)
    pane.view._emit_viewport_state()  # déclenche la mise à jour du navigateur
    qtbot.waitUntil(lambda: pane.view.navigator.isVisible(), timeout=2000)


def test_navigator_click_recenters_main_view(qtbot, opened_window: MainWindow) -> None:
    pane = opened_window._panes[0]
    height, width = pane.dataset.shape

    # Zoomer pour que le navigateur ait une image de référence et qu'un
    # déplacement soit possible/mesurable.
    pane.view.fit_to_view()
    pane.view.scale(6.0, 6.0)
    qtbot.waitUntil(lambda: pane.view.navigator.isVisible(), timeout=2000)

    center_before = pane.view.mapToScene(pane.view.viewport().rect().center())
    pane._on_navigate_to(width * 0.9, height * 0.9)
    center_after = pane.view.mapToScene(pane.view.viewport().rect().center())

    assert (center_after.x(), center_after.y()) != (center_before.x(), center_before.y())


def test_wheel_zoom_is_proportional_to_delta(opened_window: MainWindow) -> None:
    """Un petit delta (trackpad) doit zoomer beaucoup moins qu'un grand (molette).

    Avant correction, un facteur fixe (1.25) était appliqué quel que soit
    le delta, ce qui rendait le zoom bien trop sensible sur trackpad
    (beaucoup d'évènements à faible delta par geste). Le facteur observé
    est mesuré en relatif (échelle après / échelle avant) : l'échelle
    absolue de départ dépend du "fit to view" initial, pas de 1.0.
    """
    view = opened_window._panes[0].view

    scale_before = view.transform().m11()
    view.wheelEvent(_make_wheel_event(view, delta_y=10, modifiers=Qt.KeyboardModifier.NoModifier))
    small_factor = view.transform().m11() / scale_before

    scale_before = view.transform().m11()
    view.wheelEvent(_make_wheel_event(view, delta_y=120, modifiers=Qt.KeyboardModifier.NoModifier))
    large_factor = view.transform().m11() / scale_before

    assert 1.0 < small_factor < large_factor


def test_zoom_sensitivity_scales_the_wheel_factor(opened_window: MainWindow) -> None:
    view = opened_window._panes[0].view

    view.set_zoom_sensitivity(0.3)
    scale_before = view.transform().m11()
    view.wheelEvent(_make_wheel_event(view, delta_y=120, modifiers=Qt.KeyboardModifier.NoModifier))
    low_sensitivity_factor = view.transform().m11() / scale_before

    view.set_zoom_sensitivity(2.0)
    scale_before = view.transform().m11()
    view.wheelEvent(_make_wheel_event(view, delta_y=120, modifiers=Qt.KeyboardModifier.NoModifier))
    high_sensitivity_factor = view.transform().m11() / scale_before

    assert 1.0 < low_sensitivity_factor < high_sensitivity_factor


def test_zoom_sensitivity_spin_box_updates_all_panes(opened_window: MainWindow) -> None:
    opened_window._compare_action.setChecked(True)
    opened_window._sensitivity_spin.setValue(2.5)
    assert all(pytest.approx(p.view._zoom_sensitivity) == 2.5 for p in opened_window._panes)


def test_interpolation_combo_applies_to_all_panes(opened_window: MainWindow) -> None:
    opened_window._compare_action.setChecked(True)
    nearest_index = opened_window._interp_combo.findData(InterpolationMode.NEAREST)
    opened_window._interp_combo.setCurrentIndex(nearest_index)

    for pane in opened_window._panes:
        assert pane.view._interpolation is InterpolationMode.NEAREST
        assert pane.view._pixmap_item.transformationMode() == Qt.TransformationMode.FastTransformation


def test_bicubic_interpolation_resizes_the_displayed_array(qtbot, opened_window: MainWindow) -> None:
    """En mode bicubique, le tableau affiché est pré-redimensionné à la taille écran.

    Contrairement aux modes "plus proche voisin"/"bilinéaire" (où Qt fait
    la mise à l'échelle au rendu et le tableau reste à la résolution du
    niveau de pyramide choisi), le mode bicubique redimensionne le
    tableau lui-même (voir ``TiledImageView._resize_bicubic``) : la
    largeur du pixmap affiché doit donc suivre le zoom courant.
    """
    view = opened_window._panes[0].view
    bicubic_index = opened_window._interp_combo.findData(InterpolationMode.BICUBIC)
    opened_window._interp_combo.setCurrentIndex(bicubic_index)
    assert view._pixmap_item.transformationMode() == Qt.TransformationMode.FastTransformation

    view.fit_to_view()
    qtbot.wait(300)  # laisse le rafraîchissement différé (bicubique) s'exécuter
    width_at_fit = view._pixmap_item.pixmap().width()

    view.scale(6.0, 6.0)
    view._refresh()
    qtbot.waitUntil(lambda: view._pixmap_item.pixmap().width() > width_at_fit * 2, timeout=3000)


def test_clicking_inside_a_pane_activates_it(qtbot, opened_window: MainWindow) -> None:
    """Un clic dans l'image (pas seulement la barre d'outils) doit activer le panneau.

    Avant correction, ``ViewPane.mousePressEvent`` n'était jamais atteint
    par un clic dans la vue (un enfant de ``ViewPane``, qui reçoit
    l'évènement en premier) : en mode comparaison, il n'y avait donc
    aucun moyen de choisir quel panneau ciblent les onglets latéraux.
    """
    opened_window._compare_action.setChecked(True)
    left, right = opened_window._panes
    assert opened_window._active_pane is left

    qtbot.mouseClick(right.view.viewport(), Qt.MouseButton.LeftButton)
    assert opened_window._active_pane is right

    qtbot.mouseClick(left.view.viewport(), Qt.MouseButton.LeftButton)
    assert opened_window._active_pane is left


def test_active_pane_is_visually_highlighted_only_in_compare_mode(opened_window: MainWindow) -> None:
    left = opened_window._panes[0]
    assert left.styleSheet() == "ViewPane { border: 2px solid transparent; border-radius: 3px; }"

    opened_window._compare_action.setChecked(True)
    left, right = opened_window._panes
    assert "#4a90e2" in left.styleSheet()
    assert "transparent" in right.styleSheet()

    right.activated.emit()
    assert "transparent" in left.styleSheet()
    assert "#4a90e2" in right.styleSheet()

    opened_window._compare_action.setChecked(False)
    assert "transparent" in opened_window._panes[0].styleSheet()
