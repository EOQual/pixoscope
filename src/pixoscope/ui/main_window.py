"""Fenêtre principale — orchestration fine, aucune logique métier.

Toute la logique (lecture, modèle de canaux, statistiques, algorithmes)
vit dans ``pixoscope.core``/``pixoscope.io``/``pixoscope.processing`` ;
``MainWindow`` ne fait que connecter les signaux entre les panneaux
(:class:`~pixoscope.ui.view_pane.ViewPane`,
:class:`~pixoscope.ui.channel_panel.ChannelPanel`,
:class:`~pixoscope.ui.stats_panel.StatsPanel`,
:class:`~pixoscope.ui.processing_panel.ProcessingPanel`).
"""

from __future__ import annotations

from collections.abc import Callable

from loguru import logger
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QActionGroup, QIcon
from PySide6.QtWidgets import QComboBox, QDoubleSpinBox, QLabel, QMainWindow, QSplitter, QTabWidget, QToolBar, QWidget

from pixoscope.core.image_model import ChannelMapping, ImageDataset
from pixoscope.core.stats import BandStats, compute_band_stats
from pixoscope.core.viewport import ViewportLinker
from pixoscope.ui.channel_panel import ChannelPanel
from pixoscope.ui.graphics_view import InterpolationMode
from pixoscope.ui.load_worker import run_in_background
from pixoscope.ui.processing_panel import ProcessingPanel
from pixoscope.ui.qt_utils import icon_path
from pixoscope.ui.stats_panel import StatsPanel
from pixoscope.ui.view_pane import ViewPane

_INTERPOLATION_LABELS = {
    InterpolationMode.NEAREST: "Plus proche voisin",
    InterpolationMode.BILINEAR: "Bilinéaire",
    InterpolationMode.BICUBIC: "Bicubique",
}


def _compute_all_band_stats(dataset: ImageDataset) -> dict[int, BandStats]:
    """Calcule les statistiques de toutes les bandes sur un aperçu basse résolution.

    Exécuté dans un thread de fond (voir :class:`~pixoscope.ui.load_worker.CallableWorker`)
    — jamais sur le thread UI, et jamais sur la pleine résolution.

    Parameters
    ----------
    dataset : ImageDataset

    Returns
    -------
    dict of int to BandStats
    """
    overview = dataset.handle.read_overview()
    results: dict[int, BandStats] = {}
    for band in dataset.bands:
        plane = overview if overview.ndim == 2 else overview[..., band.index]
        results[band.index] = compute_band_stats(plane)
    return results


class MainWindow(QMainWindow):
    """Fenêtre principale de Pixoscope.

    Parameters
    ----------
    parent : PySide6.QtWidgets.QWidget, optional
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Pixoscope")
        self.resize(1280, 840)

        self._viewport_linker = ViewportLinker()
        self._link_unsubscribers: dict[str, Callable[[], None]] = {}
        self._panes: list[ViewPane] = []
        self._active_pane: ViewPane | None = None
        #: Outil d'interaction courant, appliqué à toutes les vues (pas
        #: seulement la vue active) : True = zoom rectangle, False = main.
        self._rubber_band_mode = False
        #: Réglages globaux appliqués à toutes les vues, y compris les
        #: nouvelles (mode comparaison) — voir _build_toolbar.
        self._zoom_sensitivity = 1.0
        self._interpolation_mode = InterpolationMode.BILINEAR

        self._pane_splitter = QSplitter(Qt.Orientation.Horizontal)
        self._add_pane()

        self._channel_panel = ChannelPanel()
        self._channel_panel.mapping_changed.connect(self._on_mapping_changed)
        self._stats_panel = StatsPanel()
        self._processing_panel = ProcessingPanel()
        self._processing_panel.processing_changed.connect(self._on_processing_changed)

        self._side_tabs = QTabWidget()
        self._side_tabs.addTab(self._channel_panel, "Canaux")
        self._side_tabs.addTab(self._stats_panel, "Statistiques")
        self._side_tabs.addTab(self._processing_panel, "Rehaussement")
        self._side_tabs.setMaximumWidth(340)

        central = QSplitter(Qt.Orientation.Horizontal)
        central.addWidget(self._pane_splitter)
        central.addWidget(self._side_tabs)
        central.setStretchFactor(0, 1)
        central.setStretchFactor(1, 0)
        self.setCentralWidget(central)

        self._build_toolbar()
        self._refresh_side_panels()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def _build_toolbar(self) -> None:
        toolbar = QToolBar("Principal", self)
        self.addToolBar(toolbar)

        open_action = QAction(QIcon(icon_path("open_folder.svg")), "Ouvrir…", self)
        open_action.triggered.connect(self._on_open_active)
        toolbar.addAction(open_action)

        home_action = QAction(QIcon(icon_path("home.svg")), "Ajuster", self)
        home_action.triggered.connect(self._on_fit_active)
        toolbar.addAction(home_action)

        toolbar.addSeparator()

        # Outils d'interaction mutuellement exclusifs : main (pan, par
        # défaut) ou zoom rectangle (cliquer-glisser pour zoomer sur une
        # zone) — appliqués à toutes les vues, pas seulement l'active.
        tool_group = QActionGroup(self)
        tool_group.setExclusive(True)

        self._pan_action = QAction(QIcon(icon_path("move.svg")), "Déplacer", self)
        self._pan_action.setCheckable(True)
        self._pan_action.setChecked(True)
        tool_group.addAction(self._pan_action)
        toolbar.addAction(self._pan_action)

        self._zoom_rect_action = QAction(QIcon(icon_path("search.svg")), "Zoom rectangle", self)
        self._zoom_rect_action.setCheckable(True)
        self._zoom_rect_action.toggled.connect(self._on_zoom_rect_toggled)
        tool_group.addAction(self._zoom_rect_action)
        toolbar.addAction(self._zoom_rect_action)

        toolbar.addSeparator()

        # Sensibilité du zoom molette/trackpad : un trackpad Mac envoie
        # des évènements bien plus fins qu'une souris, ce que le facteur
        # proportionnel de TiledImageView.wheelEvent absorbe déjà en
        # grande partie — ce réglage permet en plus à l'utilisateur
        # d'ajuster la réactivité globale selon son matériel/goût.
        toolbar.addWidget(QLabel(" Sensibilité zoom : "))
        self._sensitivity_spin = QDoubleSpinBox()
        self._sensitivity_spin.setRange(0.2, 3.0)
        self._sensitivity_spin.setSingleStep(0.1)
        self._sensitivity_spin.setDecimals(1)
        self._sensitivity_spin.setValue(self._zoom_sensitivity)
        self._sensitivity_spin.setToolTip("Sensibilité du zoom à la molette/trackpad")
        self._sensitivity_spin.valueChanged.connect(self._on_zoom_sensitivity_changed)
        toolbar.addWidget(self._sensitivity_spin)

        # Méthode de ré-échantillonnage à l'affichage zoomé.
        toolbar.addWidget(QLabel("  Interpolation : "))
        self._interp_combo = QComboBox()
        for mode in InterpolationMode:
            self._interp_combo.addItem(_INTERPOLATION_LABELS[mode], mode)
        self._interp_combo.setCurrentIndex(self._interp_combo.findData(self._interpolation_mode))
        self._interp_combo.currentIndexChanged.connect(self._on_interpolation_changed)
        toolbar.addWidget(self._interp_combo)

        toolbar.addSeparator()

        self._compare_action = QAction("Comparer", self)
        self._compare_action.setCheckable(True)
        self._compare_action.toggled.connect(self._on_compare_toggled)
        toolbar.addAction(self._compare_action)

        self._link_action = QAction("Lier les vues", self)
        self._link_action.setCheckable(True)
        self._link_action.setEnabled(False)
        self._link_action.toggled.connect(self._on_link_toggled)
        toolbar.addAction(self._link_action)

        toolbar.addSeparator()

        self._panel_action = QAction("Panneau", self)
        self._panel_action.setCheckable(True)
        self._panel_action.setChecked(True)
        self._panel_action.toggled.connect(self._side_tabs.setVisible)
        toolbar.addAction(self._panel_action)

    def _add_pane(self) -> ViewPane:
        pane = ViewPane(f"pane-{len(self._panes)}")
        pane.dataset_opened.connect(lambda dataset, p=pane: self._on_dataset_opened(p, dataset))
        pane.activated.connect(lambda p=pane: self._set_active_pane(p))
        pane.view.viewport_changed.connect(lambda state, p=pane: self._viewport_linker.publish(p.pane_id, state))
        pane.view.set_rubber_band_zoom_enabled(self._rubber_band_mode)
        pane.view.set_zoom_sensitivity(self._zoom_sensitivity)
        pane.view.set_interpolation_mode(self._interpolation_mode)
        self._panes.append(pane)
        self._pane_splitter.addWidget(pane)
        if self._active_pane is None:
            self._active_pane = pane
        self._update_pane_highlights()
        return pane

    def _on_zoom_rect_toggled(self, checked: bool) -> None:
        self._rubber_band_mode = checked
        for pane in self._panes:
            pane.view.set_rubber_band_zoom_enabled(checked)

    def _on_zoom_sensitivity_changed(self, value: float) -> None:
        self._zoom_sensitivity = value
        for pane in self._panes:
            pane.view.set_zoom_sensitivity(value)

    def _on_interpolation_changed(self, index: int) -> None:
        mode = self._interp_combo.itemData(index)
        self._interpolation_mode = mode
        for pane in self._panes:
            pane.view.set_interpolation_mode(mode)

    # ------------------------------------------------------------------
    # API publique
    # ------------------------------------------------------------------
    def open_paths(self, paths: list[str]) -> None:
        """Ouvre une ou deux images (typiquement les arguments de la CLI).

        Un seul chemin ouvre le panneau simple ; deux chemins activent
        automatiquement le mode comparaison, un fichier par panneau.

        Parameters
        ----------
        paths : list of str
        """
        if not paths:
            return
        self._panes[0].open_file(paths[0])
        if len(paths) > 1:
            if len(self._panes) == 1:
                self._compare_action.setChecked(True)  # déclenche l'ajout du second panneau
            self._panes[1].open_file(paths[1])
            if len(paths) > 2:
                logger.warning(
                    f"{len(paths)} fichiers fournis, seuls les 2 premiers sont ouverts (mode comparaison)"
                )

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def _on_open_active(self) -> None:
        if self._active_pane is not None:
            self._active_pane.open_dialog()

    def _on_fit_active(self) -> None:
        if self._active_pane is not None:
            self._active_pane.view.fit_to_view()

    def _on_compare_toggled(self, checked: bool) -> None:
        if checked and len(self._panes) == 1:
            self._add_pane()
            self._link_action.setEnabled(True)
        elif not checked and len(self._panes) == 2:
            pane = self._panes.pop()
            self._link_action.setChecked(False)
            self._link_action.setEnabled(False)
            self._unsubscribe_link(pane.pane_id)
            pane.setParent(None)
            if pane.dataset is not None:
                pane.dataset.close()
            if self._active_pane is pane:
                self._active_pane = self._panes[0]
                self._refresh_side_panels()
            else:
                self._update_pane_highlights()  # retour à un seul panneau : plus de bordure à afficher
            pane.deleteLater()

    def _on_link_toggled(self, checked: bool) -> None:
        if checked:
            for pane in self._panes:
                self._link_unsubscribers[pane.pane_id] = self._viewport_linker.subscribe(
                    pane.pane_id,
                    lambda state, p=pane: p.view.set_viewport(state),  # type: ignore[misc]
                )
        else:
            for pane in self._panes:
                self._unsubscribe_link(pane.pane_id)

    def _unsubscribe_link(self, pane_id: str) -> None:
        unsubscribe = self._link_unsubscribers.pop(pane_id, None)
        if unsubscribe is not None:
            unsubscribe()

    def _set_active_pane(self, pane: ViewPane) -> None:
        self._active_pane = pane
        self._refresh_side_panels()

    # ------------------------------------------------------------------
    # Réactions aux événements des panneaux
    # ------------------------------------------------------------------
    def _on_dataset_opened(self, pane: ViewPane, dataset: ImageDataset) -> None:
        if pane is self._active_pane:
            self._refresh_side_panels()

        run_in_background(
            _compute_all_band_stats,
            dataset,
            on_result=lambda stats, d=dataset, p=pane: self._on_stats_ready(p, d, stats),  # type: ignore[misc]
            on_error=lambda msg: logger.warning(f"Échec du calcul de statistiques : {msg}"),
        )

    def _on_stats_ready(self, pane: ViewPane, dataset: ImageDataset, stats: dict[int, BandStats]) -> None:
        for band_index, band_stats in stats.items():
            dataset.set_band_stats(band_index, band_stats)
        pane.view.request_refresh()  # l'étirement auto vient de changer
        if pane is self._active_pane:
            self._stats_panel.refresh(dataset)

    def _on_mapping_changed(self, mapping: ChannelMapping) -> None:
        pane = self._active_pane
        if pane is None or pane.dataset is None:
            return
        pane.dataset.set_channel_mapping(mapping)
        pane.view.request_refresh()
        self._stats_panel.refresh(pane.dataset, mapping)

    def _on_processing_changed(self, key: object) -> None:
        pane = self._active_pane
        if pane is None or pane.dataset is None:
            return
        pane.dataset.set_processing(key)  # type: ignore[arg-type]
        pane.view.request_refresh()

    def _refresh_side_panels(self) -> None:
        pane = self._active_pane
        dataset = pane.dataset if pane is not None else None
        if dataset is not None:
            self._channel_panel.set_bands(dataset.bands, dataset.channel_mapping)
        else:
            self._channel_panel.set_bands([], ChannelMapping())
        self._stats_panel.refresh(dataset)
        self._update_pane_highlights()

    def _update_pane_highlights(self) -> None:
        """Marque visuellement quel panneau est la cible des onglets latéraux.

        Seulement pertinent à partir de deux panneaux (mode comparaison) :
        avec un seul panneau, il n'y a aucune ambiguïté à lever.
        """
        multiple = len(self._panes) > 1
        for pane in self._panes:
            pane.set_active(multiple and pane is self._active_pane)
