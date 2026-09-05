"""Panneau d'assignation des bandes physiques aux plans R/V/B affichés.

Répond au besoin "charger des images multi-canaux en choisissant les
canaux à afficher pour les plans RVB" : indépendant de tout backend de
lecture, ce panneau ne connaît que la liste de
:class:`~pixoscope.core.image_model.BandInfo` de l'``ImageDataset``
actif.
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QCheckBox, QComboBox, QFormLayout, QGroupBox, QWidget

from pixoscope.core.image_model import BandInfo, ChannelMapping

_NONE_LABEL = "(aucune)"


class ChannelPanel(QGroupBox):
    """Sélecteurs de bandes pour R, V, B (ou niveaux de gris).

    Signals
    -------
    mapping_changed(object)
        Émis avec le nouveau :class:`ChannelMapping` à chaque changement
        de sélection.
    """

    mapping_changed = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Canaux", parent)
        self._bands: list[BandInfo] = []
        self._updating = False

        self._grayscale_check = QCheckBox("Niveaux de gris")
        self._grayscale_check.toggled.connect(self._on_changed)

        self._red_combo = QComboBox()
        self._green_combo = QComboBox()
        self._blue_combo = QComboBox()
        self._gray_combo = QComboBox()
        for combo in (self._red_combo, self._green_combo, self._blue_combo, self._gray_combo):
            combo.currentIndexChanged.connect(self._on_changed)

        layout = QFormLayout(self)
        layout.addRow(self._grayscale_check)
        layout.addRow("Rouge", self._red_combo)
        layout.addRow("Vert", self._green_combo)
        layout.addRow("Bleu", self._blue_combo)
        layout.addRow("Gris", self._gray_combo)

    def set_bands(self, bands: list[BandInfo], mapping: ChannelMapping) -> None:
        """Recharge la liste de bandes disponibles et l'état courant.

        Parameters
        ----------
        bands : list of BandInfo
        mapping : ChannelMapping
        """
        self._updating = True
        try:
            self._bands = bands
            for combo in (self._red_combo, self._green_combo, self._blue_combo, self._gray_combo):
                combo.clear()
                combo.addItem(_NONE_LABEL, None)
                for band in bands:
                    combo.addItem(band.name, band.index)

            self._grayscale_check.setChecked(mapping.is_grayscale)
            self._set_combo_value(self._red_combo, mapping.red)
            self._set_combo_value(self._green_combo, mapping.green)
            self._set_combo_value(self._blue_combo, mapping.blue)
            self._set_combo_value(self._gray_combo, mapping.gray if mapping.gray is not None else 0)
        finally:
            self._updating = False
        self._update_enabled_state()

    def _set_combo_value(self, combo: QComboBox, band_index: int | None) -> None:
        target = combo.findData(band_index)
        combo.setCurrentIndex(max(target, 0))

    def _update_enabled_state(self) -> None:
        grayscale = self._grayscale_check.isChecked()
        for combo in (self._red_combo, self._green_combo, self._blue_combo):
            combo.setEnabled(not grayscale)
        self._gray_combo.setEnabled(grayscale)

    def _on_changed(self) -> None:
        if self._updating:
            return
        self._update_enabled_state()
        mapping = self.current_mapping()
        self.mapping_changed.emit(mapping)

    def current_mapping(self) -> ChannelMapping:
        """Construit le :class:`ChannelMapping` correspondant à la sélection actuelle.

        Returns
        -------
        ChannelMapping
        """
        if self._grayscale_check.isChecked():
            return ChannelMapping(gray=self._gray_combo.currentData())
        return ChannelMapping(
            red=self._red_combo.currentData(),
            green=self._green_combo.currentData(),
            blue=self._blue_combo.currentData(),
        )
