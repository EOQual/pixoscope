"""Panneau de sélection d'un algorithme de rehaussement dynamique.

Ne liste que les algorithmes réellement disponibles dans l'environnement
courant (voir :meth:`pixoscope.processing.registry.ProcessingRegistry.available`),
avec un avertissement explicite pour les algorithmes coûteux sur de
grandes images (``slow_on_large_images``).
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QComboBox, QLabel, QVBoxLayout, QWidget

from pixoscope.processing.registry import ProcessingRegistry

_NONE_KEY = "__none__"
_NONE_LABEL = "Aucun (affichage direct)"


class ProcessingPanel(QWidget):
    """Sélecteur d'algorithme de rehaussement, appliqué à la vue active.

    Signals
    -------
    processing_changed(object)
        Émis avec la clé de l'algorithme sélectionné (``str``), ou
        ``None`` pour revenir à l'affichage direct.
    """

    processing_changed = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._combo = QComboBox()
        self._combo.addItem(_NONE_LABEL, _NONE_KEY)
        for spec in ProcessingRegistry.available():
            label = f"{spec.label} ⚠︎ lent" if spec.slow_on_large_images else spec.label
            self._combo.addItem(label, spec.key)
        self._combo.currentIndexChanged.connect(self._on_changed)

        self._warning = QLabel(
            "⚠︎ Cet algorithme peut être lent sur une grande tuile — testez sur une zone zoomée."
        )
        self._warning.setWordWrap(True)
        self._warning.hide()

        layout = QVBoxLayout(self)
        layout.addWidget(self._combo)
        layout.addWidget(self._warning)
        layout.addStretch(1)

    def _on_changed(self) -> None:
        key = self._combo.currentData()
        if key == _NONE_KEY:
            self._warning.hide()
            self.processing_changed.emit(None)
            return
        spec = ProcessingRegistry.get(key)
        self._warning.setVisible(spec.slow_on_large_images)
        self.processing_changed.emit(key)
