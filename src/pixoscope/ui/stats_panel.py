"""Panneau de statistiques + histogramme pour les canaux affichés.

Ne calcule jamais rien lui-même : affiche ce qui est déjà présent dans
``ImageDataset.stats_cache``, alimenté en arrière-plan par
``MainWindow`` (voir ``pixoscope.ui.main_window._compute_band_stats``)
sur un niveau de pyramide sous-résolu — jamais sur l'image pleine
résolution.
"""

from __future__ import annotations

from PySide6.QtWidgets import QFormLayout, QGroupBox, QLabel, QVBoxLayout, QWidget

from pixoscope.core.image_model import ChannelMapping, ImageDataset
from pixoscope.ui.histogram_widget import HistogramWidget

_PENDING_TEXT = "Calcul en cours…"


class StatsPanel(QWidget):
    """Statistiques numériques et histogramme des canaux actifs.

    Parameters
    ----------
    parent : PySide6.QtWidgets.QWidget, optional
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._histogram = HistogramWidget()

        group = QGroupBox("Statistiques")
        self._form = QFormLayout(group)
        self._value_labels: dict[str, QLabel] = {}

        layout = QVBoxLayout(self)
        layout.addWidget(self._histogram)
        layout.addWidget(group)
        layout.addStretch(1)

    def refresh(self, dataset: ImageDataset | None, mapping: ChannelMapping | None = None) -> None:
        """Met à jour l'affichage à partir de l'état courant du dataset.

        Parameters
        ----------
        dataset : ImageDataset or None
        mapping : ChannelMapping, optional
            Par défaut, ``dataset.channel_mapping``.
        """
        self._clear_rows()
        if dataset is None:
            self._histogram.set_stats({})
            return

        mapping = mapping or dataset.channel_mapping
        channels = self._active_channels(mapping)

        histogram_stats = {}
        for label, band_index in channels:
            stats = dataset.stats_cache.get(band_index)
            if stats is None:
                self._add_row(label, _PENDING_TEXT)
                continue
            histogram_stats[label] = stats
            self._add_row(
                label,
                f"min {stats.minimum:.1f}  max {stats.maximum:.1f}  "
                f"moy {stats.mean:.1f}  éc.-t. {stats.std:.1f}",
            )
        self._histogram.set_stats(histogram_stats)

    @staticmethod
    def _active_channels(mapping: ChannelMapping) -> list[tuple[str, int]]:
        if mapping.is_grayscale:
            return [("Gris", mapping.gray)] if mapping.gray is not None else []
        channels = []
        for label, band in (("Rouge", mapping.red), ("Vert", mapping.green), ("Bleu", mapping.blue)):
            if band is not None:
                channels.append((label, band))
        return channels

    def _clear_rows(self) -> None:
        while self._form.rowCount():
            self._form.removeRow(0)
        self._value_labels.clear()

    def _add_row(self, label: str, text: str) -> None:
        value_label = QLabel(text)
        self._value_labels[label] = value_label
        self._form.addRow(label, value_label)
