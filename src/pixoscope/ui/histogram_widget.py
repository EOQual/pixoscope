"""Histogramme dessiné en ``QPainter`` — pas de dépendance matplotlib.

Le widget se contente d'afficher des
:class:`~pixoscope.core.stats.BandStats` déjà calculées ailleurs (voir
``pixoscope.ui.stats_panel``, qui délègue le calcul à un thread de fond
sur un échantillon sous-résolu, jamais sur l'image pleine résolution) —
ce widget ne fait aucun calcul, seulement du dessin.
"""

from __future__ import annotations

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget

from pixoscope.core.stats import BandStats

#: Couleurs par canal, dans l'ordre où elles seront superposées (alpha
#: réduit pour que les recouvrements restent lisibles en RVB).
_CHANNEL_COLORS: dict[str, QColor] = {
    "Gris": QColor(220, 220, 220, 200),
    "Rouge": QColor(220, 60, 60, 150),
    "Vert": QColor(60, 180, 60, 150),
    "Bleu": QColor(60, 100, 220, 150),
}


class HistogramWidget(QWidget):
    """Affiche un ou plusieurs histogrammes de bande, superposés.

    Parameters
    ----------
    parent : PySide6.QtWidgets.QWidget, optional
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._stats: dict[str, BandStats] = {}
        self.setMinimumHeight(120)

    def set_stats(self, stats: dict[str, BandStats]) -> None:
        """Remplace les statistiques affichées et redessine.

        Parameters
        ----------
        stats : dict of str to BandStats
            Clés parmi ``"Gris"``, ``"Rouge"``, ``"Vert"``, ``"Bleu"``.
        """
        self._stats = stats
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 - signature Qt
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), self.palette().base())

        if not self._stats:
            painter.setPen(QPen(self.palette().mid().color()))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Aucune statistique disponible")
            painter.end()
            return

        max_count = max(
            (int(stats.histogram_counts.max()) if stats.histogram_counts.size else 0 for stats in self._stats.values()),
            default=0,
        )
        if max_count <= 0:
            painter.end()
            return

        plot_rect = QRectF(self.rect()).adjusted(4, 4, -4, -20)
        for label, stats in self._stats.items():
            color = _CHANNEL_COLORS.get(label, QColor(180, 180, 180, 150))
            self._draw_histogram(painter, plot_rect, stats, color)

        painter.setPen(QPen(self.palette().text().color()))
        first_stats = next(iter(self._stats.values()))
        note = f"aperçu {first_stats.sample_shape[0]}x{first_stats.sample_shape[1]}" if first_stats.sample_shape else ""
        painter.drawText(
            self.rect().adjusted(4, 0, -4, -4),
            Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignLeft,
            f"min {first_stats.minimum:.1f}  max {first_stats.maximum:.1f}  {note}",
        )
        painter.end()

    def _draw_histogram(self, painter: QPainter, rect: QRectF, stats: BandStats, color: QColor) -> None:
        counts = stats.histogram_counts
        if counts.size == 0:
            return
        max_count = max(int(counts.max()), 1)
        n_bins = counts.size
        bin_width = rect.width() / n_bins

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color)
        for i, count in enumerate(counts):
            bar_height = (count / max_count) * rect.height()
            x = rect.left() + i * bin_width
            y = rect.bottom() - bar_height
            painter.drawRect(QRectF(x, y, max(bin_width, 1.0), bar_height))
