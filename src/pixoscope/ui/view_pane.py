"""Un panneau de visualisation : ouverture de fichier + vue tuilée + statut.

En mode comparaison, ``MainWindow`` instancie deux :class:`ViewPane`
côte à côte dans un ``QSplitter`` ; en mode simple, une seule.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from loguru import logger
from PySide6.QtCore import QPointF, Qt, QTimer, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QFileDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from pixoscope.core.image_model import ImageDataset
from pixoscope.ui.graphics_view import TiledImageView
from pixoscope.ui.load_worker import run_in_background
from pixoscope.ui.qt_utils import array_to_qimage

_HOVER_THROTTLE_MS = 40


def _compute_navigator_thumbnail(dataset: ImageDataset) -> tuple[np.ndarray, int, int]:
    """Construit la vignette (niveau de pyramide le moins résolu) pour le navigateur.

    Exécuté en arrière-plan (voir :func:`~pixoscope.ui.load_worker.run_in_background`).

    Parameters
    ----------
    dataset : ImageDataset

    Returns
    -------
    tuple
        ``(tableau uint8, largeur pleine résolution, hauteur pleine résolution)``.
    """
    level_info = dataset.levels[-1]
    array = dataset.read_display_tile(level_info.level, 0, 0, level_info.width, level_info.height)
    height, width = dataset.shape
    return array, width, height

_OPEN_FILTER = (
    "Images (*.tif *.tiff *.png *.jpg *.jpeg *.bmp *.gif *.webp *.lum *.jp2);;Tous les fichiers (*)"
)


class ViewPane(QWidget):
    """Panneau autonome : barre d'outils + vue tuilée.

    Parameters
    ----------
    pane_id : str
        Identifiant transmis à :class:`~pixoscope.ui.graphics_view.TiledImageView`.
    parent : PySide6.QtWidgets.QWidget, optional

    Signals
    -------
    dataset_opened(object)
        Émis avec l'``ImageDataset`` nouvellement ouvert.
    activated()
        Émis quand ce panneau devient le panneau actif (clic dans la vue).
    """

    dataset_opened = Signal(object)
    activated = Signal()

    def __init__(self, pane_id: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.pane_id = pane_id
        self.dataset: ImageDataset | None = None
        # Nécessaire pour qu'une bordure de style CSS soit effectivement
        # peinte sur un QWidget nu (voir set_active).
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        self._open_button = QPushButton("Ouvrir…")
        self._open_button.clicked.connect(self._on_open_clicked)
        self._home_button = QPushButton("Ajuster")
        self._home_button.clicked.connect(self._on_home_clicked)
        self._title_label = QLabel("(aucune image)")

        top_bar = QHBoxLayout()
        top_bar.addWidget(self._open_button)
        top_bar.addWidget(self._home_button)
        top_bar.addWidget(self._title_label, stretch=1)

        self.view = TiledImageView(pane_id)
        self.view.pixel_hovered.connect(self._on_pixel_hovered)
        # Un clic dans l'image doit activer ce panneau au même titre
        # qu'un clic sur sa barre d'outils (voir mousePressEvent
        # ci-dessous) : la vue étant un enfant de ce widget, l'évènement
        # ne remonterait sinon jamais jusqu'ici.
        self.view.activated.connect(self.activated.emit)
        # Le navigateur (vignette de position) est un survol posé par
        # TiledImageView directement sur son viewport (voir
        # graphics_view.py) : il n'a pas sa place dans cette disposition,
        # justement pour ne jamais réduire l'espace disponible pour
        # l'image. ViewPane se contente de fournir sa vignette et de
        # réagir à un clic dedans.
        self.view.navigator.navigate_to.connect(self._on_navigate_to)

        self._status_label = QLabel(" ")

        layout = QVBoxLayout(self)
        layout.addLayout(top_bar)
        layout.addWidget(self.view, stretch=1)
        layout.addWidget(self._status_label)

        self._hover_timer = QTimer(self)
        self._hover_timer.setSingleShot(True)
        self._hover_timer.setInterval(_HOVER_THROTTLE_MS)
        self._pending_hover: tuple[int, int] | None = None
        self._hover_timer.timeout.connect(self._show_pixel_value)

        self.set_active(False)

    # ------------------------------------------------------------------
    def set_active(self, active: bool) -> None:
        """Marque visuellement ce panneau comme cible des onglets latéraux.

        Utile seulement en mode comparaison (deux panneaux) : indique
        sans ambiguïté lequel des deux reçoit les changements de canaux,
        de rehaussement, etc.

        Parameters
        ----------
        active : bool
        """
        border = "#4a90e2" if active else "transparent"
        self.setStyleSheet(f"ViewPane {{ border: 2px solid {border}; border-radius: 3px; }}")

    # ------------------------------------------------------------------
    def mousePressEvent(self, event) -> None:  # noqa: N802 - signature Qt
        super().mousePressEvent(event)
        self.activated.emit()

    def focusInEvent(self, event) -> None:  # noqa: N802 - signature Qt
        super().focusInEvent(event)
        self.activated.emit()

    # ------------------------------------------------------------------
    def _on_open_clicked(self) -> None:
        self.activated.emit()
        self.open_dialog()

    def open_dialog(self) -> None:
        """Ouvre la boîte de dialogue de sélection de fichier et charge le choix."""
        filename, _ = QFileDialog.getOpenFileName(self, "Ouvrir une image", "", _OPEN_FILTER)
        if filename:
            self.open_file(filename)

    def open_file(self, path: str | Path) -> None:
        """Ouvre un fichier image en arrière-plan et l'affiche une fois prêt.

        Parameters
        ----------
        path : str or pathlib.Path
        """
        self._title_label.setText(f"Ouverture de {Path(path).name}…")
        run_in_background(
            ImageDataset.open,
            path,
            on_result=lambda dataset, p=path: self._on_dataset_opened(p, dataset),  # type: ignore[misc]
            on_error=lambda msg, p=path: self._on_open_failed(p, msg),  # type: ignore[misc]
        )

    def _on_dataset_opened(self, path: str | Path, dataset: ImageDataset) -> None:
        if self.dataset is not None:
            self.dataset.close()
        self.dataset = dataset
        self.view.set_dataset(dataset)
        self._title_label.setText(Path(path).name)
        self.view.navigator.set_thumbnail(QPixmap(), 0, 0)
        run_in_background(
            _compute_navigator_thumbnail,
            dataset,
            on_result=self._on_navigator_thumbnail_ready,
            on_error=lambda msg: logger.warning(f"Échec de calcul de la vignette de navigation : {msg}"),
        )
        self.dataset_opened.emit(dataset)

    def _on_navigator_thumbnail_ready(self, result: tuple[np.ndarray, int, int]) -> None:
        array, width, height = result
        self.view.navigator.set_thumbnail(QPixmap.fromImage(array_to_qimage(array)), width, height)
        self.view.update_navigator()

    def _on_open_failed(self, path: str | Path, message: str) -> None:
        logger.warning(f"Échec d'ouverture de [{path}] : {message}")
        self._title_label.setText(f"Erreur : {message}")

    def _on_home_clicked(self) -> None:
        self.view.fit_to_view()

    # ------------------------------------------------------------------
    def _on_navigate_to(self, x: float, y: float) -> None:
        self.view.centerOn(QPointF(x, y))

    # ------------------------------------------------------------------
    def _on_pixel_hovered(self, position: object) -> None:
        self._pending_hover = position  # type: ignore[assignment]
        if not self._hover_timer.isActive():
            self._hover_timer.start()

    def _show_pixel_value(self) -> None:
        position = self._pending_hover
        if position is None or self.dataset is None:
            self._status_label.setText(" ")
            return
        x, y = position
        try:
            # Lecture synchrone d'un pixel unique : généralement rapide
            # (chunk déjà en cache disque après un premier accès à cette
            # zone) ; une lecture asynchrone serait nécessaire si cette
            # latence devenait sensible sur un stockage réseau lent.
            values = self.dataset.handle.read_region(0, x, y, 1, 1)
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"Lecture de pixel échouée en ({x}, {y}) : {exc}")
            return
        self._status_label.setText(f"x={x}  y={y}  valeur={_format_pixel(values)}")


def _format_pixel(values) -> str:  # noqa: ANN001 - numpy scalar/array
    flat = values.reshape(-1)
    if flat.size == 1:
        return str(flat[0])
    return ", ".join(str(v) for v in flat)
