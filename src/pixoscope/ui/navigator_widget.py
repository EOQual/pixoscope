"""Vignette de navigation : aperçu de l'image entière + position de la vue.

``QWidget`` léger qui dessine juste un ``QPixmap`` déjà préparé (niveau
de pyramide le moins résolu, voir ``ViewPane``) et un rectangle de
position — aucune dépendance à Qt Graphics View, pas besoin de tuilage
pour une image toujours petite.

Ce widget est un **survol** posé directement dans le coin bas-droit du
viewport de la vue image (voir ``TiledImageView``) plutôt qu'un panneau
fixe dans la disposition — il n'occupe donc aucune place tant qu'il
n'est pas nécessaire, y compris en mode comparaison où deux panneaux se
partagent déjà l'espace.
"""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPaintEvent, QPen, QPixmap
from PySide6.QtWidgets import QWidget

#: Taille fixe de la vignette flottante (indépendante de la taille du
#: panneau parent, contrairement à l'ancien panneau intégré à la mise
#: en page).
_WIDTH = 200
_HEIGHT = 140


class NavigatorWidget(QWidget):
    """Vignette cliquable montrant où se situe la vue principale dans l'image.

    Parameters
    ----------
    parent : PySide6.QtWidgets.QWidget, optional

    Signals
    -------
    navigate_to(float, float)
        Émis avec des coordonnées ``(x, y)`` en pleine résolution
        lorsque l'utilisateur clique ou glisse dans la vignette.
    """

    navigate_to = Signal(float, float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(_WIDTH, _HEIGHT)
        self._thumbnail: QPixmap | None = None
        self._image_width = 0
        self._image_height = 0
        self._viewport_rect: QRectF | None = None

    def set_thumbnail(self, pixmap: QPixmap, image_width: int, image_height: int) -> None:
        """Définit l'image d'aperçu et les dimensions pleine résolution qu'elle représente.

        Parameters
        ----------
        pixmap : PySide6.QtGui.QPixmap
        image_width, image_height : int
        """
        self._thumbnail = pixmap
        self._image_width = image_width
        self._image_height = image_height
        self.update()

    def set_viewport_rect(self, rect: QRectF | None) -> None:
        """Définit le rectangle représentant la vue principale (coordonnées image).

        Parameters
        ----------
        rect : PySide6.QtCore.QRectF or None
        """
        self._viewport_rect = rect
        self.update()

    def _thumbnail_target_rect(self) -> QRectF:
        """Rectangle (coordonnées widget) où la vignette est dessinée, centrée."""
        if self._thumbnail is None or self._thumbnail.isNull():
            return QRectF()
        pixmap_w, pixmap_h = self._thumbnail.width(), self._thumbnail.height()
        if pixmap_w <= 0 or pixmap_h <= 0:
            return QRectF()
        scale = min(self.width() / pixmap_w, self.height() / pixmap_h)
        w, h = pixmap_w * scale, pixmap_h * scale
        return QRectF((self.width() - w) / 2, (self.height() - h) / 2, w, h)

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 - signature Qt
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        # Fond semi-transparent + bordure claire : le widget flotte
        # au-dessus de l'image, il doit se lire comme une incrustation
        # (HUD) plutôt que comme un panneau opaque de la disposition.
        painter.setPen(QPen(QColor(200, 200, 200, 160), 1))
        painter.setBrush(QColor(30, 30, 30, 210))
        painter.drawRoundedRect(self.rect().adjusted(0, 0, -1, -1), 6, 6)

        target = self._thumbnail_target_rect()
        if self._thumbnail is not None and not target.isEmpty():
            painter.drawPixmap(target.toRect(), self._thumbnail)

            if self._viewport_rect is not None and self._image_width and self._image_height:
                sx = target.width() / self._image_width
                sy = target.height() / self._image_height
                r = self._viewport_rect
                overlay = QRectF(
                    target.x() + r.x() * sx,
                    target.y() + r.y() * sy,
                    r.width() * sx,
                    r.height() * sy,
                )
                painter.setPen(QPen(QColor(255, 70, 70), 2))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRect(overlay)
        painter.end()

    def _widget_pos_to_image(self, pos: QPointF) -> QPointF | None:
        target = self._thumbnail_target_rect()
        if target.isEmpty() or self._image_width == 0 or self._image_height == 0:
            return None
        x = (pos.x() - target.x()) / target.width() * self._image_width
        y = (pos.y() - target.y()) / target.height() * self._image_height
        return QPointF(x, y)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - signature Qt
        self._navigate(event.position())

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - signature Qt
        if event.buttons() & Qt.MouseButton.LeftButton:
            self._navigate(event.position())

    def _navigate(self, pos: QPointF) -> None:
        image_pos = self._widget_pos_to_image(pos)
        if image_pos is not None:
            self.navigate_to.emit(image_pos.x(), image_pos.y())
