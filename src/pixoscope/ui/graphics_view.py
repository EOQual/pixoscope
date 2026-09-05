"""Vue graphique tuilée — coeur du rendu performant sur grosses images.

Simplification assumée par rapport au plan initial (voir
``REFERENCE_TECHNIQUE.md``) : plutôt qu'une mosaïque de nombreuses
tuiles mises en cache indépendamment, cette vue ne maintient qu'**un
seul raster**, couvrant la région actuellement visible (avec une marge),
lu à la résolution de pyramide adaptée au zoom courant. C'est ce raster
unique qui borne la mémoire et le volume lu depuis le disque à "environ
un écran" quelle que soit la taille du fichier source — la propriété de
performance recherchée — sans la complexité d'un cache multi-tuiles LRU.
Entre deux rafraîchissements (pan/zoom en cours), Qt ré-échantillonne le
raster déjà affiché pour un retour visuel immédiat.

Le rafraîchissement est différé (``_REFRESH_DELAY_MS``) après le dernier
événement de pan/zoom, pour éviter de déclencher une lecture disque à
chaque pixel de molette pendant un zoom continu.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from loguru import logger
from PySide6.QtCore import QPointF, QRect, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QPainter, QPixmap, QWheelEvent
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QGraphicsView, QWidget

from pixoscope.core.image_model import ImageDataset
from pixoscope.core.pyramid import level_for_zoom
from pixoscope.core.viewport import ViewportState
from pixoscope.ui.load_worker import run_in_background
from pixoscope.ui.navigator_widget import NavigatorWidget
from pixoscope.ui.qt_utils import array_to_qimage

try:
    import cv2

    _CV2_AVAILABLE = True
except ImportError:  # pragma: no cover - dépendance optionnelle (extra "enhance")
    cv2 = None
    _CV2_AVAILABLE = False

_REFRESH_DELAY_MS = 80
#: Marge ajoutée autour de la région visible avant lecture, en fraction
#: de la taille de la vue — réduit le nombre de relectures pendant un
#: petit pan.
_MARGIN_FACTOR = 0.35
_MIN_SCALE = 0.01
_MAX_SCALE = 40.0
#: Unité de référence d'un "cran" de molette classique (convention Qt).
_WHEEL_UNIT = 120.0
#: Facteur de zoom appliqué pour un cran standard, à sensibilité 1.0.
_WHEEL_FACTOR_PER_UNIT = 1.25
#: Marge (px écran) entre le navigateur flottant et le bord du viewport.
_NAVIGATOR_MARGIN = 10


class InterpolationMode(Enum):
    """Méthode de ré-échantillonnage utilisée pour l'affichage zoomé.

    Attributes
    ----------
    NEAREST : plus proche voisin — aucun lissage, pixels bruts visibles
        (utile pour l'inspection pixel par pixel).
    BILINEAR : lissage bilinéaire natif Qt (``SmoothTransformation``) —
        réglage par défaut, bon compromis qualité/performance.
    BICUBIC : ré-échantillonnage bicubique (OpenCV), calculé sur le
        tableau affiché avant conversion en ``QPixmap``. Se replie
        silencieusement sur ``BILINEAR`` si OpenCV (extra
        ``pixoscope[enhance]``) n'est pas installé.
    """

    NEAREST = "nearest"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"


@dataclass(frozen=True)
class _TileRequest:
    """Requête de lecture émise vers le thread de fond."""

    generation: int
    level: int
    x: int
    y: int
    w: int
    h: int
    scene_x: float
    scene_y: float
    scene_w: float
    scene_h: float


class TiledImageView(QGraphicsView):
    """Vue pan/zoom d'une :class:`~pixoscope.core.image_model.ImageDataset`.

    Parameters
    ----------
    pane_id : str
        Identifiant unique de la vue (sert de clé pour
        :class:`~pixoscope.core.viewport.ViewportLinker` en mode
        comparaison).
    parent : PySide6.QtWidgets.QWidget, optional

    Signals
    -------
    viewport_changed(ViewportState)
        Émis (avec un léger différé) après chaque pan/zoom.
    pixel_hovered(object)
        Émis avec ``(x, y)`` en coordonnées image pleine résolution, ou
        ``None`` quand le curseur quitte la vue — pour l'affichage de la
        valeur du pixel survolé.
    activated()
        Émis à chaque clic dans la vue — en mode comparaison, indique à
        :class:`~pixoscope.ui.main_window.MainWindow` quel panneau doit
        devenir la cible des panneaux latéraux (canaux, statistiques,
        rehaussement). Sans ce signal, cliquer dans l'image ne
        remontait jamais jusqu'à ``ViewPane`` (l'évènement est capté par
        cette vue, pas par son parent), rendant le panneau de droite
        impossible à activer en pratique.
    """

    viewport_changed = Signal(object)
    pixel_hovered = Signal(object)
    activated = Signal()

    def __init__(self, pane_id: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.pane_id = pane_id
        self._dataset: ImageDataset | None = None
        self._generation = 0
        self._pending_generation = -1

        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item = QGraphicsPixmapItem()
        self._pixmap_item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self._scene.addItem(self._pixmap_item)

        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setMouseTracking(True)
        self.setBackgroundBrush(Qt.GlobalColor.darkGray)

        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(_REFRESH_DELAY_MS)
        self._refresh_timer.timeout.connect(self._refresh)

        self._applying_external_state = False
        self._applying_fit = False
        #: True tant que la vue est dans l'état "image entière ajustée" et
        #: qu'aucun zoom/pan manuel n'en est sorti — permet de ré-ajuster
        #: automatiquement au redimensionnement (voir resizeEvent) plutôt
        #: que de garder une transform obsolète calculée pour l'ancienne
        #: taille de la vue.
        self._is_fit_to_view = False

        #: Multiplicateur appliqué à chaque événement molette/trackpad
        #: (voir ``set_zoom_sensitivity``) — réglable par l'utilisateur,
        #: la molette d'une souris et le trackpad d'un Mac n'envoyant pas
        #: du tout les mêmes grandeurs de delta.
        self._zoom_sensitivity = 1.0
        self._interpolation = InterpolationMode.BILINEAR

        # Navigateur flottant (vignette de position) : posé directement
        # sur le viewport plutôt que dans la disposition du panneau, pour
        # ne jamais réduire la place disponible pour l'image (demande
        # explicite de l'utilisateur, notamment gênant en mode
        # comparaison où deux vues se partagent déjà l'espace).
        self.navigator = NavigatorWidget(self.viewport())
        self.navigator.setVisible(False)

    # ------------------------------------------------------------------
    # Chargement
    # ------------------------------------------------------------------
    def set_dataset(self, dataset: ImageDataset | None) -> None:
        """Change l'image affichée et recadre la vue sur l'image entière.

        Parameters
        ----------
        dataset : ImageDataset or None
        """
        self._dataset = dataset
        self._generation += 1
        if dataset is None:
            self._pixmap_item.setPixmap(QPixmap())
            self.navigator.setVisible(False)
            return
        height, width = dataset.shape
        self._scene.setSceneRect(0, 0, width, height)
        self.fit_to_view()

    @property
    def dataset(self) -> ImageDataset | None:
        """L'``ImageDataset`` actuellement affiché, s'il y en a un."""
        return self._dataset

    def request_refresh(self) -> None:
        """Force un rafraîchissement du raster affiché (différé, voir
        ``_REFRESH_DELAY_MS``) — à appeler après un changement qui
        modifie le rendu sans changer le pan/zoom (mapping de canaux,
        étirement, algorithme de rehaussement).
        """
        self._schedule_refresh()

    def fit_to_view(self) -> None:
        """Cadre l'image entière dans la vue (équivalent du bouton "Home")."""
        if self._dataset is None:
            return
        self._applying_fit = True
        try:
            self.resetTransform()
            self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        finally:
            self._applying_fit = False
        self._is_fit_to_view = True
        self._schedule_refresh()
        self._emit_viewport_state()

    def set_zoom_sensitivity(self, sensitivity: float) -> None:
        """Règle la sensibilité du zoom molette/trackpad.

        Parameters
        ----------
        sensitivity : float
            ``1.0`` = un cran de molette standard change le zoom de 25 %
            (comportement d'origine). Une valeur plus basse ralentit le
            zoom (utile sur trackpad Mac, dont chaque petit mouvement
            émet déjà de nombreux évènements) ; une valeur plus haute
            l'accélère.
        """
        self._zoom_sensitivity = max(0.1, sensitivity)

    def set_interpolation_mode(self, mode: InterpolationMode) -> None:
        """Change la méthode de ré-échantillonnage utilisée à l'affichage.

        Parameters
        ----------
        mode : InterpolationMode
        """
        if mode is InterpolationMode.BICUBIC and not _CV2_AVAILABLE:
            logger.warning(
                "Interpolation bicubique demandée mais OpenCV n'est pas installé "
                "(extra pixoscope[enhance]) : repli sur le lissage bilinéaire."
            )
            mode = InterpolationMode.BILINEAR
        self._interpolation = mode
        # "Bicubique" redimensionne déjà le tableau à la taille écran
        # exacte avant d'en faire un QPixmap (voir _resize_bicubic) : Qt
        # ne doit alors plus rééchantillonner par-dessus, d'où
        # FastTransformation également dans ce cas (pas seulement pour
        # "plus proche voisin"). Seul "bilinéaire" laisse Qt lisser.
        smooth = mode is InterpolationMode.BILINEAR
        self._pixmap_item.setTransformationMode(
            Qt.TransformationMode.SmoothTransformation if smooth else Qt.TransformationMode.FastTransformation
        )
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, smooth)
        self._schedule_refresh()

    def set_rubber_band_zoom_enabled(self, enabled: bool) -> None:
        """Bascule entre l'outil "main" (pan) et l'outil "zoom rectangle".

        Parameters
        ----------
        enabled : bool
            ``True`` : un cliquer-glisser sélectionne un rectangle et
            zoome dessus au relâchement. ``False`` : un cliquer-glisser
            déplace la vue (comportement par défaut).
        """
        self.setDragMode(
            QGraphicsView.DragMode.RubberBandDrag if enabled else QGraphicsView.DragMode.ScrollHandDrag
        )

    # ------------------------------------------------------------------
    # Interaction
    # ------------------------------------------------------------------
    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802 - signature Qt
        """Molette/trackpad : zoome, centré sur le curseur.

        Convention reprise de QGIS (molette = zoom, cliquer-glisser
        = déplacement) plutôt qu'un modificateur Ctrl : sur macOS,
        Ctrl+molette/pincement est souvent intercepté par la fonction
        d'accessibilité "Zoom" du système avant même d'atteindre l'appli,
        ce qui rendait le zoom inopérant.

        Le facteur appliqué est **proportionnel** à ``delta`` plutôt que
        fixe : un trackpad Mac envoie de très nombreux évènements par
        geste, chacun avec un delta bien plus petit qu'un "cran" de
        molette classique (souvent ``pixelDelta`` seul, de quelques
        unités). Appliquer un facteur fixe à chacun rendait le zoom
        beaucoup trop sensible sur trackpad ; ``set_zoom_sensitivity``
        permet en plus à l'utilisateur d'ajuster la réactivité globale.
        """
        if self._dataset is None:
            return
        # Sur trackpad, angleDelta peut rester à 0 (seul pixelDelta est
        # renseigné) : on retombe sur pixelDelta dans ce cas.
        delta = event.angleDelta().y() or event.pixelDelta().y()
        if delta == 0:
            return
        exponent = (delta / _WHEEL_UNIT) * self._zoom_sensitivity
        factor = _WHEEL_FACTOR_PER_UNIT**exponent
        current_scale = self.transform().m11()
        new_scale = current_scale * factor
        if not (_MIN_SCALE <= new_scale <= _MAX_SCALE):
            event.accept()
            return
        self._is_fit_to_view = False
        self.scale(factor, factor)
        self._schedule_refresh()
        self._emit_viewport_state()
        event.accept()

    def mousePressEvent(self, event) -> None:  # noqa: N802 - signature Qt
        """Un clic dans la vue signale que ce panneau devient l'actif.

        En mode comparaison, c'est le seul moyen fiable de choisir quel
        panneau les onglets latéraux (canaux, statistiques, rehaussement)
        doivent cibler : cette vue est un enfant de ``ViewPane``, un clic
        à l'intérieur n'atteint donc jamais ``ViewPane.mousePressEvent``.
        """
        self.activated.emit()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - signature Qt
        """Termine un cliquer-glisser — zoome sur la zone en mode "zoom rectangle".

        ``rubberBandRect()`` reste valide juste avant que la classe de
        base ne le réinitialise : on le lit d'abord, en coordonnées
        viewport (indépendantes du zoom courant, contrairement aux
        coordonnées scène) pour filtrer un simple clic, avant d'appeler
        ``super()``.
        """
        rubber_rect = None
        if self.dragMode() == QGraphicsView.DragMode.RubberBandDrag:
            rubber_rect = QRect(self.rubberBandRect())

        super().mouseReleaseEvent(event)

        if (
            self._dataset is not None
            and rubber_rect is not None
            and rubber_rect.width() >= 8
            and rubber_rect.height() >= 8
        ):
            scene_rect = self.mapToScene(rubber_rect).boundingRect()
            self._is_fit_to_view = False
            self.fitInView(scene_rect, Qt.AspectRatioMode.KeepAspectRatio)
            self._schedule_refresh()
            self._emit_viewport_state()

    def scrollContentsBy(self, dx: int, dy: int) -> None:  # noqa: N802 - signature Qt
        super().scrollContentsBy(dx, dy)
        if self._applying_external_state or self._applying_fit:
            return
        self._is_fit_to_view = False
        self._schedule_refresh()
        self._emit_viewport_state()

    def resizeEvent(self, event) -> None:  # noqa: N802 - signature Qt
        super().resizeEvent(event)
        if self._is_fit_to_view:
            # Ré-ajuste à la nouvelle taille plutôt que de garder une
            # transform calculée pour l'ancienne géométrie (survient
            # notamment juste après l'ouverture d'un fichier, tant que la
            # disposition des panneaux n'est pas encore stabilisée).
            self.fit_to_view()
        else:
            self._schedule_refresh()
            self._emit_viewport_state()
        self._position_navigator()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - signature Qt
        super().mouseMoveEvent(event)
        if self._dataset is None:
            return
        scene_pos = self.mapToScene(event.position().toPoint())
        height, width = self._dataset.shape
        if 0 <= scene_pos.x() < width and 0 <= scene_pos.y() < height:
            self.pixel_hovered.emit((int(scene_pos.x()), int(scene_pos.y())))
        else:
            self.pixel_hovered.emit(None)

    def leaveEvent(self, event) -> None:  # noqa: N802 - signature Qt
        super().leaveEvent(event)
        self.pixel_hovered.emit(None)

    # ------------------------------------------------------------------
    # Synchronisation multi-vues (mode comparaison)
    # ------------------------------------------------------------------
    def current_viewport_state(self) -> ViewportState:
        """État de viewport courant, pour publication via ``ViewportLinker``.

        Returns
        -------
        ViewportState
        """
        center = self.mapToScene(self.viewport().rect().center())
        return ViewportState(center_x=center.x(), center_y=center.y(), zoom=self.transform().m11())

    def visible_image_rect(self) -> QRectF:
        """Région actuellement visible, en coordonnées image pleine résolution.

        Returns
        -------
        PySide6.QtCore.QRectF
        """
        return self.mapToScene(self.viewport().rect()).boundingRect()

    def is_full_image_visible(self) -> bool:
        """Indique si l'image entière tient dans la vue actuelle.

        Utilisé pour décider si le navigateur (vignette de position)
        doit être affiché — inutile tant que l'image entière est visible.

        Returns
        -------
        bool
        """
        if self._dataset is None:
            return True
        height, width = self._dataset.shape
        visible = self.visible_image_rect()
        image_rect = QRectF(0, 0, width, height)
        # Petite tolérance : fitInView peut laisser un écart de l'ordre du
        # pixel par arrondi, sans que l'image entière ne soit pour autant
        # coupée.
        tolerance = max(width, height, 1) * 0.01
        return visible.adjusted(-tolerance, -tolerance, tolerance, tolerance).contains(image_rect)

    def set_viewport(self, state: ViewportState) -> None:
        """Applique un état de viewport reçu d'une autre vue synchronisée.

        Parameters
        ----------
        state : ViewportState
        """
        if self._dataset is None:
            return
        current = self.current_viewport_state()
        close_enough = (
            abs(current.zoom - state.zoom) < 1e-6
            and abs(current.center_x - state.center_x) < 0.5
            and abs(current.center_y - state.center_y) < 0.5
        )
        if close_enough:
            return  # évite une boucle d'aller-retour entre deux vues liées

        self._applying_external_state = True
        try:
            self.resetTransform()
            self.scale(state.zoom, state.zoom)
            self.centerOn(QPointF(state.center_x, state.center_y))
        finally:
            self._applying_external_state = False
        self._is_fit_to_view = False
        self._schedule_refresh()

    def _emit_viewport_state(self) -> None:
        self.viewport_changed.emit(self.current_viewport_state())
        self.update_navigator()

    # ------------------------------------------------------------------
    # Navigateur flottant (vignette de position)
    # ------------------------------------------------------------------
    def update_navigator(self) -> None:
        """Affiche/masque le navigateur et met à jour le cadre + sa position.

        N'est visible que lorsque l'image entière ne tient plus dans la
        vue — inutile (et gênant) tant qu'il n'y a rien à situer.
        """
        if self._dataset is None:
            self.navigator.setVisible(False)
            return
        full_visible = self.is_full_image_visible()
        self.navigator.setVisible(not full_visible)
        if not full_visible:
            self.navigator.set_viewport_rect(self.visible_image_rect())
            self._position_navigator()

    def _position_navigator(self) -> None:
        """Ancre le navigateur flottant dans le coin bas-droit du viewport."""
        size = self.navigator.size()
        vp = self.viewport().size()
        x = max(0, vp.width() - size.width() - _NAVIGATOR_MARGIN)
        y = max(0, vp.height() - size.height() - _NAVIGATOR_MARGIN)
        self.navigator.move(x, y)
        self.navigator.raise_()

    # ------------------------------------------------------------------
    # Rafraîchissement du raster affiché
    # ------------------------------------------------------------------
    def _schedule_refresh(self) -> None:
        try:
            self._refresh_timer.start()
        except RuntimeError:
            # La vue a été détruite entre-temps (fermeture d'un panneau,
            # etc.) alors qu'un résultat en arrière-plan (statistiques,
            # tuile) venait encore de déclencher un rafraîchissement —
            # rien à faire, voir _on_tile_ready pour le même cas.
            logger.trace("Rafraîchissement ignoré : la vue a été détruite")

    def _refresh(self) -> None:
        dataset = self._dataset
        if dataset is None:
            return
        height, width = dataset.shape

        visible = self.mapToScene(self.viewport().rect()).boundingRect()
        margin_x = visible.width() * _MARGIN_FACTOR
        margin_y = visible.height() * _MARGIN_FACTOR
        region = visible.adjusted(-margin_x, -margin_y, margin_x, margin_y)
        region = region.intersected(QRectF(0, 0, width, height))
        if region.width() < 1 or region.height() < 1:
            return

        zoom = self.transform().m11()
        level = level_for_zoom(dataset.levels, zoom)
        level_info = dataset.levels[level]
        downsample_x = width / level_info.width if level_info.width else 1.0
        downsample_y = height / level_info.height if level_info.height else 1.0

        lx = int(region.x() / downsample_x)
        ly = int(region.y() / downsample_y)
        lw = max(1, int(region.width() / downsample_x))
        lh = max(1, int(region.height() / downsample_y))

        self._generation += 1
        request = _TileRequest(
            generation=self._generation,
            level=level,
            x=lx,
            y=ly,
            w=lw,
            h=lh,
            scene_x=region.x(),
            scene_y=region.y(),
            scene_w=region.width(),
            scene_h=region.height(),
        )
        self._pending_generation = request.generation
        run_in_background(
            dataset.read_display_tile,
            request.level,
            request.x,
            request.y,
            request.w,
            request.h,
            on_result=lambda array, req=request: self._on_tile_ready(req, array),  # type: ignore[misc]
            on_error=lambda msg: logger.warning(f"Échec de lecture de tuile : {msg}"),
        )

    def _on_tile_ready(self, request: _TileRequest, array: np.ndarray) -> None:
        if request.generation != self._pending_generation:
            return  # une vue plus récente a déjà été demandée entre-temps

        if self._interpolation is InterpolationMode.BICUBIC and _CV2_AVAILABLE:
            array = self._resize_bicubic(array, request)

        pixmap = QPixmap.fromImage(array_to_qimage(array))
        if pixmap.width() == 0 or pixmap.height() == 0:
            return
        try:
            self._pixmap_item.setPixmap(pixmap)
            self._pixmap_item.setOffset(0, 0)
            self._pixmap_item.setPos(request.scene_x, request.scene_y)
            scale_x = request.scene_w / pixmap.width()
            scale_y = request.scene_h / pixmap.height()
            transform = self._pixmap_item.transform()
            transform.reset()
            transform.scale(scale_x, scale_y)
            self._pixmap_item.setTransform(transform)
        except RuntimeError:
            # La vue a été fermée (mode comparaison désactivé, etc.) entre
            # la requête et la réponse : rien à afficher, rien à faire.
            logger.trace("Tuile reçue pour une vue déjà fermée, ignorée")

    def _resize_bicubic(self, array: np.ndarray, request: _TileRequest) -> np.ndarray:
        """Ré-échantillonne le tableau affiché à la taille écran exacte (bicubique).

        Qt ne propose nativement que "plus proche voisin" et un lissage
        bilinéaire pour la mise à l'échelle d'un ``QGraphicsPixmapItem`` ;
        pour un vrai bicubique, le tableau est donc pré-redimensionné ici
        (OpenCV) à la taille écran attendue *avant* d'en faire un
        ``QPixmap``, puis affiché sans transformation Qt supplémentaire
        (voir ``set_interpolation_mode``, qui bascule alors l'item en
        ``FastTransformation``).

        Parameters
        ----------
        array : numpy.ndarray
        request : _TileRequest

        Returns
        -------
        numpy.ndarray
        """
        view_scale_x = abs(self.transform().m11())
        view_scale_y = abs(self.transform().m22())
        target_w = max(1, round(request.scene_w * view_scale_x))
        target_h = max(1, round(request.scene_h * view_scale_y))
        if (target_w, target_h) == (array.shape[1], array.shape[0]):
            return array
        # Redimensionner ici change ce que représente une case du
        # tableau (un pixel écran, plus un pixel image) : le facteur
        # scene/pixmap recalculé dans _on_tile_ready à partir de la
        # nouvelle largeur/hauteur compense automatiquement ce
        # changement, aucune autre correction n'est nécessaire ici.
        return cv2.resize(array, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
