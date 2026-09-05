"""État de viewport et synchronisation multi-vues (mode comparaison).

Aucune dépendance Qt ici : la vue graphique (``pixoscope.ui.graphics_view``)
traduit ses événements (zoom molette, pan, redimensionnement) en
:class:`ViewportState` et les publie via :class:`ViewportLinker` ; c'est
``pixoscope.ui.view_pane`` qui connecte ce mécanisme aux signaux Qt entre
deux panneaux quand le mode comparaison est actif.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class ViewportState:
    """Portion de l'image actuellement visible, exprimée en coordonnées
    pleine résolution (niveau 0), indépendamment du niveau de pyramide
    réellement utilisé pour le rendu.

    Attributes
    ----------
    center_x, center_y : float
        Centre de la vue, en pixels pleine résolution.
    zoom : float
        Pixels écran par pixel image pleine résolution (voir
        :func:`pixoscope.core.pyramid.level_for_zoom`).
    """

    center_x: float
    center_y: float
    zoom: float


class ViewportLinker:
    """Diffuse un :class:`ViewportState` à des abonnés, avec anti-boucle.

    Utilisé pour synchroniser le pan/zoom de plusieurs vues en mode
    comparaison : chaque vue s'abonne avec :meth:`subscribe` et publie
    ses changements avec :meth:`publish` ; les autres abonnés sont
    notifiés, mais jamais l'abonné à l'origine de la publication (évite
    la boucle infinie A déclenche B déclenche A).

    Examples
    --------
    >>> linker = ViewportLinker()
    >>> received = []
    >>> unsubscribe_a = linker.subscribe("a", received.append)
    >>> linker.subscribe("b", lambda state: None)
    >>> linker.publish("b", ViewportState(0.0, 0.0, 1.0))
    >>> len(received)
    1
    """

    def __init__(self) -> None:
        self._subscribers: dict[str, Callable[[ViewportState], None]] = {}
        self._publishing = False

    def subscribe(self, subscriber_id: str, callback: Callable[[ViewportState], None]) -> Callable[[], None]:
        """Abonne ``callback`` aux publications des autres vues.

        Parameters
        ----------
        subscriber_id : str
            Identifiant unique de l'abonné (typiquement l'identité de la
            vue qui s'abonne).
        callback : callable
            Appelé avec le nouvel état à chaque publication d'un autre
            abonné.

        Returns
        -------
        callable
            Fonction à appeler pour se désabonner.
        """
        self._subscribers[subscriber_id] = callback

        def _unsubscribe() -> None:
            self._subscribers.pop(subscriber_id, None)

        return _unsubscribe

    def publish(self, origin_id: str, state: ViewportState) -> None:
        """Publie un nouvel état, reçu par tous les abonnés sauf l'origine.

        Parameters
        ----------
        origin_id : str
            Identifiant de l'abonné à l'origine du changement — n'est
            jamais notifié en retour.
        state : ViewportState
            Nouvel état de viewport à diffuser.
        """
        if self._publishing:
            # Une publication déclenchée en cascade par une notification
            # ci-dessous ne doit pas re-déclencher une autre cascade.
            return
        self._publishing = True
        try:
            for subscriber_id, callback in list(self._subscribers.items()):
                if subscriber_id != origin_id:
                    callback(state)
        finally:
            self._publishing = False
