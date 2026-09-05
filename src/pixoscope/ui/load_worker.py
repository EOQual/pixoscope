"""Exécution en arrière-plan (ouverture, lecture de tuile, statistiques...).

Règle d'architecture : **aucune I/O potentiellement lente (ouverture de
fichier, lecture d'une région, calcul de statistiques, algorithme de
rehaussement) ne doit jamais s'exécuter sur le thread UI**. Ce module
fournit un unique wrapper générique, ``CallableWorker``, exécuté dans
``QThreadPool.globalInstance()`` — plutôt qu'une classe ``QRunnable``
dédiée par type de tâche, pour éviter la duplication.

Utiliser :func:`run_in_background` plutôt que d'instancier
``CallableWorker`` directement : un ``QRunnable`` n'est **pas** un
``QObject`` parenté par Qt — rien n'empêche le ramasse-miettes Python de
libérer le wrapper (et ses signaux) avant la fin de l'exécution si
aucune référence Python n'est conservée. ``run_in_background`` retient
une référence forte le temps de l'exécution, ce qui évite ce piège
classique.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from loguru import logger
from PySide6.QtCore import QObject, QRunnable, Signal

from pixoscope.ui.qt_utils import global_thread_pool


class WorkerSignals(QObject):
    """Signaux émis par un :class:`CallableWorker`.

    Attributes
    ----------
    result : Signal
        Émis avec la valeur de retour de l'appel, en cas de succès.
    error : Signal
        Émis avec un message d'erreur, en cas d'exception.
    finished : Signal
        Émis systématiquement à la fin, succès ou échec.
    """

    result = Signal(object)
    error = Signal(str)
    finished = Signal()


class CallableWorker(QRunnable):
    """Exécute ``fn(*args, **kwargs)`` dans le pool de threads Qt.

    Parameters
    ----------
    fn : callable
        Fonction à exécuter en arrière-plan. Doit être thread-safe
        (aucun accès direct à un widget Qt).
    *args, **kwargs
        Transmis à ``fn``.

    Attributes
    ----------
    signals : WorkerSignals
    """

    def __init__(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self._fn = fn
        self._args = args
        self._kwargs = kwargs
        self.signals = WorkerSignals()

    def run(self) -> None:
        """Point d'entrée exécuté par le pool de threads."""
        try:
            result = self._fn(*self._args, **self._kwargs)
        except Exception as exc:  # noqa: BLE001 - on relaie toute erreur à l'UI
            logger.opt(exception=exc).debug("Erreur dans un CallableWorker")
            self._safe_emit(self.signals.error, str(exc))
        else:
            self._safe_emit(self.signals.result, result)
        finally:
            self._safe_emit(self.signals.finished)

    @staticmethod
    def _safe_emit(signal: Signal, *args: Any) -> None:
        """Émet un signal en ignorant le cas où sa cible a déjà été détruite.

        Le widget qui a lancé cette tâche (une vue fermée, une fenêtre
        de comparaison retirée) peut disparaître avant la fin de
        l'exécution en arrière-plan — un ``RuntimeError`` de shiboken à
        ce moment n'est pas une erreur applicative, juste un résultat
        devenu inutile.
        """
        try:
            signal.emit(*args)
        except RuntimeError:
            logger.trace("Signal ignoré : la cible a été détruite avant la fin du traitement")


#: Références fortes vers les workers en cours, pour éviter qu'ils ne
#: soient collectés avant la fin (voir le docstring du module).
_active_workers: set[CallableWorker] = set()


def _guard_against_deleted_target(callback: Callable[..., None]) -> Callable[..., None]:
    """Enveloppe ``callback`` pour ignorer un ``RuntimeError`` de shiboken.

    Un résultat de tâche de fond (lecture de tuile, statistiques...)
    peut arriver après que le widget qui l'a demandé (une vue fermée, un
    panneau retiré du mode comparaison) a déjà été détruit — la
    livraison du signal est différée au thread UI, donc ce cas ne peut
    pas être intercepté côté émission (voir ``CallableWorker._safe_emit``,
    qui ne protège que l'émission elle-même, pas le traitement différé
    par la boucle d'événements). Sans cette garde, l'exception remonte
    telle quelle dans la boucle d'événements Qt.
    """

    def _wrapped(*args: Any) -> None:
        try:
            callback(*args)
        except RuntimeError:
            logger.trace("Callback de tâche de fond ignoré : la cible a été détruite")

    return _wrapped


def run_in_background(
    fn: Callable[..., Any],
    *args: Any,
    on_result: Callable[[Any], None] | None = None,
    on_error: Callable[[str], None] | None = None,
    **kwargs: Any,
) -> CallableWorker:
    """Soumet ``fn(*args, **kwargs)`` au pool de threads global.

    Note pour les appelants : passer ``on_result``/``on_error`` comme un
    lambda avec un paramètre par défaut (``lambda x, y=valeur_capturée: ...``,
    l'idiome standard pour figer la valeur d'une variable de boucle) fait
    échouer l'inférence de type de mypy sur ce lambda précis — limitation
    connue, sans impact à l'exécution. Les appels concernés portent un
    ``# type: ignore[misc]`` local plutôt qu'un changement de signature ici.

    Parameters
    ----------
    fn : callable
    *args
    on_result : callable, optional
        Connecté au signal ``result``.
    on_error : callable, optional
        Connecté au signal ``error``.
    **kwargs

    Returns
    -------
    CallableWorker
        Généralement inutile de conserver la référence retournée : elle
        est déjà maintenue en vie en interne jusqu'à la fin de
        l'exécution.
    """
    worker = CallableWorker(fn, *args, **kwargs)
    _active_workers.add(worker)
    worker.signals.finished.connect(lambda: _active_workers.discard(worker))
    if on_result is not None:
        worker.signals.result.connect(_guard_against_deleted_target(on_result))
    if on_error is not None:
        worker.signals.error.connect(_guard_against_deleted_target(on_error))
    global_thread_pool().start(worker)
    return worker
