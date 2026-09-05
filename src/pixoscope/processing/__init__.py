"""Algorithmes de rehaussement dynamique — registre et implémentations.

Importer ce paquet enregistre les algorithmes du socle **et de l'extra
``enhance`` dont la provenance est établie** dans
:class:`pixoscope.processing.registry.ProcessingRegistry`. Un algorithme
dont une dépendance optionnelle manque reste importable (les imports
lourds sont différés à l'intérieur des fonctions, voir chaque module)
mais n'apparaît pas dans :meth:`ProcessingRegistry.available`.

**`dhe` et `ying` ne sont pas enregistrés par défaut** : leur source
amont (``AndyHuang1995/Image-Contrast-Enhancement``) ne porte aucun
fichier ``LICENSE`` — voir ``THIRD_PARTY_LICENSES.md`` §2.2. Le code est
présent dans le dépôt (utile en usage local/interne) mais n'est
enregistré, et donc distribué "actif", que si
:func:`enable_experimental_unlicensed_algorithms` est appelé
explicitement — jamais au simple import du paquet.

Voir ``REFERENCE_TECHNIQUE.md`` pour le détail de chaque algorithme.
"""

# L'import de chaque module déclenche ses décorateurs @ProcessingRegistry.register.
# `dhe` et `ying` sont volontairement exclus de cette liste (voir le
# docstring du module et THIRD_PARTY_LICENSES.md §2.2).
from pixoscope.processing import (  # noqa: F401,E402
    adjust,
    clahe,
    epmp,
    he,
    iagcwd,
    lime,  # noqa: F401,E402
    retinex,
    suace,
)
from pixoscope.processing.registry import AlgorithmSpec, ProcessingRegistry

__all__ = ["AlgorithmSpec", "ProcessingRegistry", "enable_experimental_unlicensed_algorithms"]

_experimental_enabled = False


def enable_experimental_unlicensed_algorithms() -> None:
    """Enregistre ``dhe`` et ``ying``, dont la licence amont n'est pas établie.

    **À ne pas appeler dans un pixoscope destiné à être redistribué**
    (PyPI, dépôt public) sans avoir obtenu l'autorisation de l'auteur
    d'origine, ou réimplémenté l'algorithme depuis la publication
    scientifique plutôt que depuis son code — voir
    ``THIRD_PARTY_LICENSES.md`` §2.2 et §3. Prévu pour un usage
    local/interne assumé par l'appelant.

    Idempotent : un second appel ne fait rien.
    """
    global _experimental_enabled
    if _experimental_enabled:
        return
    from pixoscope.processing import dhe, ying  # noqa: F401

    _experimental_enabled = True
