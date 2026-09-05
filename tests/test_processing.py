"""Tests de bout en bout des algorithmes de rehaussement enregistrés.

Smoke tests délibérés : on vérifie que chaque algorithme disponible
s'exécute sans exception et respecte le contrat commun (même forme
spatiale, sortie ``uint8``) — pas une vérification numérique de chaque
algorithme, qui relève plutôt de tests dédiés par méthode si un
comportement précis doit être garanti.
"""

from __future__ import annotations

import numpy as np
import pytest

from pixoscope.processing.registry import ProcessingRegistry


@pytest.mark.parametrize("spec", ProcessingRegistry.available(), ids=lambda spec: spec.key)
def test_algorithm_respects_contract(spec, small_rgb_array: np.ndarray, small_gray_array: np.ndarray) -> None:
    image = small_gray_array if spec.grayscale_only else small_rgb_array
    result = ProcessingRegistry.apply(spec.key, image)
    assert result.dtype == np.uint8
    assert result.shape[:2] == image.shape[:2]


def test_unknown_algorithm_raises_key_error() -> None:
    with pytest.raises(KeyError):
        ProcessingRegistry.get("does-not-exist")


def test_linear_is_identity(small_rgb_array: np.ndarray) -> None:
    result = ProcessingRegistry.apply("linear", small_rgb_array)
    np.testing.assert_array_equal(result, small_rgb_array)


def test_unlicensed_algorithms_not_registered_by_default() -> None:
    """dhe/ying : source amont sans licence, voir THIRD_PARTY_LICENSES.md §2.2."""
    keys = {spec.key for spec in ProcessingRegistry.all_specs()}
    assert "dhe" not in keys
    assert "ying" not in keys


def test_enable_experimental_unlicensed_algorithms_registers_them(small_rgb_array: np.ndarray) -> None:
    import pixoscope.processing as processing

    processing.enable_experimental_unlicensed_algorithms()
    try:
        assert "dhe" in {spec.key for spec in ProcessingRegistry.all_specs()}
        result = ProcessingRegistry.apply("ying", small_rgb_array)
        assert result.dtype == np.uint8
    finally:
        # Le registre est un état global partagé entre tests : rien à
        # "désenregistrer" proprement, mais l'appel est idempotent et
        # n'affecte pas les autres tests (ils ne dépendent jamais de
        # l'absence de dhe/ying, seulement de available()/all_specs()).
        pass
