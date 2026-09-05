# Contribuer à Pixoscope

Les contributions sont bienvenues : signalement de bug, correction,
nouveau backend d'IO, nouvel algorithme de rehaussement, amélioration de
la documentation.

## Signaler un bug

Ouvrez une issue avec :
- le fichier concerné et, si possible, un exemple minimal (image de
  test, extrait de code) ;
- le résultat obtenu vs. attendu ;
- `PIXOSCOPE_BACKEND` utilisé le cas échéant.

## Ajouter un backend de lecture

Convention du projet : implémenter `pixoscope.io.backend_base.ImageBackend`
et `ImageHandle` (voir `pixoscope/io/tifffile_backend.py` pour un
exemple complet), puis l'ajouter à
`pixoscope.io.backend_registry._default_backends()`. Un backend qui
dépend d'une bibliothèque lourde optionnelle doit échouer par
`ImportError` à l'import de son module (pas ailleurs) — c'est ce que la
registry utilise pour le proposer ou non.

## Ajouter un algorithme de rehaussement

Convention du projet : une méthode = une fonction
`f(image: np.ndarray, **params) -> np.ndarray` où `image` est une tuile
`uint8` déjà étirée (voir `pixoscope/processing/registry.py`), placée
dans `src/pixoscope/processing/<methode>.py`, enregistrée via
`@ProcessingRegistry.register(key, label, requires=(...))`. Les imports
de dépendances optionnelles (`cv2`, `skimage`, `scipy`) doivent être
différés à l'intérieur des fonctions, jamais en tête de module — un
algorithme dont une dépendance manque doit rester silencieusement absent
du menu, jamais faire planter l'import du paquet.

**Licence — condition de fusion, pas une formalité.** Toute
implémentation reprenant ou s'inspirant de près d'un code publié
ailleurs doit citer sa source et sa licence *vérifiée* (fichier
`LICENSE` du dépôt d'origine, pas une supposition) dans le docstring du
module — voir `dhe.py`/`ying.py` pour le traitement d'une source sans
licence établie (non enregistrée par défaut, voir
`THIRD_PARTY_LICENSES.md` §2.2), et `THIRD_PARTY_LICENSES.md` pour le
principe déjà appliqué à l'existant.

- Une implémentation dont la licence d'origine n'est pas identifiée, ou
  copyleft forte, **ne sera pas enregistrée par défaut**.
- Réimplémenter l'algorithme depuis sa publication scientifique
  (équations), sans reprendre le code d'un dépôt tiers, lève toute
  réserve de licence.

## Tests

Toute nouvelle méthode/backend doit être accompagné(e) d'un test dans
`tests/` (smoke test minimal acceptable : l'appel ne lève pas
d'exception et respecte le contrat de forme/dtype). Voir
`tests/conftest.py` pour les fixtures disponibles — les tests ne doivent
dépendre d'aucun fichier externe au dépôt (toutes les images de test
sont générées à la volée).

## Documentation

Toute méthode/backend nouveau ou modifié doit rester reflété dans
`README.md` et `REFERENCE_TECHNIQUE.md` — les docstrings seules ne
suffisent pas, ces deux documents sont la carte d'ensemble du projet.
