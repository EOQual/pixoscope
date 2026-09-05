# Licences tierces — inventaire et statut

> **Avertissement.** Cet inventaire est tenu au meilleur effort, à partir
> d'une vérification directe des dépôts/paquets sources (fichier
> `LICENSE`, classifieur PyPI). Si une erreur y est identifiée, elle sera
> corrigée dès que signalée — et si nécessaire, l'implémentation
> concernée sera retirée plutôt que sa licence "arrangée" a posteriori.
> Pour signaler un problème, voir `CONTRIBUTING.md`.

## 1. Légende

- 🔴 **Bloquant** — licence absente ou copyleft incompatible, atteignable
  sans action explicite de l'utilisateur (import du socle ou de l'extra
  `enhance` standard).
- 🟡 **À surveiller** — licence non établie, mais accessible uniquement
  via un appel explicite documenté comme expérimental.
- 🟢 **Compatible** — licence permissive, compatible avec la licence MIT
  de Pixoscope.

## 2. Inventaire

### 2.1 Dépendances Python (socle et extras)

Toutes vérifiées via leur classifieur PyPI / dépôt officiel :

| Paquet | Licence | Où |
|---|---|---|
| numpy | BSD-3-Clause | socle |
| imageio | BSD-2-Clause | socle |
| tifffile | BSD-3-Clause | socle |
| zarr | MIT | socle |
| platformdirs | MIT | socle |
| loguru | MIT | socle |
| PySide6 | LGPL-3.0 (bindings officiels Qt) | socle |
| opencv-python-headless | Apache-2.0 | extra `enhance` |
| scikit-image | BSD-3-Clause | extra `enhance` |
| scipy | BSD-3-Clause | extra `enhance` |
| GDAL (`osgeo.gdal`) | MIT (X/MIT) | extra `gdal` |

🟢 Toutes compatibles avec la licence MIT de Pixoscope (la LGPL de
PySide6 concerne la bibliothèque Qt elle-même, pas le code de Pixoscope
qui s'y lie dynamiquement — pas d'obligation de "copyleft" sur ce
dépôt).

### 2.2 Algorithmes de rehaussement — sources amont

Chaque module de `pixoscope.processing` cite, dans son docstring, un
dépôt GitHub ou une publication comme référence/source d'inspiration.
Vérification directe de chacun :

| Algorithme | Fichier pixoscope | Source citée | Licence vérifiée | Statut |
|---|---|---|---|---|
| CLAHE (LAB) | `processing/clahe.py` | Implémentation directe via OpenCV, pas de code tiers | — | 🟢 |
| SUACE | `processing/suace.py` | github.com/ravimalb/suace | **BSD-2-Clause** — vérifié (fichier `LICENSE`) | 🟢 |
| Retinex | `processing/retinex.py` | Technique attribuée à Tomoya Kamata (ISS Inc., 2011), pas de dépôt cité | — | 🟢 (réimplémentation depuis la description, pas de code tiers identifié) |
| IAGCWD | `processing/iagcwd.py` | github.com/leowang7/iagcwd | **MIT** — vérifié (fichier `LICENSE`) | 🟢 |
| EPMP | `processing/epmp.py` | github.com/bigmms/entropy-preserving-mapping-prior | **Apache-2.0** — vérifié (fichier `LICENSE`) | 🟢 |
| LIME / DUAL | `processing/lime/` | github.com/pvnieo/Low-light-Image-Enhancement | **MIT** — vérifié (fichier `LICENSE`) | 🟢 |
| HE (Histogram Equalization) | `processing/he.py` | cite github.com/AndyHuang1995/Image-Contrast-Enhancement | **Aucun fichier LICENSE** dans ce dépôt | 🟢 *quand même* — l'égalisation d'histogramme est un algorithme générique non protégeable, et l'implémentation appelle directement `skimage.exposure.equalize_hist` : aucune expression originale de ce dépôt n'est reprise (voir le docstring du module) |
| **DHE** (Dynamic Histogram Equalization) | `processing/dhe.py` | cite github.com/AndyHuang1995/Image-Contrast-Enhancement | **Aucun fichier LICENSE** dans ce dépôt | 🟡 **non enregistré par défaut** |
| **Ying (2017)** | `processing/ying.py` | cite github.com/AndyHuang1995/Image-Contrast-Enhancement | **Aucun fichier LICENSE** dans ce dépôt | 🟡 **non enregistré par défaut** |

**Sur DHE et Ying** : contrairement à HE, ces deux ports reprennent une
structure de code substantielle (découpage en fonctions, logique
détaillée) du dépôt `AndyHuang1995/Image-Contrast-Enhancement`, qui ne
porte aucune licence — par défaut, cela signifie "tous droits réservés"
et aucune autorisation de redistribution n'est établie. Traitement
retenu, à l'image de la clause `vsnr` (AGPL) dans `eoqual-destriping`
mais en plus strict (une absence de licence n'accorde *aucun* droit,
contrairement à une licence copyleft qui en accorde sous conditions) :

- Le code reste dans le dépôt (utile en usage local/interne, et pour
  documenter le portage), mais n'est **jamais importé ni enregistré**
  au simple `import pixoscope` / `import pixoscope.processing`.
- Il faut appeler explicitement
  `pixoscope.processing.enable_experimental_unlicensed_algorithms()`
  pour les activer — geste délibéré, documenté comme tel dans le
  docstring de la fonction.
- **Ne pas appeler cette fonction dans un Pixoscope destiné à être
  redistribué** (PyPI, dépôt public) sans avoir obtenu l'autorisation de
  l'auteur d'origine.

## 3. Pistes de remédiation

- Contacter l'auteur du dépôt `AndyHuang1995/Image-Contrast-Enhancement`
  pour clarifier les conditions de réutilisation de DHE et Ying, ou
- Réimplémenter DHE (Ibrahim & Kong, 2007) et Ying et al. (CAIP 2017)
  directement depuis les publications scientifiques plutôt que depuis ce
  dépôt (à l'image du traitement déjà appliqué à Retinex et aux méthodes
  de `eoqual-destriping`), ce qui lèverait complètement la réserve.
