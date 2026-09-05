<p align="center">
  <img src="resources/eoqual_rounded.png" alt="EOQual" width="280">
</p>

# Pixoscope

Outil de visualisation d'images léger et performant — y compris sur de
très grosses images (TIFF/BigTIFF multi-Go) — sans dépendance
géospatiale/SIG. Fait partie d'[EOQual](https://github.com/EOQual), sans
en être une bibliothèque : Pixoscope est une application autonome (IHM +
CLI), pas un composant destiné à être importé par d'autres paquets
`eoqual-*`. Voir **[REFERENCE_TECHNIQUE.md](REFERENCE_TECHNIQUE.md)**
pour l'architecture et les choix techniques, et
**[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md)** pour l'audit des
dépendances tierces.

## Sommaire

1. [Installation](#installation)
2. [Démarrage rapide](#démarrage-rapide)
3. [Fonctionnalités](#fonctionnalités)
4. [Licence](#licence)
5. [Contribuer](#contribuer)
6. [Credits](#credits)

---

## Installation

### 1. Socle (obligatoire)

```bash
pip install git+https://github.com/EOQual/pixoscope.git
```

Installe le nécessaire pour l'essentiel de l'usage courant : ouverture
TIFF/BigTIFF (fenêtrage et pyramide via `tifffile`+`zarr`), PNG/JPEG/BMP
(`imageio`), interface PySide6, mapping multi-canaux, statistiques et
histogramme, et le socle d'algorithmes de rehaussement (linéaire,
étirement de contraste, gamma, log, sigmoïde — numpy uniquement).

### 2. Extras optionnels

| Extra | Commande | Débloque | Pourquoi optionnel |
|---|---|---|---|
| `enhance` | `pip install "pixoscope[enhance]"` | Les algorithmes de rehaussement avancés (CLAHE, HE, SUACE, Retinex, IAGCWD, DHE, EPMP, Ying, LIME/DUAL) | Dépend d'OpenCV, scikit-image, SciPy — gardés hors du socle pour rester léger |
| `lum` | `pip install "pixoscope[lum]"` | Le format `.lum` maison | Aucune dépendance tierce en réalité (numpy suffit) — extra conservé pour documenter explicitement ce format de niche |
| `gdal` | `pip install "pixoscope[gdal]"` | JPEG2000, COG déjà pyramidés, ECW/MrSID | GDAL n'apporte un gain réel que sur ces cas précis (voir `REFERENCE_TECHNIQUE.md` §2) — jamais requis sinon, et pénible à installer partout |

Sans l'extra `enhance`, les algorithmes correspondants sont simplement
absents du menu "Rehaussement" de l'IHM — aucun plantage au démarrage
(voir `pixoscope.processing.registry.ProcessingRegistry`).

**Deux algorithmes (`dhe`, `ying`) ne sont jamais activés par défaut**,
même avec l'extra `enhance` installé : leur source amont ne porte aucune
licence (voir `THIRD_PARTY_LICENSES.md` §2.2). Les activer explicitement
pour un usage local/interne :

```python
import pixoscope.processing
pixoscope.processing.enable_experimental_unlicensed_algorithms()
```

---

## Démarrage rapide

```bash
# Fenêtre vide
pixoscope

# Ouvre directement une image
pixoscope mon_image.tif

# Mode comparaison : deux images, vues synchronisables
pixoscope avant.tif apres.tif
```

---

## Fonctionnalités

- **Grosses images** : lecture fenêtrée + pyramidale (`tifffile`/`zarr`
  pour le TIFF/BigTIFF ; construction automatique d'un cache de pyramide
  pour les formats sans pyramide native) — jamais de chargement complet
  en mémoire pour l'affichage.
- **Multi-canaux** : assignation libre de n'importe quelle bande du
  fichier aux plans Rouge/Vert/Bleu affichés (ou niveaux de gris),
  changeable à la volée.
- **Statistiques et histogramme** : calculés en arrière-plan sur un
  aperçu basse résolution, jamais sur l'image pleine résolution.
- **Mode comparaison** : deux vues côte à côte, zoom/pan synchronisables.
- **Rehaussement dynamique** : socle numpy (linéaire, étirement,
  gamma, log, sigmoïde) + extra `enhance` pour les méthodes avancées —
  voir `REFERENCE_TECHNIQUE.md` pour le détail de chacune.
- **Format `.lum`** en plugin indépendant (`pixoscope[lum]`).

---

## Licence

Le code de Pixoscope est sous licence **MIT** (`license.txt`). Voir
**[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md)** pour les
dépendances tierces.

---

## Contribuer

Voir **[CONTRIBUTING.md](CONTRIBUTING.md)**.

---

## Credits

- Algorithmes de rehaussement implémentés à partir des dépôts et
  publications référencés dans `REFERENCE_TECHNIQUE.md` et
  `THIRD_PARTY_LICENSES.md` (AndyHuang1995/Image-Contrast-Enhancement,
  bigmms/entropy-preserving-mapping-prior, leowang7/iagcwd,
  pvnieo/Low-light-Image-Enhancement, ravimalb/suace).
- tifffile, zarr, imageio, PySide6, NumPy, SciPy, OpenCV, scikit-image.
