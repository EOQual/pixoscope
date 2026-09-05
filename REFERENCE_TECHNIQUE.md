# Référence technique — Pixoscope

Architecture et choix techniques. Pour l'installation et le catalogue de
fonctionnalités, voir `README.md` ; pour l'audit des licences,
`THIRD_PARTY_LICENSES.md`.

## 1. Principes directeurs

Pixoscope est un visualisateur d'images pur — pas de géoréférencement ni
de couches vectorielles (cartopy, shapely, GDAL-vecteur), hors
périmètre. Trois exigences structurent l'architecture :

1. **Jamais de chargement systématique en pleine résolution** — la
   lecture est toujours fenêtrée/pyramidale (voir §2), quelle que soit
   la taille de l'image ou du widget d'affichage.
2. **Rendu tuilé, pas un redessin complet à chaque pan/zoom** — voir §3.
3. **Découpage testable** — IHM, IO et logique métier sont séparés en
   modules indépendants (voir §7), plutôt qu'une seule classe qui
   mélangerait tout.

## 2. Stratégie mémoire pour les grosses images

### 2.1 Lecture fenêtrée

Interface commune `pixoscope.io.backend_base.ImageHandle` :
`read_region(level, x, y, w, h, bands)` — jamais l'image entière.

- **TIFF/BigTIFF** (`tifffile_backend.py`) : ouverture via
  `tifffile.imread(path, aszarr=True)`, qui expose un store zarr
  paresseux. Une fenêtre lue via `read_region` ne décode que les
  tuiles/bandes du fichier réellement recouvertes par la requête.
- **PNG/JPEG/BMP/...** (`imageio_backend.py`) : ces formats ne
  supportent aucun fenêtrage natif — la première lecture décode l'image
  entière (mise en cache en mémoire dans le handle), limite assumée et
  documentée. Pour ces formats au-delà d'un seuil de taille, une
  pyramide en cache est construite (voir §2.2) pour ne payer ce coût
  qu'une fois.
- **`.lum`** (`plugins/lum/`) : fenêtrage par **plage de lignes**
  uniquement (`LumReader.read_rows`) — le format ne permet pas de
  fenêtrage colonnes sans lire la ligne entière. Suffisant en pratique
  car une ligne est ensuite découpée en mémoire, beaucoup moins coûteux
  qu'une lecture intégrale.
- **GDAL** (`plugins/gdal/`, extra optionnel) : lecture par overview
  (`band.GetOverview(i)`) quand disponible, ou rééchantillonnage à la
  volée par GDAL (`buf_xsize`/`buf_ysize`) sinon.

### 2.2 Pyramide en cache disque

Quand un `ImageHandle` n'expose qu'un seul niveau et que l'image dépasse
2048 px de côté (`pyramid_builder.needs_pyramid_cache`), un cache de
niveaux basse résolution est construit automatiquement :

- Chemin de cache : `platformdirs.user_cache_dir("pixoscope")/pyramids/<hash>`,
  où `<hash>` encode `(chemin absolu, mtime, taille)` du fichier source
  — invalidation implicite, aucune étape manuelle.
- Construction par **bandes horizontales** (`_STRIPE_ROWS` lignes à la
  fois), jamais l'image entière en mémoire d'un coup.
- Le niveau 0 reste toujours servi par le handle d'origine (pas de
  duplication de la pleine résolution) ; seuls les niveaux réduits
  sont mis en cache, via `tifffile.memmap` (lecture ultérieure en
  lecture seule, sans charger le fichier de cache en RAM).

### 2.3 GDAL : verdict

GDAL apporte un gain réel pour (a) les COG déjà pyramidés (lecture
d'overview native, zéro calcul client) et (b) JPEG2000/ECW/MrSID, dont
les drivers GDAL savent fenêtrer nativement — contrairement à
`tifffile`/`imageio`. Pour le cas générique (TIFF/BigTIFF sans pyramide
préexistante), `tifffile`+`zarr`+cache maison (§2.2) offre une
expérience équivalente sans la lourdeur d'installation de GDAL.
**Décision : GDAL reste un extra optionnel, jamais requis.**

### 2.4 Statistiques et histogramme

Jamais calculés sur la pleine résolution. `MainWindow` déclenche, dès
l'ouverture d'un fichier, un calcul en arrière-plan
(`_compute_all_band_stats`) sur le niveau de pyramide le moins résolu
(`ImageHandle.read_overview()`), pour chaque bande — résultat mis en
cache dans `ImageDataset.stats_cache`, jamais recalculé pour un simple
changement de mapping de canaux.

## 3. Rendu — simplification assumée par rapport au plan initial

Le rendu (`pixoscope.ui.graphics_view.TiledImageView`) ne maintient
**qu'un seul raster**, couvrant la région visible (avec une marge de
35 %), plutôt qu'une mosaïque de tuiles indépendantes mises en cache
(LRU multi-tuiles). C'est ce raster unique, lu à la résolution de
pyramide adaptée au zoom courant (`core.pyramid.level_for_zoom`), qui
borne la mémoire et le volume lu depuis le disque à "environ un écran"
quelle que soit la taille du fichier — la propriété de performance
recherchée — sans la complexité d'un cache multi-tuiles.

Le rafraîchissement est différé de 80 ms après le dernier événement de
pan/zoom (évite une lecture disque à chaque incrément de molette), et
s'exécute toujours dans `QThreadPool` (voir `ui.load_worker`) — jamais
sur le thread UI.

**Limite connue** : un pan continu sur un stockage réseau lent peut
occasionner un léger flou temporaire (l'ancien raster est ré-échantillonné
par Qt en attendant le nouveau) plus perceptible qu'avec un vrai cache
multi-tuiles. Amélioration possible en v2 si ce cas d'usage se confirme.

## 4. Modèle multi-canaux

`core.image_model.ImageDataset` ne connaît que des bandes indexées.
`ChannelMapping(red, green, blue, gray)` est appliqué **au moment du
rendu de tuile** (`read_display_tile`) : seules les bandes réellement
mappées sont lues depuis le disque (`ChannelMapping.band_indices()`),
même si le fichier source a beaucoup plus de bandes physiques.
L'étirement d'affichage (`display_range` par bande, alimenté par les
statistiques ou fixé manuellement) est appliqué dans le même passage,
en `float32` borné à la tuile — jamais un upcast de l'image entière.

## 5. Détection de l'axe des bandes en TIFF

`tifffile_backend._infer_band_axis` détermine l'axe des bandes d'un
tableau 3D **par la taille des axes** (le plus petit des trois, s'il est
nettement plus petit que les deux autres et ≤ 32) plutôt que par la
chaîne `TiffPageSeries.axes` de tifffile. Choix déduit d'un bug constaté
en test : pour un TIFF multi-bandes écrit sans tag `ExtraSamples`
explicite, `axes` peut valoir `"QYX"` alors que le tableau réellement
retourné par `tifffile.imread(..., aszarr=True)` est en ordre `(Y, X, S)`
— la chaîne ne correspond pas toujours positionnellement au tableau.
L'heuristique par taille est plus robuste en pratique.

**Limite connue** : un TIFF réellement multi-dimensionnel (pile Z,
série temporelle) avec un axe non-bande de petite taille serait mal
interprété comme un axe de bandes.

## 6. Algorithmes de rehaussement — organisation

`pixoscope.processing.registry.ProcessingRegistry` est un registre à
décorateurs. Contrat commun : `f(image: np.ndarray, **params) -> np.ndarray`,
où `image` est une tuile déjà étirée en `uint8` (jamais les données
brutes) — évite la prolifération de branches par dtype.

Chaque algorithme déclare ses dépendances optionnelles (`requires`) ;
`ProcessingRegistry.available()` ne retourne que les algorithmes
réellement utilisables dans l'environnement courant — un algorithme dont
une dépendance manque est absent du menu plutôt que de faire planter
tout le paquet à l'import.

Deux algorithmes (`slow_on_large_images=True`) ont un coût qui croît
avec la taille de l'image et sont signalés comme tels dans l'IHM :

- **DHE** : la carte de corrélation locale itère pixel par pixel en
  Python (`np.corrcoef` sur une fenêtre 5×5 par pixel) — non vectorisé,
  limite assumée telle quelle.
- **LIME/DUAL** : la carte d'illumination est raffinée en résolvant un
  système linéaire creux de taille `n_pixels × n_pixels`.

Deux algorithmes (`dhe`, `ying`) ne sont pas enregistrés par défaut pour
une raison de licence amont non établie — voir `THIRD_PARTY_LICENSES.md`
§2.2 et `pixoscope.processing.enable_experimental_unlicensed_algorithms`.

## 7. Découpage en modules

| Package | Responsabilité | Dépend de |
|---|---|---|
| `pixoscope.io` | Backends de lecture, sélection auto, cache pyramide | numpy, tifffile, zarr, imageio |
| `pixoscope.core` | Modèle de données, sélection de niveau, statistiques, sync de viewport — **zéro Qt** | numpy, `pixoscope.io` |
| `pixoscope.processing` | Algorithmes de rehaussement, registre — **zéro Qt** | numpy (+ extra `enhance`) |
| `pixoscope.plugins.lum` | Lecteur `.lum` (extra `lum`, aucune dépendance tierce) | numpy |
| `pixoscope.plugins.gdal` | Backend GDAL (extra `gdal`) | GDAL |
| `pixoscope.ui` | Widgets PySide6 | PySide6, `core`, `processing` |

Règle stricte : `core`/`io`/`processing` n'importent jamais PySide6 —
testables headless, en CI, sans serveur d'affichage
(`QT_QPA_PLATFORM=offscreen` pour les tests `pixoscope.ui`).

## 8. Comment vérifier

- `scripts/make_synthetic_bigtiff.py` : génère un TIFF de test de la
  taille voulue sans le matérialiser en mémoire (utile en l'absence
  d'image de test volumineuse sous la main).
- `scripts/bench_open.py` : mesure temps d'ouverture et volume lu, pour
  comparer un backend à un autre (`PIXOSCOPE_BACKEND=...`).
- `pytest tests/` (voir `quality/run_tests.sh`) : tests headless sur
  `core`/`io`/`processing`, tests `pytest-qt` en mode `offscreen` sur
  `ui`.
