# Change Log

## Version History

0.1.0 (2026-09-04)

:   -   Première version : visualisateur d'images, sans
        géoréférencement/vecteur.
    -   Lecture fenêtrée + pyramidale pour les grosses images
        (TIFF/BigTIFF via `tifffile`+`zarr` ; cache de pyramide
        automatique pour les formats sans pyramide native).
    -   Modèle multi-canaux : assignation libre des bandes aux plans
        R/V/B, indépendante de la disposition physique du fichier.
    -   Statistiques et histogramme calculés en arrière-plan sur un
        aperçu basse résolution, jamais sur l'image pleine résolution.
    -   Mode comparaison (deux vues, zoom/pan synchronisables).
    -   Socle d'algorithmes de rehaussement (16 méthodes, dont 2 non
        enregistrées par défaut pour raison de licence amont — voir
        `THIRD_PARTY_LICENSES.md`).
    -   Format `.lum` en plugin indépendant (extra `pixoscope[lum]`),
        GDAL en extra optionnel (`pixoscope[gdal]`, jamais requis).
    -   Documentation : `README.md`, `REFERENCE_TECHNIQUE.md`,
        `THIRD_PARTY_LICENSES.md`, `CONTRIBUTING.md`.

## Évolutions futures

- Cache multi-tuiles pour le rendu (voir `REFERENCE_TECHNIQUE.md` §3) si
  le raster-unique-viewport actuel montre ses limites en usage réel.
- Clarifier la licence amont de DHE/Ying, ou les réimplémenter depuis
  leurs publications scientifiques (voir `THIRD_PARTY_LICENSES.md` §3).
- Fenêtrage colonnes natif pour `.lum` (actuellement lignes seulement).
