# PanelWeave — `panelrecon`

Reconstruction automatique de l'**image originale complète d'un panel de manhwa/webtoon** à partir
d'une vidéo où ce panel est animé (translation, zoom, easing) par-dessus un fond fixe ou flou.
Chaque frame n'en montre qu'une partie : les observations sont recalées dans un repère commun puis
fusionnées.

```
vidéo → séquences → segmentation panel/fond → mouvement inter-frames → recalage global
      → warp canonique → fusion médiane pondérée → couverture → recadrage → PNG RGBA + rapport
```

## État d'avancement

| Phase | Contenu | État |
|---|---|---|
| 1 | Squelette, configuration, modèles, `video_io`, détection matérielle, CLI (`inventory`) | ✅ |
| 2 | Générateur synthétique avec vérité terrain, métriques d'évaluation, reconstruction oracle | ✅ |
| 3 | Mouvement (SIFT/ORB + RANSAC + ECC, replis flot dense et corrélation log-polaire), chaînage | ✅ |
| 4 | Mosaïque, fusion médiane pondérée par tuiles, couverture, recadrage, export, pipeline | ✅ |
| 5 | Découpage en séquences (coupes, fondus), segmentation panel/fond | ✅ |
| 6 | Ajustement global des poses (images clés, liens à longue portée) | ✅ |
| 7 | Contrôle qualité (SSIM frame/panel, couverture, netteté, verdicts), refusion | ✅ |
| 8 | Interface PyQt5, application macOS (.app / .dmg) | ✅ |
| 9 | Traitement multiprocessus (tronçons, séquences, lots) | ✅ |
| 9 | Modules optionnels (LoFTR, RAFT, SAM 2) | à faire |

## Plateforme cible

macOS sur Apple Silicon (arm64 natif, pas de Rosetta), Python **3.11 ou 3.12**, macOS ≥ 14
(exigé par les wheels arm64 de PyAV 18). Le code ne dépend d'aucune bibliothèque propre à
Linux/Windows ni de CUDA ; il est aussi testé sous Linux.

Toutes les dépendances de `requirements.txt` et `requirements-dev.txt` ont été vérifiées
installables en **wheels binaires** `macosx_*_arm64` pour CPython 3.11 et 3.12
(`pip download --platform macosx_14_0_arm64 --only-binary=:all:`) : aucune compilation locale.

Choix de dépendances notables :

* `opencv-python-headless` (et non `opencv-python`) : la variante non headless embarque ses
  propres plugins Qt qui entrent en conflit avec PyQt5.
* `scenedetect[opencv-headless]==0.6.7.1` : PySceneDetect 0.7.x dépend en dur de
  `opencv-python` ; la 0.6.7.1 propose l'extra headless.
* OpenCV épinglé en 4.14 (SIFT inclus) plutôt qu'en 5.0, pour la stabilité de l'API
  (`findTransformECC`, `estimateAffinePartial2D`).

## Installation

```bash
# Python arm64 natif (vérifier : python3 -c "import platform; print(platform.machine())" → arm64)
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-dev.txt     # ou requirements.txt sans les outils de test
pip install -e . --no-deps              # rend la commande `panelrecon` disponible
```

## Utilisation

```bash
# Écrire la configuration par défaut (point de départ d'un profil)
python -m panelrecon.cli --write-default-config profils/defaut.json

# Reconstruction d'un lot (mode par défaut)
python -m panelrecon.cli --input ~/Videos/manhwa --output ~/Videos/sortie --config profils/defaut.json

# Inventaire seul : découverte récursive, décodage complet en flux, statistiques temporelles
python -m panelrecon.cli --mode inventory --input ~/Videos/manhwa --output ~/Videos/sortie
```

Le mode `reconstruct` découpe chaque vidéo en séquences (un panel par séquence) et produit, dans
`<sortie>/<vidéo>/`, pour chaque séquence `k` :

* `<vidéo>_seq<k>_<début>-<fin>.png` : panel reconstruit, PNG RGBA, limité à l'emprise du panel
  (le fond est exclu) ; les pixels jamais observés sont transparents (alpha = 0), aucun contenu
  n'est inventé ;
* `…_coverage.png` (16 bits, nombre d'observations par pixel) et `…_coverage_color.png` ;
* `…json` : séquence, taille, échelle canonique, recadrage, statistiques de couverture, recalage
  (transformations, estimations), segmentation (rectangle du panel dans le repère canonique).

`<vidéo>_sequences.json` liste les séquences et les transitions (coupe, fondu, perturbation
réunie) avec les frames écartées et leurs raisons. `batch_report.json` résume le lot ; une vidéo en échec est rapportée avec sa trace sans arrêter
les suivantes.

Le mode `inventory` écrit `inventory.json` (métadonnées, frames décodées/conservées, erreurs de
décodage, intervalles min/médian/max, détection de fréquence variable) et `panelrecon.log`.
Une vidéo illisible est journalisée avec sa trace sans interrompre le lot.
Codes de sortie : `0` succès, `1` au moins une vidéo en échec, `2` erreur d'usage/configuration,
`130` interruption.

## Application (interface graphique)

```bash
python -m panelrecon.gui            # ou : panelrecon-gui (après pip install -e .)
```

Glissez-déposez des vidéos ou des dossiers dans la fenêtre, puis **Lancer**. Pour chaque vidéo :
progression, nombre de panels et verdicts. En sélectionnant une vidéo, les panels reconstruits
s'affichent (vignettes colorées selon le verdict, visionneuse avec zoom à la molette et
déplacement, carte de couverture, aperçu de n'importe quelle frame de la vidéo). Tous les
paramètres sont modifiables dans le panneau **Paramètres** (bornes vérifiées, profils JSON,
mémorisés entre deux sessions). Le calcul s'exécute hors du thread de l'interface (pool de
processus) ; **Annuler** l'interrompt proprement.

### Construire l'application Mac

Sur un Mac Apple Silicon (Python 3.11/3.12 arm64 natif) :

```bash
./packaging/macos/build_app.sh
```

Le script crée un environnement isolé, installe les dépendances, construit
`dist/PanelRecon.app` (PyInstaller, arm64), la signe en ad hoc, vérifie qu'elle traite une vidéo
de test, puis produit `dist/PanelRecon.dmg`. Au premier lancement, macOS peut demander de
l'ouvrir via clic droit → **Ouvrir** (application non notariée). L'exécutable accepte aussi
`--cli` pour un traitement par lot sans interface :
`PanelRecon.app/Contents/MacOS/PanelRecon --cli -i dossier -o sortie`.

## Contrôle qualité (phase 7)

Pour chaque panel, le résultat est reprojeté dans les frames de la séquence et comparé à
chacune (SSIM sur la zone couverte, à 640 px). Une frame mal recalée ou floue (flou de
mouvement) obtient un SSIM faible : elle est écartée et le panel est refusionné sans elle si
l'accord s'améliore. Le rapport JSON de chaque panel contient le SSIM moyen et minimal, le
SSIM de chaque frame évaluée, la couverture du rectangle recadré, la netteté (variance du
Laplacien), les taux d'inliers et l'erreur de reprojection, et un **verdict** :

* **OK** : tous les seuils `quality.*` sont respectés ;
* **À VÉRIFIER** (sous-dossier `a_verifier/`) : SSIM moyen < 0,92, frame gardée < 0,80,
  couverture < 90 %, erreur de reprojection élevée, panel peu net ou recalage interrompu ;
* **ÉCHEC** (sous-dossier `echec/`) : SSIM moyen < 0,75, couverture < 50 % ou aucune frame
  comparable.

Les raisons sont listées dans le rapport et dans l'interface.

## Parallélisme

Le découpage en séquences est séquentiel par nature (chaque frame est recalée sur la
précédente) ; il est donc parallélisé **par tronçons** :

1. **Index des frames** : les paquets sont lus sans décodage (≈ 0,1 s pour 3 min) ; l'indice
   d'une frame est le rang de son pts. Il ne dépend pas du point de départ, ce qui permet de
   décoder n'importe quelle plage en se positionnant sur l'image clé précédente
   (`VideoReader(..., frame_index=...)`), avec la même décimation qu'en lecture continue.
2. **Tronçons** : la vidéo est coupée en tronçons d'au plus `runtime.chunk_seconds` (5 min),
   et au moins en autant de tronçons que de processus tant que chacun dure
   `runtime.min_chunk_seconds` (30 s). Chaque tronçon est découpé et recalé dans un processus
   séparé.
3. **Raccord** : à chaque frontière, la dernière frame du tronçon précédent est recalée sur la
   première du suivant. Si c'est le même panel (mouvement accepté, histogrammes corrélés,
   cohérence structurelle ≥ `scenes.dissolve_min_score`), les deux séquences sont réunies (poses
   composées, repère canonique recalculé) ; sinon la frontière est enregistrée comme une coupe,
   avec sa raison, dans `<vidéo>_sequences.json`.
4. **Séquences** : les panels sont indépendants ; chacun (segmentation, fusion, export) est
   traité par un processus, qui ne décode que la plage de sa séquence.
5. **Lots** : un seul pool de processus est partagé par toutes les vidéos du lot, et plusieurs
   vidéos sont orchestrées en même temps : tronçons et séquences de toutes les vidéos se
   répartissent sur les cœurs.

Les processus sont créés par `spawn` (fonctions de module, arguments sérialisables) ; chacun
limite ses threads internes (OpenCV, fusion par tuiles) à sa part des cœurs. L'annulation passe
par un `Event` partagé, vérifié entre les frames. Avec `runtime.num_workers = 1`, tout s'exécute
dans le processus principal, sans tronçons. Sans index exploitable (repli OpenCV, conteneur sans
pts), la vidéo est traitée séquentiellement en trois passes.

## Découpage en séquences et segmentation (phase 5)

**Découpage** (`core/scene_split.py`, en flux, recalage effectué au passage). Une paire de frames
consécutives est *en changement* si : la corrélation des histogrammes HSV chute
(`scenes.histogram_min_correlation`), le mouvement est rejeté (effondrement des inliers, incohérence
structurelle), le score de cohérence entre la frame et celle située `dissolve_lag` frames avant
(recalées) chute (changement progressif), ou PySceneDetect signale une coupure non contredite par un
recalage de forte cohérence (le ContentDetector réagit aussi aux déplacements rapides d'un même
panel). Une suite de paires en changement forme une transition : ses frames **intérieures** (mélanges
d'un fondu) sont écartées ; une transition d'une seule paire est une coupe franche ; si la frame qui
suit se recale sur le panel d'avant (frame parasite, flash), les deux morceaux sont réunis.

Résultats : fondu de 6 frames → séquences 0–19 et 26–45, les 6 frames de mélange écartées
exactement ; coupe franche → 0–19 et 20–39 ; aucun faux découpage sur les 10 scénarios à panel unique
(dont déplacement rapide, aplats, fréquence variable, frames dupliquées).

**Segmentation** (`core/segmentation.py`). Le panel étant rigide et recalé, son emprise dans le
repère canonique est un rectangle fixe, estimé une fois par séquence en cumulant toutes les frames
(canevas d'analyse à l'échelle réduite) :

1. rectangle initial = boîte de la plus grande composante des points régulièrement **nets**
   (variance locale du Laplacien ; le fond est un agrandissement flou) ;
2. chaque côté est étendu jusqu'à la limite observée (panel plus grand que l'écran, zones en aplats
   sans détail), **sauf** s'il porte une bordure (ligne nette continue : limite panel/fond) ou si la
   bande au-delà se comporte comme du fond (variation temporelle nettement supérieure à celle des
   zones plates du panel quand le cadrage bouge) ;
3. chaque côté délimitant est recalé sur le maximum de gradient de l'image moyenne (la preuve de
   netteté déborde d'une demi-fenêtre) ;
4. le rectangle est projeté dans chaque frame : masques cohérents dans le temps par construction.

Le canevas de la mosaïque est restreint à cette emprise. Le segmenteur image par image demandé
(`ClassicSegmenter`, interface `PanelSegmenter` : netteté, cohérence de mouvement, morphologie,
`minAreaRect`, lissage temporel) est fourni pour l'usage sans recalage (prévisualisation, SAM 2 en
phase 9 derrière la même interface).

Résultats **sans aucun masque de vérité terrain** (découpage + recalage + segmentation + fusion) :
SSIM 0,966 à 0,997 selon le scénario, couverture réelle ≥ 0,991, **aucun pixel de fond** compté
comme couvert (contre 14 % à 41 % de surestimation sans segmentation en phase 4).

## Mosaïque et fusion (phase 4)

`core/mosaic.py` :

1. **Canevas** : englobe toutes les frames recalées dans le repère canonique (échelle de la frame la
   plus zoomée) ; `CanvasTooLargeError` explicite au-delà de `mosaic.max_canvas_megapixels`.
2. **Observations** : chaque frame est warpée (Lanczos4, ou bicubique) sur sa seule emprise, avec un
   masque de validité = panel ∩ hors bande de bord d'écran ∩ hors zones d'exclusion, érodé du support
   du noyau ; un pixel n'est valide que s'il est entièrement couvert par la zone valide. Pile
   `(n, H, W, 4)` uint8 (BGR + poids de bord quantifié), en mémoire ou en memmap sur disque au-delà
   de `mosaic.in_memory_stack_mb` (dossier temporaire supprimé ensuite).
3. **Fusion par tuiles** (`mosaic.tile_size`) : médiane pondérée vectorisée par canal ; poids =
   rampe de distance au bord du masque × (1 / échelle frame→canevas)^p (`scale_weight_power`, 2 par
   défaut : un pixel de frame couvrant 2×2 pixels de canevas pèse 4 fois moins). Les observations
   minoritaires (sous-titre fixe, artefact) sont rejetées.
4. **Couverture et recadrage** : couverture = nombre d'observations valides ; alpha = 255 si
   couverture ≥ `min_coverage` ; recadrage `bbox` (rectangle englobant, trous transparents) ou
   `covered` (rognage glouton jusqu'à couverture totale).

`core/pipeline.py` orchestre une vidéo : passe 1 recalage sur images réduites, passe 2 décodage
natif et fusion, puis export ; progression `(fraction, message)` et annulation coopérative.

Résultats (recalage estimé en phase 3 + fusion, masques de panel exacts de la vérité terrain,
vidéos encodées crf 14), comparés au panel original rééchantillonné à l'échelle canonique :

| Scénario | SSIM | PSNR | couverture réelle | surestimation |
|---|---|---|---|---|
| static | 0,978 | 32,7 dB | 1,000 | 0 |
| pan_horizontal / pan_vertical | 0,989 / 0,979 | 35,7 / 32,4 dB | 1,000 / 1,000 | 0 |
| zoom_in / zoom_out | 0,967 / 0,966 | 30,7 / 30,7 dB | 1,000 / 1,000 | 0 |
| pan_zoom_eased | 0,981 | 33,3 dB | 0,992 | 0 |
| duplicates | 0,989 | 35,8 dB | 1,000 | 0 |
| crossfade (2 plans) | 0,989 / 0,975 | 35,8 / 31,3 dB | 1,000 / 1,000 | 0 |
| subtitles | 0,983 | 35,3 dB | 1,000 | 0 |
| flat_texture | 0,997 | 40,2 dB | 1,000 | 0 |
| short / vfr | 0,982 / 0,981 | 33,6 / 33,2 dB | 0,996 / 0,992 | 0 |

Ces valeurs égalent ou dépassent la reconstruction oracle à poses exactes de la phase 2 (la
pondération par l'échelle favorise les frames les plus zoomées : `zoom_in` 0,967 contre 0,964).

## Mouvement et recalage (phase 3)

```bash
# Recale chaque vidéo comme une séquence unique (le découpage en séquences arrive en phase 5)
python -m panelrecon.cli --mode register -i synth/zoom_in.mp4 -o sortie
```

Écrit `<vidéo>_registration.json` : transformations `frame → repère canonique`, estimations
acceptées (méthode, inliers, RMS, score de cohérence) et rejetées avec leur raison.

**Estimation inter-frames** (`core/motion.py`, sur l'image réduite) — cascade, chaque étage n'étant
tenté que si le précédent est rejeté :

1. SIFT (ou ORB) restreint au masque du panel érodé, sur image à contraste étiré ; seconde détection
   à seuil de contraste abaissé si la frame est peu texturée ; kNN + ratio de Lowe ; RANSAC
   (`estimateAffinePartial2D`, graine fixée) ; moindres carrés sur les inliers avec rotation
   régularisée (Umeyama pondéré, `b / (1 + λ)`), deux passes ;
2. flot de Farneback (aller et retour) échantillonné sur le point le plus structuré de chaque
   cellule, vecteurs filtrés par cohérence aller-retour, puis même ajustement robuste ;
3. corrélation de phase log-polaire (échelle, rotation) sur spectre carré, puis corrélation de phase
   (translation).

Chaque candidat est raffiné par ECC (`MOTION_AFFINE` reprojeté sur une similarité, correction
bornée) puis **validé** : bornes de rotation et d'échelle, recouvrement minimal, et score de
cohérence structurelle = médiane par tuiles de la NCC des normes de gradient. Ce score sépare
nettement un alignement correct (≥ 0,985 sur tous les scénarios) d'une erreur de 2 px (≤ 0,92), y
compris sur aplats, et la médiane le rend insensible aux sous-titres fixes. Les replis ne sont
acceptés que si l'ECC converge. Une paire rejetée est journalisée avec la raison de chaque étage.

**Recalage** (`core/registration.py`) : chaînage sur la dernière frame acceptée ; une frame rejetée
est exclue et la suivante est estimée par rapport à l'ancre ; au-delà de
`registration.max_consecutive_failures` échecs consécutifs, la séquence est déclarée interrompue.
Les transformations sont converties en coordonnées natives (conjugaison par le facteur de proxy) et
exprimées dans le repère canonique : la frame la plus zoomée est à l'échelle 1.

Résultats sur les vidéos synthétiques encodées (erreurs après ajustement de jauge, toutes frames) :

| Scénario | translation max (px) | coins max (px) | échelle max |
|---|---|---|---|
| static | 0,001 | 0,002 | 0,000 % |
| pan_horizontal | 0,061 | 0,132 | 0,020 % |
| pan_vertical | 0,193 | 0,238 | 0,029 % |
| zoom_in / zoom_out | 0,022 / 0,016 | 0,025 / 0,215 | 0,003 / 0,056 % |
| pan_zoom_eased | 0,081 | 0,191 | 0,037 % |
| duplicates | 0,049 | 0,069 | 0,008 % |
| crossfade (2 plans) | 0,047 / 0,012 | 0,062 / 0,043 | 0,008 / 0,009 % |
| subtitles | 0,247 | 0,761 | 0,159 % |
| flat_texture | 0,365 | 0,792 | 0,128 % |
| short / vfr | 0,007 / 0,027 | 0,024 / 0,154 | 0,005 / 0,037 % |

Seuils de la spécification : translation < 1 px, échelle < 0,5 %. La dérive du chaînage est visible
sur `subtitles` et `flat_texture` (coins à ~0,8 px) : c'est l'objet de la phase 6.

## Vidéos synthétiques de référence (phase 2)

```bash
python -m panelrecon.synth_cli --list                                  # scénarios disponibles
python -m panelrecon.synth_cli --output synth --all --oracle           # génère + évalue l'oracle
python -m panelrecon.synth_cli --output synth -s zoom_in --panel-image mon_panel.png
```

Chaque scénario produit `<nom>.mp4|mkv`, `<nom>_panel<k>.png` (panel original) et
`<nom>_ground_truth.json` : pour chaque frame, la similarité exacte `panel → écran`, l'instant de
présentation, le plan d'appartenance (ou le mélange pour une frame de fondu) et l'indicateur de
duplication ; la boîte du sous-titre incrusté le cas échéant.

| Scénario | Cas limite couvert |
|---|---|
| `static` | panel entièrement visible, séquence statique |
| `pan_horizontal`, `pan_vertical` | translation pure (panel plus large / plus étroit que l'écran) |
| `zoom_in`, `zoom_out` | zoom (avec léger recentrage) |
| `pan_zoom_eased` | translation + zoom, 3 poses clés, easing `ease_in_out`, fond qui suit le cadrage |
| `duplicates` | chaque pose répétée 2 fois (animation à 12,5 i/s dans une vidéo à 25 i/s) |
| `crossfade` | deux panels séparés par un fondu enchaîné de 6 frames |
| `subtitles` | sous-titre fixe incrusté |
| `flat_texture` | panel en aplats, très peu de points d'intérêt |
| `short` | séquence de 4 frames |
| `vfr` | fréquence d'images variable (conteneur mkv, pts à la milliseconde) |

Le rendu applique une pré-réduction INTER_AREA composée exactement avec le warp résiduel
(anti-repliement sans biais géométrique) ; l'exactitude de la vérité terrain est vérifiée de façon
indépendante par le centroïde de taches gaussiennes (< 0,05 px, en réduction, agrandissement et
rotation).

`panelrecon/core/evaluation.py` fournit les mesures que les phases suivantes devront satisfaire
(`ReferenceThresholds` : translation < 1 px, échelle < 0,5 %, SSIM > 0,95, couverture ≥ 98 %,
surestimation ≤ 1 %) :

* `pose_errors` : erreurs des poses `frame → canevas` estimées, après ajustement par moindres carrés
  de la jauge `panel → canevas` (le repère du canevas est arbitraire) ; erreurs en pixels écran ;
* `pairwise_error` : erreur d'un mouvement inter-frames ;
* `masked_ssim`, `evaluate_mosaic` : SSIM/PSNR de la reconstruction contre le panel original
  rééchantillonné dans le canevas, couverture réelle et couverture surestimée ;
* `oracle_mosaic` : fusion médiane avec les poses **exactes** — borne supérieure de référence.

Résultats de l'oracle sur les vidéos encodées (crf 14) : SSIM de 0,964 (`zoom_in`) à 0,996
(`flat_texture`), couverture ≥ 0,996, surestimation 0.

## Configuration

Tout paramètre passe par `PipelineConfig` (`panelrecon/core/config.py`), sérialisée en JSON strict :
clé inconnue, type incompatible ou valeur hors bornes → `ConfigError`. Chaque champ porte ses bornes,
ses choix et un texte d'aide (réutilisés par l'interface en phase 8). Sections actuelles :

* `video` : extensions, récursivité, repli OpenCV, `frame_step`, `max_fps` (sur timestamps),
  threads de décodage, rotation d'affichage, tolérance aux paquets corrompus ;
* `preprocess` : côté long de la version réduite pour le mouvement (960 px), zones d'exclusion
  relatives (sous-titres, logos) ;
* `motion` : détecteur, nombre de points, étirement de contraste et détection « peu texturée »,
  ratio de Lowe, paramètres RANSAC, inliers et taux minimaux, érosion du masque, régularisation et
  borne de rotation, variation d'échelle maximale, ECC (itérations, epsilon, filtre, correction
  maximale, exigence pour les replis), score de cohérence minimal et taille des tuiles,
  recouvrement minimal, replis (flot, corrélation de phase) et leurs seuils ;
* `registration` : nombre d'échecs consécutifs tolérés ;
* `scenes` : PySceneDetect (activation, seuil), vignettes, corrélation d'histogramme minimale,
  écart et seuil du contrôle de changement progressif, seuil de réunion d'une perturbation,
  longueur maximale d'une transition, longueur minimale d'une séquence ;
* `segmentation` : méthode (`classic` / `none`), fenêtre et seuils de netteté, fraction minimale de
  netteté, déplacement minimal, bande et fraction d'une bordure, critères de fond (ratio et minimum
  d'écart-type temporel), fermeture morphologique, paramètres du segmenteur image par image ;
* `mosaic` : interpolation, taille maximale du canevas, bande de bord d'écran, érosion du masque,
  rampe et poids minimal de bord, puissance du poids d'échelle, taille des tuiles, couverture
  minimale, mode de recadrage, seuil de pile sur disque et dossier temporaire ;
* `quality` : résolution et nombre de frames de l'évaluation, seuil de frame mal reprojetée et
  refusion, seuils des verdicts, sous-dossiers des panels non OK ;
* `export` : carte de couverture, transformations dans le rapport, compression PNG ;
* `runtime` : graine, device (`auto` = CUDA → MPS → CPU), nombre de processus
  (`0` = cœurs performance via `sysctl hw.perflevel0.physicalcpu`), durée maximale et minimale
  des tronçons de découpage (`chunk_seconds` = 300 s, `min_chunk_seconds` = 30 s), niveau de
  journalisation.

Les sections des phases suivantes seront ajoutées au fil de l'eau (`schema_version` contrôle la
compatibilité des profils).

## Validation

```bash
python -m pytest            # tests (~10 min) : config, modèles, video_io, matériel, CLI,
                            # générateur synthétique, évaluation, oracle, mouvement, recalage,
                            # mosaïque, export, pipeline, découpage, segmentation
python -m mypy              # mode strict sur tout le paquet
```

## Architecture

```
panelrecon/
  __init__.py
  cli.py                # mode headless
  synth_cli.py          # génération de vidéos synthétiques de référence
  core/                 # aucune dépendance à PyQt5 (vérifié par test_architecture.py)
    config.py           # PipelineConfig + sections, JSON strict, validation par métadonnées
    models.py           # SimilarityTransform, MotionEstimate, FrameObs, Sequence, CropBox,
                        # MosaicResult, QualityReport, VideoInfo, CancellationToken
    video_io.py         # VideoReader (PyAV, repli cv2), proxy, masque d'exclusion, tampon circulaire
    hardware.py         # cœurs performance, sélection unique du device PyTorch
    geometry.py         # warp anti-repliement exact, masques de couverture, coins
    synthetic.py        # générateur de vidéos + vérité terrain, scénarios de référence
    evaluation.py       # erreurs de pose (jauge), SSIM masqué, couverture, oracle
    motion.py           # estimation de similarité inter-frames (cascade + ECC + validation)
    registration.py     # recalage par chaînage, repère canonique
    scene_split.py      # découpage en séquences (coupes, fondus, perturbations) + recalage
    segmentation.py     # netteté, segmenteur image par image, emprise du panel par séquence
    mosaic.py           # canevas, warp + masques de validité, médiane pondérée par tuiles
    export.py           # PNG RGBA, couverture, rapport JSON
    pipeline.py         # orchestration : 3 passes de décodage (découpage, segmentation, fusion)
  tests/
    conftest.py, videofactory.py
    test_config.py, test_models.py, test_video_io.py, test_hardware.py,
    test_cli.py, test_architecture.py, test_synthetic.py, test_evaluation.py,
    test_motion.py, test_registration.py, test_mosaic.py, test_pipeline.py,
    test_scene_split.py, test_segmentation.py
```

Modules ajoutés à l'arborescence initiale :

* `hardware.py` : exigence d'un module **unique** de sélection du device (et de comptage des cœurs
  performance), utilisé par la CLI, la GUI et les modules optionnels ;
* `geometry.py` : warp et masques partagés par le générateur, l'évaluation et (phase 4) la mosaïque ;
* `synthetic.py`, `evaluation.py` : générateur et métriques de référence ; ils sont dans `core`
  (et non dans `tests`) pour être utilisables en ligne de commande et par la GUI.

### Conventions

* Coordonnées pixel `(x, y)`, origine au centre du pixel haut-gauche (convention `cv2.warpAffine`).
* Une `SimilarityTransform` associée à une frame envoie les coordonnées **de la frame** vers le
  repère cible. Composition : `(A @ B)(p) = A(B(p))`.
* Proxy : `p_proxy = proxy_factor · p_natif`. Une transformation estimée sur proxys est ramenée en
  natif par `M.rescaled(f_src, f_dst)` = `S_dst⁻¹ · M · S_src`.
* Index de frame = rang de la frame **décodée** dans l'ordre de présentation, avant
  sous-échantillonnage. `time_s` (issu des pts) est l'instant de référence, y compris en VFR.

## Hypothèses et points ambigus

1. **Panel rigide et plan** : le contenu du panel ne se déforme pas (pas de parallaxe, pas de
   perspective). Un modèle de similarité suffit ; une inclinaison 3D exigerait une homographie.
   Les micro-animations internes (effets, scintillements) seront traitées comme du bruit par la
   médiane.
2. **Fond = agrandissement flou du panel** (cas typique) : il bouge souvent *avec* le panel, ce qui
   rend la cohérence de mouvement peu discriminante ; la netteté sera le critère principal.
3. **Sous-titres/logos fixes** : la médiane ne les élimine que si le panel bouge sous eux. Pour une
   séquence statique, seule la zone d'exclusion configurable les retire.
4. **Repère canonique à l'échelle maximale observée** : avec un zoom ×4 sur du 1080p, le canevas
   peut dépasser plusieurs centaines de mégapixels. La spécification demande à la fois « ne jamais
   perdre de résolution » et « mémoire bornée » : je prévois un plafond configurable du nombre de
   pixels du canevas, avec erreur explicite (et non une réduction silencieuse).
5. **« Rectangle englobant maximal propre »** : interprété comme le rectangle englobant de la zone
   de couverture ≥ seuil, les trous restant transparents et le ratio de couverture étant rapporté.
   Option alternative prévue : plus grand rectangle inscrit entièrement couvert.
6. **Seuils de validation** (SSIM > 0,95, erreur < 1 px) : garantis sur le synthétique uniquement ;
   sur vidéo réelle compressée, les seuils du verdict qualité seront à calibrer sur un échantillon.
7. **Rotation d'affichage** (display matrix) : appliquée par quart de tour, validée contre
   l'auto-rotation de ffmpeg ; un angle non multiple de 90° est ignoré avec avertissement.
8. **SAM 2** : les poids (plusieurs centaines de Mo) doivent être téléchargés ; le paquet n'est pas
   distribué de façon stable sur PyPI. Ce sera traité en phase 9 (installation depuis le dépôt
   officiel, chemin des poids configurable).
9. **Frames manquantes** dans un flux endommagé : pas de trou dans les indices, mais `time_s` reste
   exact (testé sur un fichier tronqué).

## Limites connues (phase 5)

Les problèmes signalés en phase 4 sont résolus : le fond n'est plus fusionné (segmentation), une
vidéo est découpée en autant de séquences que de panels (coupes franches et fondus), et le canevas
est limité à l'emprise du panel. Limites restantes :

* **Sous-titres** : les vidéos cibles n'en contiennent pas ; aucune détection automatique d'éléments
  incrustés n'est donc faite. Les zones d'exclusion configurables restent disponibles (vides par
  défaut).
* **Panel rectangulaire et aligné** : l'emprise est un rectangle aligné sur les axes du repère
  canonique (rotation attendue ≈ 0). Un panel non rectangulaire (bulle débordante, découpe
  irrégulière) est approché par son rectangle englobant ; les zones de fond incluses sont alors
  fusionnées (et rejetées par la médiane là où le fond bouge relativement au panel).
* **Bord de panel sans contraste** : un côté sans bordure nette, dont le fond adjacent ne bouge pas
  par rapport à l'écran et ne varie pas (séquence statique ou fond uni), ne peut pas être distingué
  d'un aplat du panel ; il est alors étendu à la limite observée.
* **Fondu sans changement d'histogramme** (deux panels aux palettes identiques) : détecté par le
  contrôle à décalage, avec un retard ; les 1 à 2 premières frames de mélange peuvent rester dans
  la séquence précédente (leur faible contamination est atténuée par la médiane).
* **Frontières de tronçons** : un fondu qui chevauche une frontière peut laisser une ou deux
  frames de mélange former une séquence très courte ; un panel coupé par une frontière perd les
  liens anti-dérive entre ses deux morceaux (chaque morceau est ajusté séparément).
* Dérive du chaînage (≤ 0,8 px aux coins sur 30 frames peu texturées) : recalage sur mosaïque et
  ajustement global en phase 6.

## Limites connues (phase 3)

* Sans segmentation (phase 5), l'estimation porte sur toute la frame (hors zones d'exclusion) : le
  fond flou fournit peu de points et RANSAC les écarte, mais un fond net animé indépendamment
  biaiserait l'estimation. Le paramètre `panel_masks` de `register_frames` est prêt à recevoir les
  masques.
* Chaînage simple : l'erreur s'accumule (jusqu'à ~0,8 px aux coins sur 30 frames peu texturées) ;
  recalage sur mosaïque et ajustement global en phase 6.
* Les frames de fondu enchaîné sont acceptées si elles restent proches d'un des deux panels : leur
  rejet relève du découpage en séquences (phase 5).
* Coût : ~0,1 s par paire en 640×360 sur un cœur (SIFT + appariement exhaustif + ECC). Les replis
  (flot aller-retour, corrélation de phase) sont plus coûteux mais rares sur des panels texturés.
* Le module LoFTR (et RAFT comme flot) de la cascade sera branché en phase 9.

## Limites connues (phase 2)

* Les panels procéduraux imitent la structure d'un manhwa (aplats cernés, trames, hachures,
  bulles, texte) mais pas son style graphique : valider aussi avec `--panel-image` sur de vrais
  panels.
* Le fond flou suiveur (`blur_follow`) suit la translation du cadrage, pas son zoom.
* La compression est simulée par un unique encodage x264/crf ; pas de bruit de capture ni de
  ré-encodage multiple (cas réel des vidéos republiées).
* `oracle_mosaic` garde toute la pile d'observations en mémoire (bornée par `max_stack_bytes`) :
  c'est un outil d'évaluation, pas la fusion par tuiles de la phase 4.

## Limites connues (phase 1)

* La CLI ne fait encore que l'inventaire ; la reconstruction arrive en phases 3–4.
* Le facteur de proxy est mesuré sur l'axe horizontal après arrondi des dimensions : pour des
  tailles impaires, l'anisotropie résiduelle est < 1 px sur toute l'image (négligeable, mais non nulle).
* Le repli `cv2.VideoCapture` ne fournit pas les pts bruts (timestamps en millisecondes seulement)
  ni la rotation d'affichage.
* Le décodage relit le fichier depuis le début à chaque appel de `frames()` (exactitude préférée à
  la précision variable du seek selon les conteneurs).
