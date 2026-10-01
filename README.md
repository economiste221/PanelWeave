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
| 4 | Mosaïque, fusion médiane, couverture, export | à faire |
| 5 | Segmentation classique, découpage en séquences | à faire |
| 6 | Recalage sur mosaïque, ajustement global | à faire |
| 7 | Contrôle qualité, rapport | à faire |
| 8 | Interface PyQt5 | à faire |
| 9 | Modules optionnels (LoFTR, RAFT, SAM 2), traitement par lot multiprocessus | à faire |

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

## Utilisation (phase 1)

```bash
# Écrire la configuration par défaut (point de départ d'un profil)
python -m panelrecon.cli --write-default-config profils/defaut.json

# Inventaire d'un lot : découverte récursive, décodage complet en flux, statistiques temporelles
python -m panelrecon.cli --input ~/Videos/manhwa --output ~/Videos/sortie --config profils/defaut.json
```

Le mode `inventory` écrit `inventory.json` (métadonnées, frames décodées/conservées, erreurs de
décodage, intervalles min/médian/max, détection de fréquence variable) et `panelrecon.log`.
Une vidéo illisible est journalisée avec sa trace sans interrompre le lot.
Codes de sortie : `0` succès, `1` au moins une vidéo en échec, `2` erreur d'usage/configuration,
`130` interruption.

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
* `runtime` : graine, device (`auto` = CUDA → MPS → CPU), nombre de workers
  (`0` = cœurs performance via `sysctl hw.perflevel0.physicalcpu`), niveau de journalisation.

Les sections des phases suivantes seront ajoutées au fil de l'eau (`schema_version` contrôle la
compatibilité des profils).

## Validation

```bash
python -m pytest            # 170 tests (~2 min 30) : config, modèles, video_io, matériel, CLI,
                            # générateur synthétique, évaluation, oracle, mouvement, recalage
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
  tests/
    conftest.py, videofactory.py
    test_config.py, test_models.py, test_video_io.py, test_hardware.py,
    test_cli.py, test_architecture.py, test_synthetic.py, test_evaluation.py,
    test_motion.py, test_registration.py
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
