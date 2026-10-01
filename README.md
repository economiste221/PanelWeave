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
| 2 | Générateur synthétique + tests de référence | à faire |
| 3 | Mouvement (SIFT + RANSAC + ECC), chaînage | à faire |
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

## Configuration

Tout paramètre passe par `PipelineConfig` (`panelrecon/core/config.py`), sérialisée en JSON strict :
clé inconnue, type incompatible ou valeur hors bornes → `ConfigError`. Chaque champ porte ses bornes,
ses choix et un texte d'aide (réutilisés par l'interface en phase 8). Sections actuelles :

* `video` : extensions, récursivité, repli OpenCV, `frame_step`, `max_fps` (sur timestamps),
  threads de décodage, rotation d'affichage, tolérance aux paquets corrompus ;
* `preprocess` : côté long de la version réduite pour le mouvement (960 px), zones d'exclusion
  relatives (sous-titres, logos) ;
* `runtime` : graine, device (`auto` = CUDA → MPS → CPU), nombre de workers
  (`0` = cœurs performance via `sysctl hw.perflevel0.physicalcpu`), niveau de journalisation.

Les sections des phases suivantes seront ajoutées au fil de l'eau (`schema_version` contrôle la
compatibilité des profils).

## Validation

```bash
python -m pytest            # 80 tests : config, modèles, video_io, matériel, CLI, architecture
python -m mypy              # mode strict sur tout le paquet
```

## Architecture

```
panelrecon/
  __init__.py
  cli.py                # mode headless
  core/                 # aucune dépendance à PyQt5 (vérifié par test_architecture.py)
    config.py           # PipelineConfig + sections, JSON strict, validation par métadonnées
    models.py           # SimilarityTransform, MotionEstimate, FrameObs, Sequence, CropBox,
                        # MosaicResult, QualityReport, VideoInfo, CancellationToken
    video_io.py         # VideoReader (PyAV, repli cv2), proxy, masque d'exclusion, tampon circulaire
    hardware.py         # cœurs performance, sélection unique du device PyTorch
  tests/
    conftest.py, videofactory.py
    test_config.py, test_models.py, test_video_io.py, test_hardware.py,
    test_cli.py, test_architecture.py
```

`hardware.py` ne figure pas dans l'arborescence initiale : il répond à l'exigence d'un module
**unique** de sélection du device (et de comptage des cœurs performance), utilisé à la fois par la
CLI, la GUI et les modules optionnels.

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

## Limites connues (phase 1)

* La CLI ne fait encore que l'inventaire ; la reconstruction arrive en phases 3–4.
* Le facteur de proxy est mesuré sur l'axe horizontal après arrondi des dimensions : pour des
  tailles impaires, l'anisotropie résiduelle est < 1 px sur toute l'image (négligeable, mais non nulle).
* Le repli `cv2.VideoCapture` ne fournit pas les pts bruts (timestamps en millisecondes seulement)
  ni la rotation d'affichage.
* Le décodage relit le fichier depuis le début à chaque appel de `frames()` (exactitude préférée à
  la précision variable du seek selon les conteneurs).
