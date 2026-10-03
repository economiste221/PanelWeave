#!/usr/bin/env bash
# Construit PanelRecon.app (et une image disque .dmg) sur un Mac Apple Silicon.
#
#   ./packaging/macos/build_app.sh            # Python 3.12 par défaut
#   PYTHON=python3.11 ./packaging/macos/build_app.sh
#
# Résultat : dist/PanelRecon.app et dist/PanelRecon.dmg
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="${PYTHON:-python3.12}"
VENV="$ROOT/build/venv-app"
cd "$ROOT"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "Ce script construit l'application macOS : à lancer sur un Mac." >&2
  exit 1
fi
ARCH="$("$PYTHON" -c 'import platform; print(platform.machine())')"
if [[ "$ARCH" != "arm64" ]]; then
  echo "Python $ARCH détecté : un Python arm64 natif est requis (pas de Rosetta)." >&2
  exit 1
fi

"$PYTHON" -m venv "$VENV"
"$VENV/bin/python" -m pip install --upgrade pip
"$VENV/bin/python" -m pip install -r requirements.txt -r requirements-build.txt
"$VENV/bin/python" -m pip install --no-deps -e .

rm -rf build/PanelRecon dist/PanelRecon dist/PanelRecon.app dist/PanelRecon.dmg
"$VENV/bin/pyinstaller" --noconfirm --clean --distpath dist --workpath build \
  packaging/panelrecon.spec

# Signature ad hoc : nécessaire pour lancer l'application sur Apple Silicon.
codesign --force --deep --sign - dist/PanelRecon.app

# Vérification : le binaire empaqueté traite une vidéo synthétique en ligne de commande.
SMOKE="$(mktemp -d)"
"$VENV/bin/python" -m panelrecon.synth_cli --scenario pan_horizontal --output "$SMOKE/video"
dist/PanelRecon.app/Contents/MacOS/PanelRecon --cli --input "$SMOKE/video" --output "$SMOKE/out"
ls "$SMOKE/out"
rm -rf "$SMOKE"

hdiutil create -volname PanelRecon -srcfolder dist/PanelRecon.app -ov -format UDZO dist/PanelRecon.dmg
echo "OK : dist/PanelRecon.app et dist/PanelRecon.dmg"
