# -*- mode: python ; coding: utf-8 -*-
# Spécification PyInstaller de PanelRecon (macOS arm64 ; fonctionne aussi sous Linux
# pour la validation). Construire avec : packaging/macos/build_app.sh
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules

ROOT = Path(SPECPATH).parent
IS_MAC = sys.platform == "darwin"

hiddenimports = (
    collect_submodules("panelrecon")
    + collect_submodules("scenedetect")
    + ["scipy.optimize", "scipy.sparse", "skimage.metrics", "av"]
)
excludes = ["tkinter", "torch", "torchvision", "kornia", "matplotlib", "IPython", "pytest"]

a = Analysis(
    [str(ROOT / "packaging" / "launcher.py")],
    pathex=[str(ROOT)],
    hiddenimports=hiddenimports,
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="PanelRecon",
    console=not IS_MAC,
    target_arch="arm64" if IS_MAC else None,
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="PanelRecon", upx=False)
if IS_MAC:
    app = BUNDLE(
        coll,
        name="PanelRecon.app",
        bundle_identifier="com.panelweave.panelrecon",
        info_plist={
            "CFBundleName": "PanelRecon",
            "CFBundleDisplayName": "PanelRecon",
            "CFBundleShortVersionString": "0.1.0",
            "NSHighResolutionCapable": True,
            "LSMinimumSystemVersion": "14.0",
            "LSArchitecturePriority": ["arm64"],
            "CFBundleDocumentTypes": [{
                "CFBundleTypeName": "Vidéo",
                "CFBundleTypeRole": "Viewer",
                "LSItemContentTypes": ["public.movie", "public.mpeg-4", "com.apple.quicktime-movie",
                                       "org.matroska.mkv", "org.webmproject.webm"],
            }],
        },
    )
