from __future__ import annotations

import subprocess
import sys

CORE_MODULES = (
    "panelrecon.core.config",
    "panelrecon.core.models",
    "panelrecon.core.video_io",
    "panelrecon.core.hardware",
    "panelrecon.core.parallel",
    "panelrecon.core.geometry",
    "panelrecon.core.synthetic",
    "panelrecon.core.evaluation",
    "panelrecon.core.motion",
    "panelrecon.core.registration",
    "panelrecon.core.mosaic",
    "panelrecon.core.scene_split",
    "panelrecon.core.segmentation",
    "panelrecon.core.export",
    "panelrecon.core.pipeline",
    "panelrecon.cli",
    "panelrecon.synth_cli",
)


def test_core_does_not_import_gui_or_torch() -> None:
    """La couche core (et la CLI) ne doit charger ni PyQt5 ni torch à l'import."""
    code = (
        "import sys\n"
        + "".join(f"import {m}\n" for m in CORE_MODULES)
        + "bad = sorted(m for m in sys.modules if m.split('.')[0] in ('PyQt5', 'torch'))\n"
        + "print(','.join(bad))\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == ""
