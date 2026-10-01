"""Détection matérielle : cœurs CPU et device PyTorch.

C'est le **seul** module autorisé à importer ``torch`` pour choisir un device.
``PYTORCH_ENABLE_MPS_FALLBACK=1`` est positionné avant tout import de torch afin
que les opérations non implémentées sur MPS retombent automatiquement sur CPU.
PyTorch est optionnel : en son absence, le device résolu est ``cpu`` et
``torch_available`` vaut ``False``.
"""

from __future__ import annotations

import importlib
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from types import ModuleType

logger = logging.getLogger(__name__)

_MPS_FALLBACK_ENV = "PYTORCH_ENABLE_MPS_FALLBACK"


def _sysctl_int(name: str) -> int | None:
    exe = shutil.which("sysctl") or ("/usr/sbin/sysctl" if sys.platform == "darwin" else None)
    if exe is None:
        return None
    try:
        out = subprocess.run(
            [exe, "-n", name], capture_output=True, text=True, timeout=2.0, check=True
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    try:
        value = int(out)
    except ValueError:
        return None
    return value if value > 0 else None


def performance_core_count() -> int:
    """Nombre de cœurs « performance ».

    * macOS (Apple Silicon) : ``sysctl hw.perflevel0.physicalcpu`` ;
      ``hw.physicalcpu`` sur les Mac sans niveaux de performance.
    * autres systèmes : CPU réellement attribués au processus (affinité).
    """
    if sys.platform == "darwin":
        for key in ("hw.perflevel0.physicalcpu", "hw.physicalcpu"):
            value = _sysctl_int(key)
            if value is not None:
                return value
    sched_getaffinity = getattr(os, "sched_getaffinity", None)
    if sched_getaffinity is not None:
        try:
            return max(1, len(sched_getaffinity(0)))
        except OSError:
            pass
    return max(1, os.cpu_count() or 1)


def resolve_num_workers(configured: int) -> int:
    """``configured`` si > 0, sinon le nombre de cœurs performance."""
    if configured < 0:
        raise ValueError(f"Nombre de workers invalide : {configured}")
    return configured if configured > 0 else performance_core_count()


@dataclass(frozen=True)
class DeviceInfo:
    """Device PyTorch retenu."""

    name: str  # "cuda", "mps" ou "cpu"
    torch_available: bool
    reason: str


def import_torch() -> ModuleType | None:
    """Importe torch après avoir activé le repli MPS → CPU ; ``None`` si indisponible."""
    os.environ.setdefault(_MPS_FALLBACK_ENV, "1")
    try:
        return importlib.import_module("torch")
    except ImportError:
        return None
    except Exception as exc:  # bibliothèque native cassée : on journalise et on continue sans
        logger.warning("Import de torch impossible : %s", exc)
        return None


def _mps_available(torch: ModuleType) -> bool:
    backends = getattr(torch, "backends", None)
    mps = getattr(backends, "mps", None)
    if mps is None:
        return False
    try:
        return bool(mps.is_available())
    except Exception as exc:
        logger.warning("Interrogation de MPS impossible : %s", exc)
        return False


def _cuda_available(torch: ModuleType) -> bool:
    cuda = getattr(torch, "cuda", None)
    if cuda is None:
        return False
    try:
        return bool(cuda.is_available())
    except Exception as exc:
        logger.warning("Interrogation de CUDA impossible : %s", exc)
        return False


def select_torch_device(preference: str = "auto") -> DeviceInfo:
    """Choisit le device : ``auto`` = CUDA, puis MPS, puis CPU.

    Une préférence explicite indisponible retombe sur CPU avec un avertissement.
    """
    if preference not in ("auto", "cpu", "mps", "cuda"):
        raise ValueError(f"Préférence de device inconnue : {preference!r}")
    torch = import_torch()
    if torch is None:
        info = DeviceInfo("cpu", False, "PyTorch non installé")
        if preference not in ("auto", "cpu"):
            logger.warning("Device %r demandé mais PyTorch est absent : CPU utilisé", preference)
        return info
    if preference == "cpu":
        return DeviceInfo("cpu", True, "CPU demandé")
    if preference in ("auto", "cuda") and _cuda_available(torch):
        return DeviceInfo("cuda", True, "CUDA disponible")
    if preference == "cuda":
        logger.warning("CUDA demandé mais indisponible : CPU utilisé")
        return DeviceInfo("cpu", True, "CUDA indisponible")
    if _mps_available(torch):
        return DeviceInfo("mps", True, "MPS (Apple Silicon) disponible")
    if preference == "mps":
        logger.warning("MPS demandé mais indisponible : CPU utilisé")
        return DeviceInfo("cpu", True, "MPS indisponible")
    return DeviceInfo("cpu", True, "Aucun accélérateur disponible")
