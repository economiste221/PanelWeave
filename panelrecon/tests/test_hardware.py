from __future__ import annotations

import os
import sys
from types import ModuleType, SimpleNamespace

import pytest

from panelrecon.core import hardware


def _fake_torch(cuda: bool, mps: bool) -> ModuleType:
    module = ModuleType("torch")
    module.cuda = SimpleNamespace(is_available=lambda: cuda)  # type: ignore[attr-defined]
    module.backends = SimpleNamespace(  # type: ignore[attr-defined]
        mps=SimpleNamespace(is_available=lambda: mps)
    )
    return module


def test_performance_core_count_positive() -> None:
    assert hardware.performance_core_count() >= 1


def test_performance_cores_on_darwin(monkeypatch: pytest.MonkeyPatch) -> None:
    queried: list[str] = []

    def fake_sysctl(name: str) -> int | None:
        queried.append(name)
        return {"hw.perflevel0.physicalcpu": 4, "hw.physicalcpu": 10}.get(name)

    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(hardware, "_sysctl_int", fake_sysctl)
    assert hardware.performance_core_count() == 4
    assert queried[0] == "hw.perflevel0.physicalcpu"

    # Mac Intel : pas de niveaux de performance.
    monkeypatch.setattr(
        hardware, "_sysctl_int", lambda name: 8 if name == "hw.physicalcpu" else None
    )
    assert hardware.performance_core_count() == 8


def test_resolve_num_workers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hardware, "performance_core_count", lambda: 6)
    assert hardware.resolve_num_workers(0) == 6
    assert hardware.resolve_num_workers(3) == 3
    with pytest.raises(ValueError):
        hardware.resolve_num_workers(-1)


def test_import_torch_sets_mps_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)
    hardware.import_torch()
    assert os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] == "1"


def test_select_device_without_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hardware, "import_torch", lambda: None)
    for pref in ("auto", "cpu", "mps", "cuda"):
        info = hardware.select_torch_device(pref)
        assert info.name == "cpu" and not info.torch_available


@pytest.mark.parametrize(
    ("cuda", "mps", "pref", "expected"),
    [
        (True, True, "auto", "cuda"),
        (False, True, "auto", "mps"),
        (False, False, "auto", "cpu"),
        (True, True, "cpu", "cpu"),
        (False, True, "cuda", "cpu"),
        (True, False, "mps", "cpu"),
        (False, True, "mps", "mps"),
    ],
)
def test_select_device_priority(
    monkeypatch: pytest.MonkeyPatch, cuda: bool, mps: bool, pref: str, expected: str
) -> None:
    monkeypatch.setattr(hardware, "import_torch", lambda: _fake_torch(cuda, mps))
    info = hardware.select_torch_device(pref)
    assert info.name == expected and info.torch_available


def test_select_device_rejects_unknown() -> None:
    with pytest.raises(ValueError):
        hardware.select_torch_device("tpu")
