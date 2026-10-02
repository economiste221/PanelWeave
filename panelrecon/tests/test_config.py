from __future__ import annotations

import json
from pathlib import Path

import pytest

from panelrecon.core.config import (
    CONFIG_SCHEMA_VERSION,
    ConfigError,
    PipelineConfig,
    VideoIOConfig,
    field_metadata,
)


def test_defaults_are_valid() -> None:
    config = PipelineConfig()
    config.validate()
    assert config.preprocess.motion_long_side == 640
    assert config.video.extensions == (".mp4", ".mkv", ".webm", ".mov")
    assert config.runtime.device == "auto"


def test_json_roundtrip(tmp_path: Path) -> None:
    config = PipelineConfig()
    config.video.frame_step = 2
    config.video.max_fps = 12.5
    config.preprocess.exclusion_zones = ((0.0, 0.85, 1.0, 1.0), (0.9, 0.0, 1.0, 0.1))
    config.runtime.device = "mps"
    path = tmp_path / "profiles" / "cfg.json"
    config.save(path)
    loaded = PipelineConfig.load(path)
    assert loaded == config
    assert isinstance(loaded.preprocess.exclusion_zones[0], tuple)
    assert not path.with_name("cfg.json.tmp").exists()


def test_partial_dict_uses_defaults() -> None:
    config = PipelineConfig.from_dict({"video": {"frame_step": 3}})
    assert config.video.frame_step == 3
    assert config.preprocess == PipelineConfig().preprocess


def test_int_accepted_for_float_field() -> None:
    config = PipelineConfig.from_dict({"video": {"max_fps": 10}})
    assert config.video.max_fps == 10.0
    assert isinstance(config.video.max_fps, float)


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({"unknown_section": {}}, "Sections inconnues"),
        ({"video": {"nope": 1}}, "Clés inconnues"),
        ({"video": {"frame_step": "2"}}, "entier attendu"),
        ({"video": {"frame_step": True}}, "entier attendu"),
        ({"video": {"frame_step": 2.5}}, "entier attendu"),
        ({"video": {"recursive": 1}}, "booléen attendu"),
        ({"video": {"frame_step": 0}}, "minimum"),
        ({"preprocess": {"motion_long_side": 100000}}, "maximum"),
        ({"runtime": {"device": "tpu"}}, "valeurs possibles"),
        ({"video": {"extensions": ["mp4"]}}, "extension invalide"),
        ({"video": {"extensions": []}}, "au moins une extension"),
        ({"preprocess": {"exclusion_zones": [[0.5, 0.5, 0.2, 0.9]]}}, "x0 < x1"),
        ({"preprocess": {"exclusion_zones": [[0.0, 0.0, 1.5, 1.0]]}}, "hors de"),
        ({"preprocess": {"exclusion_zones": [[0.0, 0.0, 1.0]]}}, "4 éléments"),
        ({"video": []}, "objet JSON"),
        ({"schema_version": 999}, "non supportée"),
    ],
)
def test_invalid_configs_rejected(data: dict[str, object], message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        PipelineConfig.from_dict(data)


def test_validate_catches_direct_assignment() -> None:
    config = PipelineConfig()
    config.video.frame_step = -1
    with pytest.raises(ConfigError):
        config.validate()
    config = PipelineConfig()
    config.video.max_fps = float("nan")
    with pytest.raises(ConfigError, match="non finie"):
        config.validate()


def test_invalid_json_and_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="JSON invalide"):
        PipelineConfig.from_json("{not json")
    with pytest.raises(ConfigError, match="Impossible de lire"):
        PipelineConfig.load(tmp_path / "absent.json")


def test_to_dict_is_json_serialisable() -> None:
    data = PipelineConfig().to_dict()
    assert data["schema_version"] == CONFIG_SCHEMA_VERSION
    json.dumps(data)
    assert isinstance(data["video"]["extensions"], list)


def test_field_metadata() -> None:
    meta = field_metadata(VideoIOConfig, "frame_step")
    assert meta["min"] == 1 and meta["max"] == 1000 and meta["help"]
    with pytest.raises(KeyError):
        field_metadata(VideoIOConfig, "nope")
