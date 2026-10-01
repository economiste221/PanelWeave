"""Configuration centralisée du pipeline.

Tous les seuils, chemins et paramètres passent par :class:`PipelineConfig`.
La configuration est organisée en sections (une dataclass par sous-système) ;
chaque champ porte des métadonnées (bornes, choix possibles, aide) utilisées à la
fois par la validation et, plus tard, par la génération automatique des
éditeurs de l'interface graphique.

Le format de sérialisation est un JSON strict : toute clé inconnue, tout type
incompatible ou toute valeur hors bornes lève :class:`ConfigError`.
"""

from __future__ import annotations

import json
import logging
import math
import types
import typing
from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Final, TypeVar, Union

logger = logging.getLogger(__name__)

CONFIG_SCHEMA_VERSION: Final[int] = 1

_T = TypeVar("_T")


class ConfigError(ValueError):
    """Erreur de configuration (clé inconnue, type invalide, valeur hors bornes)."""


def param(
    default: Any = MISSING,
    *,
    default_factory: Any = MISSING,
    minimum: float | None = None,
    maximum: float | None = None,
    choices: tuple[str, ...] | None = None,
    help: str = "",  # noqa: A002 - nom volontairement explicite pour l'UI
) -> Any:
    """Déclare un champ de configuration avec ses contraintes de validation."""
    metadata = {"min": minimum, "max": maximum, "choices": choices, "help": help}
    if default_factory is not MISSING:
        return field(default_factory=default_factory, metadata=metadata)
    return field(default=default, metadata=metadata)


ExclusionZone = tuple[float, float, float, float]


@dataclass
class VideoIOConfig:
    """Découverte et décodage des fichiers vidéo."""

    extensions: tuple[str, ...] = param(
        default_factory=lambda: (".mp4", ".mkv", ".webm", ".mov"),
        help="Extensions de fichiers reconnues comme vidéos (insensible à la casse).",
    )
    recursive: bool = param(True, help="Parcourir récursivement les dossiers d'entrée.")
    allow_opencv_fallback: bool = param(
        True, help="Utiliser cv2.VideoCapture si PyAV ne parvient pas à ouvrir le fichier."
    )
    frame_step: int = param(
        1, minimum=1, maximum=1000, help="Ne garder qu'une frame décodée sur N."
    )
    max_fps: float = param(
        0.0,
        minimum=0.0,
        maximum=1000.0,
        help="Cadence maximale conservée, appliquée sur les timestamps (0 = pas de limite).",
    )
    decode_threads: int = param(
        0, minimum=0, maximum=64, help="Threads de décodage FFmpeg (0 = automatique)."
    )
    apply_display_rotation: bool = param(
        True, help="Appliquer la rotation d'affichage (display matrix) stockée dans le conteneur."
    )
    max_decode_errors: int = param(
        25,
        minimum=0,
        maximum=100000,
        help="Paquets corrompus tolérés avant d'abandonner la vidéo.",
    )


@dataclass
class PreprocessConfig:
    """Prétraitement des frames avant estimation du mouvement."""

    motion_long_side: int = param(
        960,
        minimum=64,
        maximum=8192,
        help="Côté long (px) de la version réduite utilisée pour l'estimation du mouvement.",
    )
    exclusion_zones: tuple[ExclusionZone, ...] = param(
        default_factory=tuple,
        help=(
            "Rectangles exclus (x0, y0, x1, y1) en coordonnées relatives [0, 1] "
            "de l'écran : sous-titres, logos, filigranes."
        ),
    )


@dataclass
class RuntimeConfig:
    """Paramètres d'exécution : reproductibilité, matériel, parallélisme, journalisation."""

    seed: int = param(0, minimum=0, maximum=2**31 - 1, help="Graine de tous les tirages aléatoires.")
    device: str = param(
        "auto",
        choices=("auto", "cpu", "mps", "cuda"),
        help="Device PyTorch des modules optionnels (auto = CUDA, puis MPS, puis CPU).",
    )
    num_workers: int = param(
        0,
        minimum=0,
        maximum=256,
        help="Processus de traitement par lot (0 = nombre de cœurs performance).",
    )
    log_level: str = param(
        "INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"), help="Niveau de journalisation."
    )


@dataclass
class PipelineConfig:
    """Configuration complète du pipeline."""

    video: VideoIOConfig = field(default_factory=VideoIOConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    schema_version: int = CONFIG_SCHEMA_VERSION

    # ------------------------------------------------------------------ validation
    def validate(self) -> None:
        """Valide l'ensemble de la configuration ; lève :class:`ConfigError` sinon."""
        if self.schema_version != CONFIG_SCHEMA_VERSION:
            raise ConfigError(
                f"schema_version={self.schema_version} non supportée "
                f"(attendue : {CONFIG_SCHEMA_VERSION})"
            )
        for section_field in _section_fields(type(self)):
            section = getattr(self, section_field.name)
            _validate_section(section_field.name, section)
        self._validate_cross_fields()

    def _validate_cross_fields(self) -> None:
        for ext in self.video.extensions:
            if not ext.startswith(".") or len(ext) < 2:
                raise ConfigError(f"video.extensions : extension invalide {ext!r} (ex. '.mp4')")
        if not self.video.extensions:
            raise ConfigError("video.extensions : au moins une extension est requise")
        for i, zone in enumerate(self.preprocess.exclusion_zones):
            x0, y0, x1, y1 = zone
            if not all(0.0 <= v <= 1.0 for v in zone):
                raise ConfigError(f"preprocess.exclusion_zones[{i}] : coordonnées hors de [0, 1]")
            if not (x0 < x1 and y0 < y1):
                raise ConfigError(
                    f"preprocess.exclusion_zones[{i}] : il faut x0 < x1 et y0 < y1, reçu {zone}"
                )

    # ------------------------------------------------------------- sérialisation
    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"schema_version": self.schema_version}
        for section_field in _section_fields(type(self)):
            section = getattr(self, section_field.name)
            out[section_field.name] = {
                f.name: _to_jsonable(getattr(section, f.name)) for f in fields(section)
            }
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineConfig:
        if not isinstance(data, dict):
            raise ConfigError("La racine de la configuration doit être un objet JSON")
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ConfigError(f"Sections inconnues : {sorted(unknown)}")
        version = data.get("schema_version", CONFIG_SCHEMA_VERSION)
        if not isinstance(version, int) or isinstance(version, bool):
            raise ConfigError("schema_version doit être un entier")
        kwargs: dict[str, Any] = {"schema_version": version}
        for section_field in _section_fields(cls):
            section_cls = _resolve_section_type(cls, section_field.name)
            raw = data.get(section_field.name, {})
            kwargs[section_field.name] = _section_from_dict(section_field.name, section_cls, raw)
        config = cls(**kwargs)
        config.validate()
        return config

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False, sort_keys=False)

    @classmethod
    def from_json(cls, text: str) -> PipelineConfig:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"JSON invalide : {exc}") from exc
        return cls.from_dict(data)

    def save(self, path: Path) -> None:
        """Valide puis écrit la configuration en JSON (écriture atomique)."""
        self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(self.to_json() + "\n", encoding="utf-8")
        tmp.replace(path)
        logger.debug("Configuration écrite dans %s", path)

    @classmethod
    def load(cls, path: Path) -> PipelineConfig:
        path = Path(path)
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ConfigError(f"Impossible de lire la configuration {path} : {exc}") from exc
        config = cls.from_json(text)
        logger.debug("Configuration chargée depuis %s", path)
        return config


# ---------------------------------------------------------------------------
# Implémentation générique (introspection des dataclasses de section)
# ---------------------------------------------------------------------------


def _section_fields(cls: type) -> list[Any]:
    hints = typing.get_type_hints(cls)
    return [f for f in fields(cls) if is_dataclass(hints[f.name])]


def _resolve_section_type(cls: type, name: str) -> type:
    hint = typing.get_type_hints(cls)[name]
    assert isinstance(hint, type)
    return hint


def _section_from_dict(section_name: str, section_cls: type[_T], raw: Any) -> _T:
    if not isinstance(raw, dict):
        raise ConfigError(f"La section {section_name!r} doit être un objet JSON")
    hints = typing.get_type_hints(section_cls)
    known = {f.name for f in fields(section_cls)}  # type: ignore[arg-type]
    unknown = set(raw) - known
    if unknown:
        raise ConfigError(f"Clés inconnues dans {section_name!r} : {sorted(unknown)}")
    kwargs = {
        key: _coerce(value, hints[key], f"{section_name}.{key}") for key, value in raw.items()
    }
    return section_cls(**kwargs)


def _coerce(value: Any, hint: Any, where: str) -> Any:
    """Convertit une valeur JSON vers le type annoté, strictement."""
    origin = typing.get_origin(hint)
    if hint is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{where} : booléen attendu, reçu {value!r}")
        return value
    if hint is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{where} : entier attendu, reçu {value!r}")
        return value
    if hint is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{where} : nombre attendu, reçu {value!r}")
        return float(value)
    if hint is str:
        if not isinstance(value, str):
            raise ConfigError(f"{where} : chaîne attendue, reçu {value!r}")
        return value
    if origin is tuple:
        args = typing.get_args(hint)
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{where} : liste attendue, reçu {value!r}")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_coerce(v, args[0], f"{where}[{i}]") for i, v in enumerate(value))
        if len(value) != len(args):
            raise ConfigError(f"{where} : {len(args)} éléments attendus, reçu {len(value)}")
        return tuple(_coerce(v, a, f"{where}[{i}]") for i, (v, a) in enumerate(zip(value, args)))
    if origin in (Union, types.UnionType):
        args = typing.get_args(hint)
        if value is None and type(None) in args:
            return None
        errors: list[str] = []
        for arg in args:
            if arg is type(None):
                continue
            try:
                return _coerce(value, arg, where)
            except ConfigError as exc:
                errors.append(str(exc))
        raise ConfigError("; ".join(errors))
    raise ConfigError(f"{where} : type d'annotation non supporté {hint!r}")


def _validate_section(section_name: str, section: Any) -> None:
    hints = typing.get_type_hints(type(section))
    for f in fields(section):
        where = f"{section_name}.{f.name}"
        value = getattr(section, f.name)
        # Re-vérifie le type (protège contre une affectation directe erronée).
        _coerce(_to_jsonable(value), hints[f.name], where)
        meta = f.metadata
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if isinstance(value, float) and not math.isfinite(value):
                raise ConfigError(f"{where} : valeur non finie {value!r}")
            lo, hi = meta.get("min"), meta.get("max")
            if lo is not None and value < lo:
                raise ConfigError(f"{where} = {value} < minimum {lo}")
            if hi is not None and value > hi:
                raise ConfigError(f"{where} = {value} > maximum {hi}")
        choices = meta.get("choices")
        if choices is not None and value not in choices:
            raise ConfigError(f"{where} = {value!r} ; valeurs possibles : {list(choices)}")


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_to_jsonable(v) for v in value]
    return value


def field_metadata(section_cls: type, name: str) -> dict[str, Any]:
    """Métadonnées (bornes, choix, aide) d'un champ ; utilisé par l'UI."""
    for f in fields(section_cls):
        if f.name == name:
            return dict(f.metadata)
    raise KeyError(f"{section_cls.__name__} n'a pas de champ {name!r}")
