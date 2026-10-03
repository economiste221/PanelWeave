"""Éditeur de tous les champs de ``PipelineConfig``, construit par introspection.

Chaque champ reçoit le widget adapté à son type (case à cocher, champ numérique
borné par ses métadonnées ``min``/``max``, liste de choix, texte, ou JSON pour
les listes) et son texte d'aide en infobulle. :meth:`ConfigEditor.config`
reconstruit la configuration et la valide (``ConfigError`` sinon).
"""

from __future__ import annotations

import json
import typing
from dataclasses import fields
from typing import Any

from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QLineEdit,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QWidget,
)

from panelrecon.core.config import ConfigError, PipelineConfig

_SECTION_TITLES = {
    "video": "Vidéo",
    "preprocess": "Prétraitement",
    "motion": "Mouvement",
    "registration": "Recalage",
    "scenes": "Séquences",
    "segmentation": "Segmentation",
    "mosaic": "Fusion",
    "quality": "Qualité",
    "export": "Export",
    "runtime": "Exécution",
}
_INT_LIMIT = 2**31 - 1


class _Field:
    """Lien entre un champ de configuration et son widget."""

    def __init__(self, hint: Any, meta: dict[str, Any], value: Any) -> None:
        self.hint = hint
        self.widget: QWidget
        choices = meta.get("choices")
        lo, hi = meta.get("min"), meta.get("max")
        if hint is bool:
            box = QCheckBox()
            self.widget = box
        elif hint is int:
            spin = QSpinBox()
            spin.setRange(int(lo) if lo is not None else -_INT_LIMIT,
                          int(hi) if hi is not None else _INT_LIMIT)
            self.widget = spin
        elif hint is float:
            dspin = QDoubleSpinBox()
            dspin.setDecimals(6)
            dspin.setRange(float(lo) if lo is not None else -1e12,
                           float(hi) if hi is not None else 1e12)
            dspin.setSingleStep(_step(lo, hi))
            self.widget = dspin
        elif hint is str and choices:
            combo = QComboBox()
            combo.addItems(list(choices))
            self.widget = combo
        else:  # chaînes libres et listes (JSON)
            self.widget = QLineEdit()
        self.widget.setToolTip(str(meta.get("help", "")))
        self.set(value)

    def set(self, value: Any) -> None:
        w = self.widget
        if isinstance(w, QCheckBox):
            w.setChecked(bool(value))
        elif isinstance(w, QSpinBox):
            w.setValue(int(value))
        elif isinstance(w, QDoubleSpinBox):
            w.setValue(float(value))
        elif isinstance(w, QComboBox):
            w.setCurrentText(str(value))
        elif isinstance(w, QLineEdit):
            w.setText(value if self.hint is str else json.dumps(_jsonable(value)))

    def get(self) -> Any:
        w = self.widget
        if isinstance(w, QCheckBox):
            return w.isChecked()
        if isinstance(w, QSpinBox):
            return w.value()
        if isinstance(w, QDoubleSpinBox):
            return w.value()
        if isinstance(w, QComboBox):
            return w.currentText()
        assert isinstance(w, QLineEdit)
        if self.hint is str:
            return w.text()
        try:
            return json.loads(w.text())
        except json.JSONDecodeError as exc:
            raise ConfigError(f"JSON invalide : {w.text()!r} ({exc.msg})") from exc


def _step(lo: Any, hi: Any) -> float:
    if lo is not None and hi is not None and float(hi) - float(lo) <= 1.0:
        return 0.01
    return 0.1


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    return value


class ConfigEditor(QTabWidget):
    """Un onglet par section de la configuration."""

    def __init__(self, config: PipelineConfig | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._fields: dict[tuple[str, str], _Field] = {}
        config = config or PipelineConfig()
        for section_field in fields(PipelineConfig):
            section = getattr(config, section_field.name)
            if not hasattr(section, "__dataclass_fields__"):
                continue
            hints = typing.get_type_hints(type(section))
            page = QWidget()
            form = QFormLayout(page)
            for f in fields(section):
                editor = _Field(hints[f.name], dict(f.metadata), getattr(section, f.name))
                self._fields[(section_field.name, f.name)] = editor
                form.addRow(f.name, editor.widget)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(page)
            self.addTab(scroll, _SECTION_TITLES.get(section_field.name, section_field.name))

    def widget_for(self, section: str, name: str) -> QWidget:
        return self._fields[(section, name)].widget

    def set_config(self, config: PipelineConfig) -> None:
        for (section, name), editor in self._fields.items():
            editor.set(getattr(getattr(config, section), name))

    def config(self) -> PipelineConfig:
        """Configuration saisie, validée (``ConfigError`` si un champ est invalide)."""
        data: dict[str, Any] = PipelineConfig().to_dict()
        for (section, name), editor in self._fields.items():
            try:
                data[section][name] = editor.get()
            except ConfigError as exc:
                raise ConfigError(f"{section}.{name} : {exc}") from exc
        config = PipelineConfig.from_dict(data)
        config.validate()
        return config
