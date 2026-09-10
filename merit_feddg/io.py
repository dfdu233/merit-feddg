from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import yaml

from .types import EvidenceRecord, Prediction


def load_yaml(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_experiment_yaml(path):
    """One-level base and explicit per-expert overrides; paths are repo-relative."""
    config = load_yaml(path)
    if "base_config" in config:
        base = load_yaml(config.pop("base_config"))
        if "base_config" in base:
            raise ValueError("nested experiment inheritance is unsupported")
        # A bridge checkpoint is an add-on to the same generalist, not a full
        # replacement for its local model/source/vision paths. Keep the base
        # identity when an experiment overrides only generalist runtime fields.
        if "generalist" in config:
            if not isinstance(base.get("generalist"), dict) or not isinstance(
                config["generalist"], dict
            ):
                raise TypeError("generalist config and override must be dictionaries")
            config["generalist"] = {**base["generalist"], **config["generalist"]}
        base.update(config)
        config = base
    overrides = config.pop("expert_overrides", {})
    if not isinstance(overrides, dict) or set(overrides) - set(config.get("experts", {})):
        raise ValueError("expert_overrides must name existing experts")
    for name, update in overrides.items():
        if not isinstance(update, dict):
            raise TypeError("expert override must be a dictionary")
        config["experts"][name].update(update)
    return config


def save_json(path: str | Path, payload: object) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def save_yaml(path: str | Path, payload: object) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def save_records(path: str | Path, records: Iterable[EvidenceRecord]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_json(), ensure_ascii=False) + "\n")


def load_records(path: str | Path) -> list[EvidenceRecord]:
    records: list[EvidenceRecord] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(EvidenceRecord.from_json(json.loads(line)))
    return records


def save_predictions(path: str | Path, predictions: Iterable[Prediction]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for prediction in predictions:
            handle.write(json.dumps(prediction.__dict__, ensure_ascii=False) + "\n")
