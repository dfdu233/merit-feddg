from __future__ import annotations

import json
import math
import re
from pathlib import Path

MODALITY_ALIASES = {
    "cr": "cxr",
    "dx": "cxr",
    "xray": "cxr",
    "cxr": "cxr",
    "ct": "ct",
    "mr": "mri",
    "mri": "mri",
    "us": "ultrasound",
    "ultrasound": "ultrasound",
    "oct": "oct",
    "fundus": "fundus",
    "path": "pathology",
    "histology": "pathology",
    "derm": "dermatology",
    "endoscopy": "endoscopy",
}


class MetadataRouter:
    """Auditable metadata/filename router with explicit uncertainty."""

    def __init__(self, available: list[str]) -> None:
        if len(available) < 2:
            raise ValueError("router needs at least two available specialists")
        self.available = available

    def route(self, image_path: str | Path, metadata: dict | None = None) -> dict[str, float]:
        metadata = metadata or {}
        tokens = [str(image_path).lower()]
        tokens.extend(str(value).lower() for value in metadata.values())
        selected = None
        for alias, canonical in MODALITY_ALIASES.items():
            if canonical in self.available and any(alias in token for token in tokens):
                selected = canonical
                break
        if selected is None:
            return {name: 1.0 / len(self.available) for name in self.available}
        peak = 0.92
        tail = (1.0 - peak) / (len(self.available) - 1)
        return {name: peak if name == selected else tail for name in self.available}


def normalized_entropy(probabilities: dict[str, float]) -> float:
    values = [max(float(value), 1e-12) for value in probabilities.values()]
    total = sum(values)
    return -sum((value / total) * math.log(value / total) for value in values) / math.log(len(values))


def route_with_medical_vlm(model, image_path: str | Path, available: list[str]) -> dict[str, float]:
    """Ask a compact medical VLM for modality and input contract, not diagnosis."""

    prompt = (
        "Classify only the imaging modality. Return strict JSON with keys modality, "
        "input_contract, confidence. Allowed modality values: "
        + ", ".join(available)
        + ". Do not diagnose the image."
    )
    text = model.generate(image_path, prompt, max_new_tokens=48)
    match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    selected = None
    confidence = 0.0
    if match:
        try:
            payload = json.loads(match.group(0))
            candidate = str(payload.get("modality", "")).lower().strip()
            selected = MODALITY_ALIASES.get(candidate, candidate)
            confidence = float(payload.get("confidence", 0.0))
        except (ValueError, TypeError, json.JSONDecodeError):
            selected = None
    if selected not in available:
        for name in available:
            if name in text.lower():
                selected = name
                confidence = max(confidence, 0.60)
                break
    if selected not in available:
        return {name: 1.0 / len(available) for name in available}
    confidence = float(min(max(confidence, 1.0 / len(available)), 0.98))
    tail = (1.0 - confidence) / (len(available) - 1)
    return {name: confidence if name == selected else tail for name in available}
