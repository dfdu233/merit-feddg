"""Candidate-free capability requests and native evidence contracts.

The controller can request only registered tools, never execute returned code.
Confidence is an optional model-native signal, not a calibrated probability.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field

CAPABILITIES = frozenset({"classification", "segmentation", "detection", "retrieval", "generation"})


def scoped_key(expert, modality, task, capability, scope):
    parts = (expert, modality, task, capability, scope)
    if any(not isinstance(p, str) or not p or "|" in p for p in parts):
        raise ValueError("scope key fields must be nonempty strings without '|'")
    return "|".join(parts)


@dataclass(frozen=True)
class CapabilityRequest:
    sample_id: str
    image: str
    question: str
    modality: str
    task: str
    domain: str
    group_id: str
    capability: str
    query: str = ""
    scope: str = ""
    generated_prefix: str = ""
    region: tuple[float, float, float, float] | None = None

    def __post_init__(self):
        if self.capability not in CAPABILITIES:
            raise ValueError("unsupported capability")
        if self.region is not None:
            if len(self.region) != 4 or not all(math.isfinite(x) for x in self.region):
                raise ValueError("region must contain four finite numbers")
            x0, y0, x1, y1 = self.region
            if not 0 <= x0 < x1 <= 1 or not 0 <= y0 < y1 <= 1:
                raise ValueError("region must be normalized xyxy")


@dataclass(frozen=True)
class EvidenceItem:
    evidence_id: str
    expert_id: str
    capability: str
    scope: str
    payload: dict
    summary: str = ""
    confidence: float | None = None
    provenance: dict = field(default_factory=dict)


@dataclass(frozen=True)
class CapabilityResult:
    expert_id: str
    capability: str
    items: tuple[EvidenceItem, ...]
    reason: str = "ok"


def validate_result(result, expert_id, request):
    if not isinstance(result, CapabilityResult):
        raise TypeError("adapter must return CapabilityResult")
    if result.expert_id != expert_id or result.capability != request.capability:
        raise ValueError("native result identity mismatch")
    ids = set()
    for item in result.items:
        if not isinstance(item, EvidenceItem) or not item.evidence_id or item.evidence_id in ids:
            raise ValueError("evidence IDs must be nonempty and unique per result")
        ids.add(item.evidence_id)
        if (item.expert_id, item.capability, item.scope) != (
            expert_id,
            request.capability,
            request.scope,
        ):
            raise ValueError("evidence scope mismatch")
        if not isinstance(item.payload, dict) or not item.payload:
            raise ValueError("native payload must be a nonempty dictionary")
        if item.confidence is not None and not math.isfinite(item.confidence):
            raise ValueError("nonfinite confidence")
        # Reject NaN, tensors, executable objects, etc. Raw native outputs remain
        # JSON observations; no global answer vocabulary is required.
        json.dumps(asdict(item), allow_nan=False)
    return result


def tool_descriptors(specs, row, allowed_pairs=None):
    descriptors = []
    for name, spec in specs.items():
        if spec.get("modalities") and row["modality"] not in spec["modalities"]:
            continue
        if spec.get("tasks") and row["task"] not in spec["tasks"]:
            continue
        for capability in spec.get("capabilities", []):
            if capability not in CAPABILITIES:
                raise ValueError(f"unknown capability for {name}: {capability}")
            scope = spec.get("scope", capability)
            if allowed_pairs is not None and (name, capability, scope) not in allowed_pairs:
                continue
            scoped_key(name, row["modality"], row["task"], capability, scope)
            descriptors.append(
                {
                    "expert": name,
                    "capability": capability,
                    "scope": scope,
                    "description": spec.get("description", ""),
                    "requires_region": bool(
                        spec.get("requires_region", capability == "segmentation")
                    ),
                }
            )
    return descriptors
