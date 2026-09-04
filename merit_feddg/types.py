from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass
class EvidenceRecord:
    """Model outputs needed by every comparison method.

    ``general_visual_layers`` contains image-minus-null concept evidence from
    intermediate layers. Specialists expose the same clinical concepts, but
    their scores are never added by MERIT itself.
    """

    sample_id: str
    domain: str
    modality: str
    candidates: list[str]
    label: int
    general_null_logits: np.ndarray
    general_visual_layers: np.ndarray
    expert_scores: dict[str, np.ndarray]
    broad_specialist_scores: np.ndarray
    router_probs: dict[str, float]
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        count = len(self.candidates)
        if count < 2:
            raise ValueError("at least two candidate concepts are required")
        if not 0 <= self.label < count:
            raise ValueError("label is outside the candidate list")
        if self.general_null_logits.shape != (count,):
            raise ValueError("general_null_logits has the wrong shape")
        if self.general_visual_layers.ndim != 2 or self.general_visual_layers.shape[1] != count:
            raise ValueError("general_visual_layers must be [layers, candidates]")
        if self.broad_specialist_scores.shape != (count,):
            raise ValueError("broad_specialist_scores has the wrong shape")
        for name, scores in self.expert_scores.items():
            if scores.shape != (count,):
                raise ValueError(f"expert {name!r} has the wrong shape")
        if not self.router_probs:
            raise ValueError("router_probs cannot be empty")

    @property
    def general_final_logits(self) -> np.ndarray:
        return self.general_null_logits + self.general_visual_layers[-1]

    def modalities_for_expert(self, expert_id: str) -> tuple[str, ...]:
        configured = (self.metadata or {}).get("expert_modalities", {}).get(expert_id)
        if configured:
            if isinstance(configured, str):
                return (configured,)
            return tuple(str(value) for value in configured)
        # Backwards-compatible caches used modality names as expert IDs.
        return (expert_id,)

    def capabilities_for_expert(self, expert_id: str) -> tuple[str, ...]:
        configured = (self.metadata or {}).get("expert_capabilities", {}).get(expert_id)
        if configured:
            if isinstance(configured, str):
                return (configured,)
            return tuple(str(value) for value in configured)
        return ("classification",)

    def compatible_experts(
        self,
        modality: str | None = None,
        capability: str | None = None,
    ) -> tuple[str, ...]:
        wanted_modality = self.modality if modality is None else modality
        return tuple(
            expert_id
            for expert_id in sorted(self.expert_scores)
            if wanted_modality in self.modalities_for_expert(expert_id)
            and (
                capability is None
                or capability in self.capabilities_for_expert(expert_id)
                )
        )

    def modality_probabilities(self) -> dict[str, float]:
        """Return normalized router mass over modalities, not expert IDs.

        New caches preserve the router's native modality distribution.  For
        legacy caches, split each expert's mass evenly across the modalities
        on its card, then aggregate.  This prevents a modality with several
        registered experts from being penalized merely because its probability
        was fanned out across more plugins.
        """

        configured = (self.metadata or {}).get("modality_router_probs")
        if configured:
            values = {str(name): float(value) for name, value in configured.items()}
        else:
            values: dict[str, float] = {}
            for expert_id, probability in self.router_probs.items():
                modalities = self.modalities_for_expert(expert_id)
                share = float(probability) / max(len(modalities), 1)
                for modality in modalities:
                    values[modality] = values.get(modality, 0.0) + share
        if not values or any(not np.isfinite(value) or value < 0.0 for value in values.values()):
            raise ValueError("modality router probabilities must be finite and non-negative")
        total = sum(values.values())
        if total <= 0.0:
            raise ValueError("modality router probabilities must have positive mass")
        return {name: value / total for name, value in values.items()}

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("general_null_logits", "general_visual_layers", "broad_specialist_scores"):
            payload[key] = np.asarray(payload[key]).tolist()
        payload["expert_scores"] = {
            name: np.asarray(value).tolist() for name, value in self.expert_scores.items()
        }
        return payload

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> EvidenceRecord:
        copied = dict(payload)
        copied["general_null_logits"] = np.asarray(copied["general_null_logits"], dtype=float)
        copied["general_visual_layers"] = np.asarray(copied["general_visual_layers"], dtype=float)
        copied["broad_specialist_scores"] = np.asarray(
            copied["broad_specialist_scores"], dtype=float
        )
        copied["expert_scores"] = {
            name: np.asarray(value, dtype=float) for name, value in copied["expert_scores"].items()
        }
        return cls(**copied)


@dataclass(frozen=True)
class Prediction:
    method: str
    sample_id: str
    domain: str
    modality: str
    label: int
    predicted: int
    confidence: float
    route: str
    route_confidence: float
    intervention_gate: float
    erasure: float
    scores: tuple[float, ...]
