from __future__ import annotations

import numpy as np

from .feddg import FederatedReliabilityCalibrator
from .math_utils import clip_norm, cosine, sigmoid, softmax
from .types import EvidenceRecord, Prediction


def selected_route(record: EvidenceRecord) -> tuple[str, float]:
    route = max(record.router_probs, key=record.router_probs.get)
    probabilities = np.asarray(list(record.router_probs.values()), dtype=float)
    probabilities = probabilities / probabilities.sum()
    uniform = 1.0 / len(probabilities)
    certainty = (float(np.max(probabilities)) - uniform) / (1.0 - uniform)
    return route, float(np.clip(certainty, 0.0, 1.0))


def _wrong_route(record: EvidenceRecord) -> tuple[str, float]:
    choices = sorted(record.router_probs, key=record.router_probs.get, reverse=True)
    wrong = next((name for name in choices if name != record.modality), choices[-1])
    return wrong, float(record.router_probs.get(wrong, 0.0))


def specialist_lens(
    record: EvidenceRecord,
    route: str,
    route_confidence: float,
    settings: dict,
    reliability: float,
) -> tuple[np.ndarray, float, float]:
    """Restore generalist evidence selected, but never supplied, by an expert."""

    expert = record.expert_scores[route]
    layers = record.general_visual_layers
    agreements = np.asarray([cosine(layer, expert) for layer in layers], dtype=float)
    final_agreement = float(agreements[-1])
    erasure = max(0.0, float(np.max(agreements[:-1]) - final_agreement))
    weights = softmax(agreements[:-1], float(settings["agreement_temperature"]))
    selected_generalist_evidence = np.sum(layers[:-1] * weights[:, None], axis=0)
    residual = selected_generalist_evidence - layers[-1]
    residual = clip_norm(residual, float(settings["max_correction_norm"]))
    erasure_gate = sigmoid((erasure - float(settings["erasure_floor"])) / 0.10)
    gate = float(np.clip(route_confidence * reliability * erasure_gate, 0.0, 1.0))
    corrected = record.general_final_logits + float(settings["correction_strength"]) * gate * residual
    return corrected, gate, erasure


def predict(
    record: EvidenceRecord,
    method: str,
    settings: dict,
    calibrator: FederatedReliabilityCalibrator | None = None,
    shuffled_expert_scores: np.ndarray | None = None,
) -> Prediction:
    route, route_confidence = selected_route(record)
    gate = 0.0
    erasure = 0.0

    if method == "generalist":
        scores = record.general_final_logits
    elif method == "broad_specialist":
        scores = record.general_null_logits + record.broad_specialist_scores
    elif method == "gsco_context":
        # A score-space proxy for GSCo's specialist diagnosis in the context.
        scores = record.general_final_logits + 0.60 * record.expert_scores[route]
    elif method == "routed_logit_fusion":
        scores = record.general_final_logits + route_confidence * record.expert_scores[route]
    elif method in {"merit", "merit_feddg"}:
        reliability = 1.0
        if method == "merit_feddg":
            if calibrator is None:
                raise ValueError("merit_feddg requires a fitted calibrator")
            reliability = calibrator.score(route)
        scores, gate, erasure = specialist_lens(
            record, route, route_confidence, settings, reliability
        )
    elif method == "wrong_route":
        route, route_confidence = _wrong_route(record)
        scores, gate, erasure = specialist_lens(
            record, route, route_confidence, settings, reliability=1.0
        )
    elif method == "shuffled_expert":
        if shuffled_expert_scores is None:
            raise ValueError("shuffled_expert requires shuffled scores")
        copied = EvidenceRecord(
            sample_id=record.sample_id,
            domain=record.domain,
            modality=record.modality,
            candidates=record.candidates,
            label=record.label,
            general_null_logits=record.general_null_logits,
            general_visual_layers=record.general_visual_layers,
            expert_scores={**record.expert_scores, route: shuffled_expert_scores},
            broad_specialist_scores=record.broad_specialist_scores,
            router_probs=record.router_probs,
            metadata=record.metadata,
        )
        scores, gate, erasure = specialist_lens(
            copied, route, route_confidence, settings, reliability=1.0
        )
    else:
        raise ValueError(f"unknown comparison method: {method}")

    probabilities = softmax(np.asarray(scores, dtype=float))
    predicted = int(np.argmax(probabilities))
    return Prediction(
        method=method,
        sample_id=record.sample_id,
        domain=record.domain,
        modality=record.modality,
        label=record.label,
        predicted=predicted,
        confidence=float(probabilities[predicted]),
        route=route,
        route_confidence=route_confidence,
        intervention_gate=gate,
        erasure=erasure,
        scores=tuple(float(value) for value in scores),
    )
