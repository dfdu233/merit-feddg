from __future__ import annotations

from collections import defaultdict

import numpy as np

from .types import Prediction


def expected_calibration_error(predictions: list[Prediction], bins: int = 10) -> float:
    if not predictions:
        return 0.0
    confidences = np.asarray([item.confidence for item in predictions])
    correct = np.asarray([item.predicted == item.label for item in predictions], dtype=float)
    total = len(predictions)
    result = 0.0
    for lower in np.linspace(0.0, 1.0, bins, endpoint=False):
        upper = lower + 1.0 / bins
        mask = (confidences >= lower) & (confidences < upper if upper < 1.0 else confidences <= 1.0)
        if np.any(mask):
            result += float(np.sum(mask) / total) * abs(
                float(np.mean(correct[mask])) - float(np.mean(confidences[mask]))
            )
    return result


def summarize(predictions: list[Prediction], base: list[Prediction] | None = None) -> dict:
    if not predictions:
        raise ValueError("cannot summarize an empty prediction list")
    correct = np.asarray([item.predicted == item.label for item in predictions])
    by_domain: dict[str, list[bool]] = defaultdict(list)
    by_modality: dict[str, list[bool]] = defaultdict(list)
    for item, is_correct in zip(predictions, correct):
        by_domain[item.domain].append(bool(is_correct))
        by_modality[item.modality].append(bool(is_correct))

    result = {
        "n": len(predictions),
        "accuracy": float(np.mean(correct)),
        "hallucination_rate": float(1.0 - np.mean(correct)),
        "ece": expected_calibration_error(predictions),
        "mean_intervention_gate": float(np.mean([item.intervention_gate for item in predictions])),
        "mean_erasure": float(np.mean([item.erasure for item in predictions])),
        "domain_accuracy": {key: float(np.mean(values)) for key, values in sorted(by_domain.items())},
        "modality_accuracy": {
            key: float(np.mean(values)) for key, values in sorted(by_modality.items())
        },
    }
    result["worst_domain_accuracy"] = min(result["domain_accuracy"].values())

    if base is not None:
        base_by_id = {item.sample_id: item for item in base}
        rescued = harmed = unchanged = 0
        exact_equal = 0
        for item in predictions:
            reference = base_by_id[item.sample_id]
            base_correct = reference.predicted == reference.label
            now_correct = item.predicted == item.label
            rescued += int(not base_correct and now_correct)
            harmed += int(base_correct and not now_correct)
            unchanged += int(base_correct == now_correct)
            exact_equal += int(item.predicted == reference.predicted)
        result.update(
            {
                "rescued": rescued,
                "harmed": harmed,
                "unchanged_correctness": unchanged,
                "prediction_equal_to_generalist": exact_equal / len(predictions),
            }
        )
    return result
