from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from .evaluation import summarize
from .feddg import FederatedReliabilityCalibrator
from .io import save_json, save_predictions, save_records
from .methods import predict, selected_route
from .types import EvidenceRecord, Prediction


def make_oracle_records(records: list[EvidenceRecord], peak: float = 0.98) -> list[EvidenceRecord]:
    """Replace routing probabilities without rerunning any vision-language model."""

    for record in records:
        available = sorted(record.expert_scores)
        compatible = list(record.compatible_experts())
        if not compatible:
            raise ValueError(f"no oracle expert is available for modality {record.modality!r}")
        modalities = sorted(
            {
                modality
                for expert_id in available
                for modality in record.modalities_for_expert(expert_id)
            }
        )
        other_modalities = [name for name in modalities if name != record.modality]
        if not other_modalities:
            modality_probs = {record.modality: 1.0}
        else:
            modality_probs = {
                name: peak if name == record.modality else (1.0 - peak) / len(other_modalities)
                for name in modalities
            }
        experts_per_modality = {
            modality: sum(
                modality in record.modalities_for_expert(expert_id) for expert_id in available
            )
            for modality in modalities
        }
        record.router_probs = {
            expert_id: sum(
                modality_probs.get(modality, 0.0) / experts_per_modality[modality]
                for modality in record.modalities_for_expert(expert_id)
            )
            for expert_id in available
        }
        record.metadata = {
            **dict(record.metadata or {}),
            "modality_router_probs": modality_probs,
        }
    return records


def route_metrics(records: list[EvidenceRecord]) -> dict:
    correct = []
    confidences = []
    for record in records:
        probabilities = record.modality_probabilities()
        route = max(probabilities, key=probabilities.get)
        correct.append(route == record.modality)
        confidences.append(float(probabilities[route]))
    return {
        "accuracy": float(np.mean(correct)),
        "mean_confidence": float(np.mean(confidences)),
        "overconfidence_gap": float(np.mean(confidences) - np.mean(correct)),
    }


def _shuffled_scores(records: list[EvidenceRecord], seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    result: dict[str, np.ndarray] = {}
    grouped: dict[tuple[str, int], list[EvidenceRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.modality, len(record.candidates))].append(record)
    for items in grouped.values():
        order = rng.permutation(len(items))
        if len(items) > 1 and np.all(order == np.arange(len(items))):
            order = np.roll(order, 1)
        for record, shuffled_index in zip(items, order):
            donor = items[int(shuffled_index)]
            route, _ = selected_route(record)
            donor_choices = donor.compatible_experts()
            if route in donor.expert_scores:
                result[record.sample_id] = donor.expert_scores[route]
            elif donor_choices:
                result[record.sample_id] = donor.expert_scores[donor_choices[0]]
            else:
                raise ValueError(f"donor {donor.sample_id} has no compatible expert")
    return result


def compare_records(
    records: list[EvidenceRecord],
    config: dict,
    output: str | Path | None = None,
    repetition: int = 0,
) -> dict:
    source_domains = set(config["source_domains"])
    target_domains = set(config["target_domains"])
    source = [record for record in records if record.domain in source_domains]
    target = [record for record in records if record.domain in target_domains]
    if not source or not target:
        raise ValueError("both source and held-out target records are required")

    settings = config["method"]
    calibrator = FederatedReliabilityCalibrator(
        prior=float(settings["reliability_prior"]),
        lcb_z=float(settings["reliability_lcb_z"]),
    )
    calibrator.fit(source, source_domains)
    shuffled = _shuffled_scores(target, int(config["seed"]) + repetition)

    methods = list(config["evaluation"]["methods"])
    predictions: dict[str, list[Prediction]] = {}
    for method in methods:
        predictions[method] = [
            predict(
                record,
                method,
                settings,
                calibrator=calibrator,
                shuffled_expert_scores=shuffled.get(record.sample_id),
            )
            for record in target
        ]

    base = predictions["generalist"]
    metrics = {
        method: summarize(items, None if method == "generalist" else base)
        for method, items in predictions.items()
    }
    report = {
        "protocol": config.get("evaluation_protocol", "source-only leave-one-domain-out"),
        "strict_hospital_dg_claim_allowed": bool(
            config.get("strict_hospital_dg_claim_allowed", True)
        ),
        "target_labels_used_during_fit": False,
        "source_domains": sorted(source_domains),
        "target_domains": sorted(target_domains),
        "source_samples": len(source),
        "target_samples": len(target),
        "route": route_metrics(target),
        "federated_calibration": calibrator.report(),
        "metrics": metrics,
    }

    if output is not None:
        directory = Path(output)
        directory.mkdir(parents=True, exist_ok=True)
        save_records(directory / "evidence.jsonl", records)
        save_json(directory / "metrics.json", report)
        save_json(directory / "config.resolved.json", config)
        save_json(directory / "federated_calibration.json", calibrator.report())
        for method, items in predictions.items():
            save_predictions(directory / f"predictions.{method}.jsonl", items)
        write_markdown_table(directory / "comparison.md", report)
    return report


def write_markdown_table(path: Path, report: dict) -> None:
    lines = [
        "# Held-out domain comparison",
        "",
        "Target-domain labels were not used during fitting or calibration.",
        "",
        "| Method | Accuracy | Hallucination ↓ | Worst domain | Rescue | Harm | Same as base |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method, values in report["metrics"].items():
        lines.append(
            "| {method} | {accuracy:.4f} | {hallucination_rate:.4f} | "
            "{worst_domain_accuracy:.4f} | {rescued} | {harmed} | {same:.4f} |".format(
                method=method,
                accuracy=values["accuracy"],
                hallucination_rate=values["hallucination_rate"],
                worst_domain_accuracy=values["worst_domain_accuracy"],
                rescued=values.get("rescued", "—"),
                harmed=values.get("harmed", "—"),
                same=values.get("prediction_equal_to_generalist", 1.0),
            )
        )
    lines.extend(
        [
            "",
            "The shuffled- and wrong-expert controls are mandatory mechanism checks, not optional ablations.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def aggregate_repetitions(reports: list[dict]) -> dict:
    methods = reports[0]["metrics"]
    summary = {}
    for method in methods:
        summary[method] = {}
        for metric in ("accuracy", "hallucination_rate", "worst_domain_accuracy", "ece"):
            values = [report["metrics"][method][metric] for report in reports]
            summary[method][metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            }
    return {"repetitions": len(reports), "aggregate": summary}
