from __future__ import annotations

import hashlib

import numpy as np

from .types import EvidenceRecord


def _seed(base: int, *parts: object) -> int:
    text = "|".join([str(base), *(str(part) for part in parts)])
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16) % (2**32)


def _router_probs(
    rng: np.random.Generator,
    modalities: list[str],
    truth: str,
    accuracy: float,
) -> dict[str, float]:
    if rng.random() < accuracy:
        selected = truth
    else:
        selected = rng.choice([name for name in modalities if name != truth]).item()
    peak = float(rng.uniform(0.70, 0.96))
    remainder = (1.0 - peak) / (len(modalities) - 1)
    return {name: peak if name == selected else remainder for name in modalities}


def simulate_records(config: dict, repetition: int = 0) -> list[EvidenceRecord]:
    seed = int(config["seed"]) + repetition * 10_007
    modalities = list(config["modalities"])
    domains = list(config["source_domains"]) + list(config["target_domains"])
    target_domains = set(config["target_domains"])
    layers = list(config["layers"])
    count = int(config["samples_per_domain"])
    records: list[EvidenceRecord] = []

    for domain in domains:
        for index in range(count):
            rng = np.random.default_rng(_seed(seed, domain, index))
            modality = modalities[index % len(modalities)]
            candidates = list(config["concepts"][modality])
            label = int(rng.integers(0, len(candidates)))
            truth = np.full(len(candidates), -0.55)
            truth[label] = 0.95

            # The null pass carries a modest language prior. Target domains have
            # stronger shortcut pressure and more late-layer visual erasure.
            prior_index = 0 if domain != "hospital_b" else 1
            null_logits = rng.normal(0.0, 0.18, len(candidates))
            null_logits[prior_index] += 0.45
            target = domain in target_domains
            erasure_strength = rng.uniform(0.45, 0.90) if target else rng.uniform(0.10, 0.50)

            visual_layers = []
            for position, _ in enumerate(layers):
                phase = position / max(1, len(layers) - 1)
                early_gain = 1.15 - 0.20 * phase
                erased = erasure_strength * (phase**2)
                vector = early_gain * truth - erased * truth
                vector += rng.normal(0.0, 0.18 + 0.08 * phase, len(candidates))
                visual_layers.append(vector)

            expert_scores: dict[str, np.ndarray] = {}
            for expert_modality in modalities:
                if expert_modality == modality:
                    scores = 1.20 * truth + rng.normal(0.0, 0.16, len(candidates))
                else:
                    scores = rng.normal(0.0, 0.50, len(candidates))
                expert_scores[expert_modality] = scores

            broad_scores = 0.58 * truth + rng.normal(0.0, 0.34, len(candidates))
            route_accuracy = (
                config["router"]["target_accuracy"]
                if target
                else config["router"]["source_accuracy"]
            )
            router = _router_probs(rng, modalities, modality, float(route_accuracy))
            records.append(
                EvidenceRecord(
                    sample_id=f"{domain}-{index:05d}",
                    domain=domain,
                    modality=modality,
                    candidates=candidates,
                    label=label,
                    general_null_logits=np.asarray(null_logits, dtype=float),
                    general_visual_layers=np.asarray(visual_layers, dtype=float),
                    expert_scores=expert_scores,
                    broad_specialist_scores=np.asarray(broad_scores, dtype=float),
                    router_probs=router,
                    metadata={"synthetic": True, "target_domain": target},
                )
            )
    return records
