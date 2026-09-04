from __future__ import annotations

from collections import defaultdict

import numpy as np

from .feddg import FederatedReliabilityCalibrator
from .med_defer import (
    ClaimDeferralController,
    ClaimRequest,
    DomainSignal,
    DomainTrustCalibrator,
    ExpertCard,
    LazyExpertPool,
    MedDeferEngine,
    NativeEvidence,
)
from .simulation import simulate_records
from .types import EvidenceRecord


def _margin_confidence(scores: np.ndarray) -> float:
    ordered = np.sort(np.asarray(scores, dtype=float))
    margin = float(ordered[-1] - ordered[-2])
    return float(1.0 / (1.0 + np.exp(-margin)))


def _domain_scores(records, source_domains: set[str]) -> dict[str, tuple[float, ...]]:
    counts: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        if record.domain not in source_domains:
            continue
        for expert, scores in record.expert_scores.items():
            if expert == record.modality:
                counts[expert][record.domain].append(int(np.argmax(scores) == record.label))
    return {
        expert: tuple(float(np.mean(values)) for _, values in sorted(domains.items()))
        for expert, domains in counts.items()
    }


def run_med_defer_records(records: list[EvidenceRecord], config: dict) -> dict:
    """Evaluate claim deferral on a reusable cache without refitting on target labels."""

    source_domains = set(config["source_domains"])
    targets = [record for record in records if record.domain in set(config["target_domains"])]
    calibrator = FederatedReliabilityCalibrator(
        prior=float(config["method"].get("reliability_prior", 4.0)),
        lcb_z=float(config["method"].get("reliability_lcb_z", 1.0)),
    )
    calibrator.fit(records, source_domains)
    domain_scores = _domain_scores(records, source_domains)
    settings = config.get("med_defer", {})
    trust_settings = settings.get("domain_trust", {})
    controller_settings = settings.get("controller", {})
    guidance = settings.get("guidance", {})
    trust_calibrator = DomainTrustCalibrator(
        cvar_alpha=float(trust_settings.get("cvar_alpha", 0.25)),
        ood_temperature=float(trust_settings.get("ood_temperature", 2.0)),
    )
    controller = ClaimDeferralController(
        trust_calibrator,
        uncertainty_threshold=float(controller_settings.get("uncertainty_threshold", 0.35)),
        minimum_utility=float(controller_settings.get("minimum_utility", 0.08)),
        cost_weight=float(controller_settings.get("cost_weight", 0.05)),
    )
    engine = MedDeferEngine(
        controller,
        guidance_strength=float(guidance.get("strength", 0.8)),
        max_bias_norm=float(guidance.get("max_bias_norm", 1.25)),
        minimum_post_call_trust=float(trust_settings.get("minimum_post_call_trust", 0.05)),
    )

    modalities = sorted({expert for record in records for expert in record.expert_scores})
    cards = {
        modality: ExpertCard(
            expert_id=modality,
            modalities=(modality,),
            capabilities=("classification",),
            source_reliability_lcb=calibrator.score(modality),
            validation_domain_scores=domain_scores.get(modality, ()),
            expected_gain=1.0,
            latency_ms=100.0,
        )
        for modality in modalities
    }
    correct_base = 0
    correct_guided = 0
    abstained = 0
    traces = []
    total_calls: dict[str, int] = defaultdict(int)

    for record in targets:
        pool = LazyExpertPool()
        for expert_id, card in cards.items():
            scores = np.asarray(record.expert_scores[expert_id], dtype=float)

            def provider(request, expert_id=expert_id, scores=scores):
                return NativeEvidence(
                    expert_id=expert_id,
                    capability="classification",
                    concept_scores=dict(zip(request.concepts, scores)),
                    confidence=_margin_confidence(scores),
                    provenance={"synthetic": True, "sample_id": request.sample_id},
                )

            pool.register(card, provider)

        base = np.asarray(record.general_final_logits, dtype=float)
        probabilities = np.exp(base - np.max(base))
        probabilities /= probabilities.sum()
        entropy = -float(np.sum(probabilities * np.log(probabilities + 1e-12)))
        uncertainty = entropy / np.log(len(probabilities))
        route_peak = max(record.router_probs.values())
        request = ClaimRequest(
            sample_id=record.sample_id,
            claim_id="diagnostic-claim-0",
            modality=max(record.router_probs, key=record.router_probs.get),
            required_capabilities=("classification",),
            concepts=tuple(record.candidates),
            base_logits=tuple(base),
            uncertainty=float(np.clip(uncertainty, 0.0, 1.0)),
            router_probs=record.router_probs,
            domain_signals={
                expert_id: DomainSignal(ood_score=1.0 - route_peak, image_quality=1.0)
                for expert_id in cards
            },
        )
        trace = engine.guide(request, pool)
        traces.append(trace.to_json())
        correct_base += int(np.argmax(base) == record.label)
        correct_guided += int(np.argmax(trace.guided_logits) == record.label)
        abstained += int(trace.selected_expert is None)
        for expert_id, calls in pool.call_counts.items():
            total_calls[expert_id] += calls

    total = len(targets)
    return {
        "method": "med_defer",
        "granularity": "clinical_claim",
        "target_samples": total,
        "generalist_accuracy": correct_base / max(total, 1),
        "med_defer_accuracy": correct_guided / max(total, 1),
        "abstentions": abstained,
        "expert_calls": dict(sorted(total_calls.items())),
        "dense_call_baseline": total * len(cards),
        "sparse_calls": sum(total_calls.values()),
        "source_only_domain_trust": True,
        "target_labels_used_for_routing_or_trust": False,
        "traces": traces,
    }


def run_med_defer_study(config: dict, repetition: int = 0) -> dict:
    """Run the sparse conditional-computation path on deterministic records."""

    return run_med_defer_records(simulate_records(config, repetition=repetition), config)
