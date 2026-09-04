from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .claims import ClaimSpec
from .expert_bridge import LazyConceptExpertProvider
from .extract import _expert_from_spec, _local_or_remote, _release
from .feddg import FederatedReliabilityCalibrator
from .generalist import QwenLayerProbe
from .med_defer import (
    ClaimDeferralController,
    ClaimRequest,
    DomainSignal,
    DomainTrustCalibrator,
    ExpertCard,
    LazyExpertPool,
    MedDeferEngine,
)
from .med_defer_study import _domain_scores
from .routing import MetadataRouter, normalized_entropy
from .types import EvidenceRecord


def _candidate_uncertainty(logits: np.ndarray) -> float:
    """Normalized entropy in the real closed-set candidate space."""

    values = np.asarray(logits, dtype=float)
    shifted = values - float(np.max(values))
    probabilities = np.exp(np.clip(shifted, -60.0, 60.0))
    probabilities /= max(float(np.sum(probabilities)), 1e-12)
    entropy = -float(np.sum(probabilities * np.log(np.clip(probabilities, 1e-12, 1.0))))
    return float(entropy / np.log(len(values)))


def _explanation_prompt(question: str, locked_answer: str) -> str:
    return (
        f"{question}\n\n"
        f"The selected answer is locked as: {locked_answer}. "
        "Briefly explain the image evidence for that selected answer. "
        "Do not reconsider, replace, or output a different answer."
    )


def _candidate_signature(candidates: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    return tuple(" ".join(value.strip().casefold().split()) for value in candidates)


def _route_case(
    case: dict[str, Any], model_config: dict, artifact_root: str | Path | None
) -> dict[str, float]:
    available = sorted(
        {
            str(modality)
            for expert_id, spec in model_config["experts"].items()
            for modality in spec.get("modalities", [spec.get("modality", expert_id)])
        }
    )
    if len(available) == 1:
        return {available[0]: 1.0}
    broad_spec = model_config["broad_specialist"]
    if broad_spec.get("adapter") == "contrastive_biomedclip":
        router = _expert_from_spec(broad_spec, artifact_root)
        try:
            route = router.route(case["image"], available)
        finally:
            # Drop the owning reference before collecting so the router's GPU
            # tensors cannot remain resident while the generalist is loaded.
            del router
            _release()
    else:
        route = MetadataRouter(available).route(case["image"], case.get("metadata"))
    threshold = float(model_config["router"].get("abstain_entropy", 1.0))
    if normalized_entropy(route) >= threshold:
        return {name: 1.0 / len(available) for name in available}
    return route


def guided_generate_case(
    case: dict[str, Any],
    model_config: dict,
    comparison_config: dict,
    source_records: list[EvidenceRecord],
    artifact_root: str | Path | None = None,
    max_new_tokens: int = 96,
) -> dict[str, Any]:
    """Make a locked closed-set decision before optionally generating an explanation.

    The medical VLM first supplies real candidate sequence likelihoods. Med-DEFER
    then performs exactly one pre-generation decision in candidate space; the
    resulting argmax is the returned answer. Any later free-text generation is an
    explanation only and cannot overwrite that answer. This function deliberately
    makes no claim that later open-ended clinical claims are already supported.
    """

    required = {"id", "image", "prompt", "candidates"}
    missing = required - case.keys()
    if missing:
        raise ValueError(f"guided case is missing: {sorted(missing)}")
    raw_candidates = case["candidates"]
    if not isinstance(raw_candidates, (list, tuple)):
        raise TypeError("candidates must be a list or tuple")
    concepts = tuple(" ".join(str(value).strip().split()) for value in raw_candidates)
    if len(concepts) < 2 or any(not concept for concept in concepts):
        raise ValueError("guided generation needs at least two non-empty candidates")
    signature = _candidate_signature(list(concepts))
    if len(set(signature)) != len(signature):
        raise ValueError("guided generation candidates must be unique")
    case_metadata = dict(case.get("metadata") or {})
    route = _route_case(case, model_config, artifact_root)
    routed_modality = max(route, key=route.get)
    claim = ClaimSpec.from_vqa(
        claim_id="diagnostic-claim-0",
        question=str(case["prompt"]),
        candidates=concepts,
        modality=routed_modality,
        required_capabilities=("classification",),
        metadata=case_metadata,
    )

    source_domains = set(comparison_config["source_domains"])
    source = [
        record
        for record in source_records
        if record.domain in source_domains
        and record.modality == routed_modality
        and _candidate_signature(record.candidates) == signature
    ]
    if not source:
        raise ValueError(
            "source cache contains no records with the same source domain, modality, "
            "and ordered candidate vocabulary"
        )
    method = comparison_config["method"]
    reliability = FederatedReliabilityCalibrator(
        prior=float(method.get("reliability_prior", 4.0)),
        lcb_z=float(method.get("reliability_lcb_z", 1.0)),
    )
    reliability.fit(source, source_domains)
    validation_scores = _domain_scores(source, source_domains)
    settings = comparison_config.get("med_defer", model_config.get("med_defer", {}))
    trust_settings = settings.get("domain_trust", {})
    control_settings = settings.get("controller", {})
    guidance_settings = settings.get("guidance", {})
    trust = DomainTrustCalibrator(
        cvar_alpha=float(trust_settings.get("cvar_alpha", 0.25)),
        ood_temperature=float(trust_settings.get("ood_temperature", 2.0)),
    )
    controller = ClaimDeferralController(
        trust,
        uncertainty_threshold=float(control_settings.get("uncertainty_threshold", 0.35)),
        minimum_utility=float(control_settings.get("minimum_utility", 0.08)),
        cost_weight=float(control_settings.get("cost_weight", 0.05)),
    )
    engine = MedDeferEngine(
        controller,
        guidance_strength=float(guidance_settings.get("strength", 0.8)),
        max_bias_norm=float(guidance_settings.get("max_bias_norm", 1.25)),
        minimum_post_call_trust=float(trust_settings.get("minimum_post_call_trust", 0.05)),
    )
    pool = LazyExpertPool()
    providers: list[LazyConceptExpertProvider] = []
    for expert_id, spec in sorted(model_config["experts"].items()):
        capabilities = tuple(spec.get("capabilities", ["classification"]))
        card = ExpertCard(
            expert_id=expert_id,
            modalities=tuple(spec.get("modalities", [spec.get("modality", expert_id)])),
            capabilities=capabilities,
            source_reliability_lcb=reliability.score(expert_id),
            validation_domain_scores=validation_scores.get(expert_id, ()),
            expected_gain=float(spec.get("expected_gain", 1.0)),
            latency_ms=float(spec.get("latency_ms", 100.0)),
        )
        provider = LazyConceptExpertProvider(
            expert_id,
            "classification",
            lambda spec=spec: _expert_from_spec(spec, artifact_root),
            case["image"],
            case["prompt"],
        )
        providers.append(provider)
        pool.register(card, provider)

    general_spec = model_config["generalist"]
    generalist = QwenLayerProbe(
        _local_or_remote(general_spec["id"], artifact_root),
        layers=general_spec["layers"],
        dtype=general_spec.get("dtype", "bfloat16"),
        device_map=general_spec.get("device_map", "auto"),
    )
    first_claim_policy = str(control_settings.get("first_claim_policy", "qualified_first_claim"))
    expert_ood_scores = case_metadata.get("expert_ood_scores")
    if expert_ood_scores is not None and not isinstance(expert_ood_scores, dict):
        raise ValueError("metadata.expert_ood_scores must be a mapping")
    expert_ood_scores = expert_ood_scores or {}
    try:
        base_candidate_logits = generalist.candidate_log_likelihoods(
            case["image"], case["prompt"], list(concepts)
        )
        request = ClaimRequest(
            sample_id=str(case["id"]),
            claim_id=claim.claim_id,
            modality=routed_modality,
            required_capabilities=("classification",),
            concepts=concepts,
            base_logits=tuple(float(value) for value in base_candidate_logits),
            uncertainty=_candidate_uncertainty(base_candidate_logits),
            router_probs=route,
            domain_signals={
                expert_id: DomainSignal(
                    ood_score=float(expert_ood_scores[expert_id]),
                    image_quality=float(case_metadata.get("image_quality", 1.0)),
                )
                for expert_id in pool.cards
                if expert_id in expert_ood_scores
            },
            question=str(case["prompt"]),
            generated_prefix="",
            task_type=str(case.get("task_type", "closed_set")),
            deferral_policy=first_claim_policy,
            expert_queries=claim.expert_queries,
        )
        first_trace = engine.guide(request, pool)
        selected_index = int(np.argmax(first_trace.guided_logits))
        answer = concepts[selected_index]

        generate_explanation = bool(case.get("generate_explanation", False))
        if generate_explanation:
            explanation = generalist.generate(
                case["image"],
                _explanation_prompt(str(case["prompt"]), answer),
                max_new_tokens=max_new_tokens,
            )
            generation_mode = "closed_set_locked_answer_with_explanation"
        else:
            explanation = None
            generation_mode = "closed_set_locked_answer_only"
        loaded_experts = [
            expert_id
            for expert_id, provider in zip(sorted(model_config["experts"]), providers)
            if provider.loaded
        ]
    finally:
        for provider in providers:
            provider.release()
        del generalist
        _release()

    return {
        "sample_id": str(case["id"]),
        "answer": answer,
        "explanation": explanation,
        "generation_mode": generation_mode,
        "trace": first_trace.to_json(),
        "candidate_answers": list(concepts),
        "generalist_candidate_log_likelihoods": [float(value) for value in base_candidate_logits],
        "first_claim_selected_answer": answer,
        "first_claim_policy": first_claim_policy,
        "route": route,
        "selected_modality": routed_modality,
        "expert_calls": pool.call_counts,
        "loaded_experts": loaded_experts,
        "claim_traces": [first_trace.to_json()],
        "open_claim_decoding_validated": False,
        "target_label_used": False,
    }
