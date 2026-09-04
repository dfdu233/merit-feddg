from __future__ import annotations

from pathlib import Path
from typing import Any

from .decoding import MedDeferLogitsProcessor
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


def _route_case(
    case: dict[str, Any], model_config: dict, artifact_root: str | Path | None
) -> dict[str, float]:
    available = sorted(model_config["experts"])
    broad_spec = model_config["broad_specialist"]
    if broad_spec.get("adapter") == "contrastive_biomedclip":
        router = _expert_from_spec(broad_spec, artifact_root)
        try:
            route = router.route(case["image"], available)
        finally:
            _release(router)
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
    """Run a real medical VLM with lazy claim-level specialist calls.

    The case needs an image, prompt and at least two candidate clinical concepts,
    but it deliberately does not require a target-domain label.
    """

    required = {"id", "image", "prompt", "candidates"}
    missing = required - case.keys()
    if missing:
        raise ValueError(f"guided case is missing: {sorted(missing)}")
    concepts = tuple(str(value) for value in case["candidates"])
    if len(concepts) < 2:
        raise ValueError("guided generation needs at least two candidate concepts")

    source_domains = set(comparison_config["source_domains"])
    source = [record for record in source_records if record.domain in source_domains]
    if not source:
        raise ValueError("source cache contains no configured source-domain records")
    method = comparison_config["method"]
    reliability = FederatedReliabilityCalibrator(
        prior=float(method.get("reliability_prior", 4.0)),
        lcb_z=float(method.get("reliability_lcb_z", 1.0)),
    )
    reliability.fit(source, source_domains)
    validation_scores = _domain_scores(source, source_domains)
    route = _route_case(case, model_config, artifact_root)
    routed_modality = max(route, key=route.get)

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
        minimum_post_call_trust=float(
            trust_settings.get("minimum_post_call_trust", 0.05)
        ),
    )
    pool = LazyExpertPool()
    providers: list[LazyConceptExpertProvider] = []
    for expert_id, spec in sorted(model_config["experts"].items()):
        capabilities = tuple(spec.get("capabilities", ["classification"]))
        card = ExpertCard(
            expert_id=expert_id,
            modalities=(expert_id,),
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

    route_peak = float(route[routed_modality])

    def request_factory(prefix: str, uncertainty: float, claim_index: int) -> ClaimRequest:
        return ClaimRequest(
            sample_id=str(case["id"]),
            claim_id=f"claim-{claim_index}",
            modality=routed_modality,
            required_capabilities=("classification",),
            concepts=concepts,
            base_logits=tuple(0.0 for _ in concepts),
            uncertainty=uncertainty,
            router_probs=route,
            domain_signals={
                expert_id: DomainSignal(
                    ood_score=1.0 - route_peak,
                    image_quality=float(case.get("metadata", {}).get("image_quality", 1.0)),
                )
                for expert_id in pool.cards
            },
        )

    general_spec = model_config["generalist"]
    generalist = QwenLayerProbe(
        _local_or_remote(general_spec["id"], artifact_root),
        layers=general_spec["layers"],
        dtype=general_spec.get("dtype", "bfloat16"),
        device_map=general_spec.get("device_map", "auto"),
    )
    tokenizer = getattr(generalist.processor, "tokenizer", generalist.processor)
    processor = MedDeferLogitsProcessor(
        tokenizer,
        engine,
        pool,
        request_factory,
        uncertainty_threshold=float(control_settings.get("uncertainty_threshold", 0.35)),
    )
    try:
        answer = generalist.generate(
            case["image"],
            case["prompt"],
            max_new_tokens=max_new_tokens,
            logits_processor=processor,
        )
    finally:
        _release(generalist)

    return {
        "sample_id": str(case["id"]),
        "answer": answer,
        "route": route,
        "selected_modality": routed_modality,
        "expert_calls": pool.call_counts,
        "loaded_experts": [
            expert_id
            for expert_id, provider in zip(sorted(model_config["experts"]), providers)
            if provider.loaded
        ],
        "claim_traces": [trace.to_json() for trace in processor.traces],
        "target_label_used": False,
    }
