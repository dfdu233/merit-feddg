import numpy as np

from merit_feddg.expert_bridge import LazyConceptExpertProvider
from merit_feddg.io import load_yaml
from merit_feddg.med_defer import (
    ClaimDeferralController,
    ClaimRequest,
    DomainSignal,
    DomainTrustCalibrator,
    ExpertCard,
    LazyExpertPool,
    MedDeferEngine,
    NativeEvidence,
)
from merit_feddg.med_defer_study import run_med_defer_study


def _request(uncertainty=0.9, ood=0.0):
    return ClaimRequest(
        sample_id="case-1",
        claim_id="claim-0",
        modality="cxr",
        required_capabilities=("classification",),
        concepts=("normal", "pneumothorax"),
        base_logits=(0.4, 0.2),
        uncertainty=uncertainty,
        router_probs={"cxr": 0.95, "oct": 0.05},
        domain_signals={"cxr-expert": DomainSignal(ood_score=ood)},
    )


def _engine():
    trust = DomainTrustCalibrator(cvar_alpha=0.5, ood_temperature=3.0)
    controller = ClaimDeferralController(trust, minimum_utility=0.01)
    return MedDeferEngine(controller, guidance_strength=1.0, max_bias_norm=0.5)


def _pool(call_counter):
    pool = LazyExpertPool()
    card = ExpertCard(
        expert_id="cxr-expert",
        modalities=("cxr",),
        capabilities=("classification",),
        source_reliability_lcb=0.9,
        validation_domain_scores=(0.85, 0.8, 0.2),
    )

    def provider(request):
        call_counter.append(request.claim_id)
        return NativeEvidence(
            expert_id="cxr-expert",
            capability="classification",
            concept_scores={"normal": -2.0, "pneumothorax": 4.0},
            confidence=0.95,
            masks=({"rle": "native-mask-is-preserved"},),
        )

    pool.register(card, provider)
    return pool


def test_sparse_deferral_calls_only_selected_expert_once_and_caches():
    calls = []
    pool = _pool(calls)
    engine = _engine()
    first = engine.guide(_request(), pool)
    second = engine.guide(_request(), pool)
    assert first.selected_expert == "cxr-expert"
    assert first.expert_called and not first.cache_hit
    assert second.cache_hit
    assert calls == ["claim-0"]
    assert np.linalg.norm(first.concept_delta) <= 0.5 + 1e-9


def test_confident_generalist_abstains_without_calling_expert():
    calls = []
    trace = _engine().guide(_request(uncertainty=0.1), _pool(calls))
    assert trace.selected_expert is None
    assert trace.reason == "generalist-confident"
    assert calls == []


def test_ood_signal_suppresses_guidance():
    in_domain = _engine().guide(_request(ood=0.0), _pool([]))
    shifted = _engine().guide(_request(ood=1.0), _pool([]))
    assert shifted.trust < in_domain.trust
    assert shifted.gate < in_domain.gate


def test_end_to_end_study_is_sparse_and_source_only():
    report = run_med_defer_study(load_yaml("configs/smoke.yaml"))
    assert report["sparse_calls"] <= report["target_samples"]
    assert report["sparse_calls"] < report["dense_call_baseline"]
    assert report["source_only_domain_trust"]
    assert not report["target_labels_used_for_routing_or_trust"]


def test_existing_expert_is_not_loaded_until_deferred():
    calls = []

    class StubExpert:
        def image_null_scores(self, image, prompt, concepts):
            return np.asarray([-1.0, 2.0])

    provider = LazyConceptExpertProvider(
        "cxr-expert",
        "classification",
        lambda: calls.append("loaded") or StubExpert(),
        "unused-by-stub.png",
        "Is there an acute finding?",
    )
    assert not provider.loaded
    evidence = provider(_request())
    assert provider.loaded
    assert calls == ["loaded"]
    assert evidence.concept_scores["pneumothorax"] == 2.0


def test_transformers_processor_defers_again_at_next_claim_boundary():
    torch = __import__("pytest").importorskip("torch")
    from merit_feddg.decoding import MedDeferLogitsProcessor

    class Tokenizer:
        def __init__(self):
            self.mapping = {" normal": [3], " pneumothorax": [4]}

        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": self.mapping[text]}

        def decode(self, tokens, skip_special_tokens=True):
            values = tokens.tolist() if hasattr(tokens, "tolist") else list(tokens)
            return "." if values and values[-1] == 9 else "finding"

    calls = []
    pool = _pool(calls)

    def request_factory(prefix, uncertainty, claim_index):
        request = _request(uncertainty=uncertainty)
        return ClaimRequest(**{**request.__dict__, "claim_id": f"claim-{claim_index}"})

    processor = MedDeferLogitsProcessor(
        Tokenizer(), _engine(), pool, request_factory, top_k_entropy=8
    )
    initial = torch.tensor([[10, 11]])
    processor(initial, torch.zeros((1, 12)))
    processor(torch.tensor([[10, 11, 5]]), torch.zeros((1, 12)))
    processor(torch.tensor([[10, 11, 5, 9]]), torch.zeros((1, 12)))
    assert calls == ["claim-0", "claim-1"]
    assert len(processor.traces) == 2
