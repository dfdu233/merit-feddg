import numpy as np
import pytest

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


def test_qualified_first_claim_checks_a_high_confidence_answer():
    calls = []
    request = _request(uncertainty=0.01)
    request = ClaimRequest(**{**request.__dict__, "deferral_policy": "qualified_first_claim"})
    trace = _engine().guide(request, _pool(calls))
    assert trace.selected_expert == "cxr-expert"
    assert calls == ["claim-0"]
    assert trace.gate > 0


def test_ood_signal_suppresses_guidance():
    in_domain = _engine().guide(_request(ood=0.0), _pool([]))
    shifted = _engine().guide(_request(ood=1.0), _pool([]))
    assert shifted.trust < in_domain.trust
    assert shifted.gate < in_domain.gate


def test_source_lcb_and_lower_tail_use_geometric_not_repeated_product_attenuation():
    card = ExpertCard(
        expert_id="qualified",
        modalities=("cxr",),
        capabilities=("classification",),
        source_reliability_lcb=0.25,
        validation_domain_scores=(0.25, 0.25),
    )
    trust = DomainTrustCalibrator(cvar_alpha=1.0, ood_temperature=1.0).score(card, DomainSignal())
    assert trust.score == 0.25
    assert trust.score > card.source_reliability_lcb * trust.lower_tail_stability


def test_expert_without_source_domain_qualification_is_not_selected():
    pool = LazyExpertPool()
    pool.register(
        ExpertCard(
            expert_id="unqualified-cxr-expert",
            modalities=("cxr",),
            capabilities=("classification",),
            source_reliability_lcb=0.9,
            validation_domain_scores=(),
        ),
        lambda request: NativeEvidence(
            expert_id="unqualified-cxr-expert",
            capability="classification",
            concept_scores={"normal": -2.0, "pneumothorax": 4.0},
            confidence=0.95,
        ),
    )

    trace = _engine().guide(_request(), pool)

    assert trace.selected_expert is None
    assert trace.reason == "no-compatible-expert"
    assert pool.call_counts["unqualified-cxr-expert"] == 0


def test_missing_domain_signal_fails_closed_before_expert_loading():
    calls = []
    request = ClaimRequest(**{**_request().__dict__, "domain_signals": {}})
    trace = _engine().guide(request, _pool(calls))
    assert trace.selected_expert is None
    assert trace.reason == "missing-domain-signal"
    assert calls == []


def test_semantic_request_rejects_third_party_evidence_without_validated_bridge():
    calls = []
    request = ClaimRequest(
        **{
            **_request().__dict__,
            "expert_queries": (
                "The image is normal.",
                "The image shows pneumothorax.",
            ),
            "deferral_policy": "qualified_first_claim",
        }
    )
    trace = _engine().guide(request, _pool(calls))
    assert calls == ["claim-0"]
    assert trace.reason == "semantic-bridge-not-validated"
    assert trace.gate == 0.0
    assert trace.guided_logits == trace.base_logits


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
    assert evidence.concept_scores["pneumothorax"] == pytest.approx(0.9051482536)
    assert evidence.provenance["semantic_bridge_validated"] is True


def test_provider_passes_current_question_prefix_and_semantic_queries():
    observed = {}

    class StubExpert:
        def score_claims(self, image, question, generated_prefix, claims):
            observed.update(
                question=question,
                generated_prefix=generated_prefix,
                claims=claims,
            )
            return np.asarray([-1.0, 2.0])

    provider = LazyConceptExpertProvider(
        "cxr-expert",
        "classification",
        StubExpert,
        "unused-by-stub.png",
        "legacy prompt",
    )
    request = _request()
    request = ClaimRequest(
        **{
            **request.__dict__,
            "question": "What is the current diagnosis?",
            "generated_prefix": "There is a focal opacity.",
            "expert_queries": (
                "The image is normal.",
                "The image shows pneumothorax.",
            ),
        }
    )
    evidence = provider(request)
    assert observed["question"] == "What is the current diagnosis?"
    assert observed["generated_prefix"] == "There is a focal opacity."
    assert observed["claims"] == list(request.expert_queries)
    assert set(evidence.concept_scores) == set(request.concepts)
    assert evidence.provenance["semantic_bridge_validated"] is True


def test_transformers_processor_defers_again_at_next_claim_boundary():
    torch = __import__("pytest").importorskip("torch")
    from merit_feddg.decoding import MedDeferLogitsProcessor

    class Tokenizer:
        def __init__(self):
            self.mapping = {
                "normal": [3],
                " normal": [5],
                "Normal": [6],
                " Normal": [7],
                "pneumothorax": [4],
                " pneumothorax": [8],
                "Pneumothorax": [10],
                " Pneumothorax": [11],
            }

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
    initial_scores = processor(initial, torch.zeros((1, 12)))
    processor(torch.tensor([[10, 11, 5]]), torch.zeros((1, 12)))
    processor(torch.tensor([[10, 11, 5, 9]]), torch.zeros((1, 12)))
    assert calls == ["claim-0", "claim-1"]
    assert len(processor.traces) == 2
    assert initial_scores[0, 10] > 0  # sentence-initial ``Pneumothorax``
    assert initial_scores[0, 6] < 0  # sentence-initial ``Normal``


def test_transformers_processor_records_none_at_confident_claim_start():
    torch = __import__("pytest").importorskip("torch")
    from merit_feddg.decoding import MedDeferLogitsProcessor

    class Tokenizer:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [3]}

        def decode(self, tokens, skip_special_tokens=True):
            return "finding"

    calls = []

    def request_factory(prefix, uncertainty, claim_index):
        request = _request(uncertainty=uncertainty)
        return ClaimRequest(**{**request.__dict__, "claim_id": f"claim-{claim_index}"})

    processor = MedDeferLogitsProcessor(
        Tokenizer(), _engine(), _pool(calls), request_factory, top_k_entropy=8
    )
    confident_scores = torch.tensor([[20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

    processor(torch.tensor([[10, 11]]), confident_scores)

    assert calls == []
    assert processor.claim_started
    assert len(processor.traces) == 1
    assert processor.traces[0].selected_expert is None
    assert processor.traces[0].reason == "generalist-confident"


def test_phrase_token_collisions_are_capped_in_vocabulary_space():
    torch = __import__("pytest").importorskip("torch")
    from merit_feddg.decoding import MedDeferLogitsProcessor

    class Tokenizer:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [3]}

        def decode(self, tokens, skip_special_tokens=True):
            return ""

    request = _request()

    def request_factory(prefix, uncertainty, claim_index):
        return ClaimRequest(
            **{
                **request.__dict__,
                "claim_id": f"claim-{claim_index}",
                "deferral_policy": "qualified_first_claim",
            }
        )

    engine = _engine()
    processor = MedDeferLogitsProcessor(Tokenizer(), engine, _pool([]), request_factory)
    scores = torch.zeros((1, 8))
    changed = processor(torch.tensor([[1, 2]]), scores)
    assert torch.linalg.vector_norm(changed - scores).item() <= engine.max_bias_norm + 1e-6
