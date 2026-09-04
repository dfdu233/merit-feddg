import numpy as np

from merit_feddg import guided_generate
from merit_feddg.types import EvidenceRecord


def _source_record(domain):
    return EvidenceRecord(
        sample_id=f"source-{domain}",
        domain=domain,
        modality="pathology",
        candidates=["benign", "malignant"],
        label=0,
        general_null_logits=np.zeros(2),
        general_visual_layers=np.zeros((2, 2)),
        expert_scores={"pathology": np.asarray([2.0, -2.0])},
        broad_specialist_scores=np.zeros(2),
        router_probs={"pathology": 1.0},
    )


def test_closed_set_answer_is_locked_to_guided_argmax_before_free_text(monkeypatch):
    events = []

    class FakeGeneralist:
        def __init__(self, *args, **kwargs):
            pass

        def candidate_log_likelihoods(self, image, prompt, candidates):
            events.append("candidate-likelihoods")
            assert candidates == ["benign", "malignant"]
            # The unaided generalist narrowly prefers malignant.
            return np.asarray([-0.4, 0.0])

        def generate(self, image, prompt, max_new_tokens=64, logits_processor=None):
            events.append("free-explanation")
            assert logits_processor is None
            assert "selected answer is locked as: benign" in prompt.casefold()
            # Deliberately contradictory free text must not replace the locked answer.
            return "Despite the instruction, this free explanation says malignant."

    class FakeExpert:
        def image_null_scores(self, image, prompt, concepts):
            events.append("expert")
            assert concepts == [
                "The image shows benign.",
                "The image shows malignant.",
            ]
            return np.asarray([10.0, -10.0])

    monkeypatch.setattr(guided_generate, "QwenLayerProbe", FakeGeneralist)
    monkeypatch.setattr(guided_generate, "_expert_from_spec", lambda *args: FakeExpert())
    monkeypatch.setattr(guided_generate, "_local_or_remote", lambda model_id, root: model_id)
    monkeypatch.setattr(guided_generate, "_release", lambda *args: events.append("released"))

    model_config = {
        "generalist": {"id": "fake-generalist", "layers": [-1]},
        "router": {"abstain_entropy": 1.0},
        "broad_specialist": {"adapter": "unused-single-modality"},
        "experts": {
            "pathology": {
                "id": "fake-pathology-expert",
                "adapter": "fake",
                "modalities": ["pathology"],
                "capabilities": ["classification"],
                "expected_gain": 1.0,
                "latency_ms": 0.0,
            }
        },
    }
    comparison_config = {
        "source_domains": ["source-a", "source-b"],
        "method": {"reliability_prior": 1.0, "reliability_lcb_z": 0.0},
        "med_defer": {
            "controller": {
                "first_claim_policy": "qualified_first_claim",
                "minimum_utility": 0.0,
                "cost_weight": 0.0,
            },
            "domain_trust": {
                "cvar_alpha": 1.0,
                "ood_temperature": 1.0,
                "minimum_post_call_trust": 0.0,
            },
            "guidance": {"strength": 2.0, "max_bias_norm": 2.0},
        },
    }
    case = {
        "id": "target-without-label",
        "image": "unused.png",
        "prompt": "Which diagnosis is most likely?",
        "candidates": ["benign", "malignant"],
        "modality": "pathology",
        "generate_explanation": True,
        "metadata": {"expert_ood_scores": {"pathology": 0.0}},
    }

    result = guided_generate.guided_generate_case(
        case,
        model_config,
        comparison_config,
        [_source_record("source-a"), _source_record("source-b")],
    )

    assert result["answer"] == "benign"
    assert result["answer"] == result["first_claim_selected_answer"]
    assert (
        result["answer"]
        == result["candidate_answers"][int(np.argmax(result["trace"]["guided_logits"]))]
    )
    assert "malignant" in result["explanation"]
    assert result["generation_mode"] == "closed_set_locked_answer_with_explanation"
    assert result["open_claim_decoding_validated"] is False
    assert result["target_label_used"] is False
    assert events.index("candidate-likelihoods") < events.index("expert")
    assert events.index("expert") < events.index("free-explanation")

    no_ood_case = {
        key: value for key, value in case.items() if key != "generate_explanation"
    }
    no_ood_case["metadata"] = {}
    no_ood = guided_generate.guided_generate_case(
        no_ood_case,
        model_config,
        comparison_config,
        [_source_record("source-a"), _source_record("source-b")],
    )
    assert no_ood["answer"] == "malignant"
    assert no_ood["explanation"] is None
    assert no_ood["generation_mode"] == "closed_set_locked_answer_only"
    assert no_ood["trace"]["reason"] == "missing-domain-signal"
    assert no_ood["expert_calls"] == {"pathology": 0}

    mismatched_sources = [_source_record("source-a"), _source_record("source-b")]
    for record in mismatched_sources:
        record.candidates = ["malignant", "benign"]
    with __import__("pytest").raises(ValueError, match="ordered candidate vocabulary"):
        guided_generate.guided_generate_case(
            case,
            model_config,
            comparison_config,
            mismatched_sources,
        )


def test_guided_generation_rejects_empty_and_duplicate_candidates():
    for candidates in (["normal", " NORMAL "], ["normal", "   "]):
        with __import__("pytest").raises(ValueError):
            guided_generate.guided_generate_case(
                {
                    "id": "invalid",
                    "image": "unused.png",
                    "prompt": "What is shown?",
                    "candidates": candidates,
                },
                {},
                {},
                [],
            )
