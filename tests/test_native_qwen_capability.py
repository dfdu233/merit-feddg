"""Native text generation adapter uses real generation API, not answer scoring."""

import json

import pytest

import merit_feddg.experts.native_qwen as module
from merit_feddg.capabilities import CapabilityRequest, validate_result
from merit_feddg.experts.native_qwen import QwenVqaCapabilityExpert


def request(**kwargs):
    values = {
        "sample_id": "image-1",
        "image": "/images/query.png",
        "question": "What is visible?",
        "modality": "pathology",
        "task": "open_vqa",
        "domain": "target",
        "group_id": "p-1",
        "capability": "generation",
        "scope": "histology_description",
        "query": "Describe nuclei",
    }
    values.update(kwargs)
    return CapabilityRequest(**values)


@pytest.fixture
def fake_probe(monkeypatch):
    constructors, calls = [], []

    class Probe:
        def __init__(self, *args, **kwargs):
            constructors.append((args, kwargs))

        def generate_with_usage(self, image, prompt, max_new_tokens):
            calls.append((image, prompt, max_new_tokens))
            return {
                "text": "Several dark nuclei are visible.",
                "input_tokens": 143,
                "output_tokens": 9,
            }

        def candidate_log_likelihoods(self, *args, **kwargs):
            raise AssertionError("generation tool must not score candidates")

        def probe(self, *args, **kwargs):
            raise AssertionError("generation tool must not project layer scores")

    monkeypatch.setattr(module, "QwenLayerProbe", Probe)
    return constructors, calls


def test_native_generation_and_actual_usage(fake_probe):
    constructors, calls = fake_probe
    expert = QwenVqaCapabilityExpert(
        "local-specialist",
        "specialist",
        "histology_description",
        dtype="float32",
        device_map="cpu",
        max_new_tokens=35,
    )
    assert not constructors  # configuration never downloads/loads weights
    req = request(generated_prefix="This previous draft is unverified.")
    result = expert.infer(req)
    validate_result(result, "specialist", req)
    assert constructors == [
        (("local-specialist",), {"layers": [-1], "dtype": "float32", "device_map": "cpu"})
    ]
    image, prompt, limit = calls[0]
    assert image == req.image and limit == 35
    context = json.loads(prompt.split("\n", 1)[1])
    assert context["question"] == req.question
    assert context["capability_request"] == req.query
    assert context["unverified_prior_context"] == req.generated_prefix
    assert "NOT verified evidence" in prompt
    assert "label" not in context and "reference_answer" not in context
    item = result.items[0]
    assert item.payload["generated_text"] == "Several dark nuclei are visible."
    assert item.payload["usage"] == {"input_tokens": 143, "output_tokens": 9}
    assert item.payload["observation_status"] == "unverified"
    assert item.confidence is None
    assert item.provenance["target_answers_used"] is False
    expert.infer(req)
    assert len(constructors) == 1 and len(calls) == 2


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"capability": "classification"}, "wrong_capability"),
        ({"scope": "diagnosis"}, "wrong_scope"),
        ({"region": (0.1, 0.1, 0.9, 0.9)}, "unsupported_region"),
    ],
)
def test_unsupported_requests_do_not_load(fake_probe, changes, reason):
    constructors, calls = fake_probe
    expert = QwenVqaCapabilityExpert("local", "specialist", "histology_description")
    result = expert.infer(request(**changes))
    assert result.reason == reason and not result.items
    assert not constructors and not calls


def test_prefix_bounded_and_new_query_has_distinct_evidence_id(fake_probe):
    _, calls = fake_probe
    expert = QwenVqaCapabilityExpert("local", "specialist", "histology_description")
    first = expert.infer(request(generated_prefix="A" * 800))
    second = expert.infer(request(query="Describe the tissue arrangement", generated_prefix="B"))
    context = json.loads(calls[0][1].split("\n", 1)[1])
    assert len(context["unverified_prior_context"]) == 512
    assert first.items[0].evidence_id != second.items[0].evidence_id


def test_empty_query_falls_back_to_question(fake_probe):
    _, calls = fake_probe
    expert = QwenVqaCapabilityExpert("local", "specialist", "histology_description")
    req = request(query="")
    expert.infer(req)
    context = json.loads(calls[0][1].split("\n", 1)[1])
    assert context["capability_request"] == req.question
    assert "unverified_prior_context" not in context
