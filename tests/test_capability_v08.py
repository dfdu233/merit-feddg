"""v0.8 finite actions, image-only routing and evidence-prefix invariants."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from merit_feddg.block_decode import Block
from merit_feddg.capabilities import CapabilityRequest, tool_descriptors
from merit_feddg.capability_generation import QwenCapabilitySession, _parse_action
from merit_feddg.capability_routing import IMAGE_TYPES, infer_image_type, question_type
from merit_feddg.constrained import finite_choice_constraint
from merit_feddg.generalist_factory import make_capability_session
from merit_feddg.io import load_yaml


class TokenList(list):
    def tolist(self):
        return list(self)


class DigitTokenizer:
    eos_token_id = 99

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return [int(char) + 10 for char in text]


def test_finite_action_trie_handles_shared_prefixes_and_disallows_other_tokens():
    constraint = finite_choice_constraint(DigitTokenizer(), ["0", "1", "10"], prompt_length=2)
    assert constraint(0, TokenList([50, 51])) == [10, 11]
    assert constraint(0, TokenList([50, 51, 11])) == [10, 99]
    assert constraint(0, TokenList([50, 51, 11, 10])) == [99]
    assert constraint(0, TokenList([50, 51, 10])) == [99]
    with pytest.raises(ValueError, match="departed"):
        constraint(0, TokenList([50, 51, 12]))


def test_finite_action_infers_prompt_offset_separately_for_each_batch():
    constraint = finite_choice_constraint(DigitTokenizer(), ["1", "2"])
    assert constraint(0, TokenList([50, 51])) == [11, 12]
    assert constraint(1, TokenList([50, 51, 52])) == [11, 12]
    assert constraint(0, TokenList([50, 51, 12])) == [99]
    assert constraint(1, TokenList([50, 51, 52, 11])) == [99]


def tool(expert="retriever", capability="retrieval", scope="related_cases", roi=False):
    return {"expert": expert, "capability": capability, "scope": scope,
            "description": f"Native {capability} observations", "requires_region": roi}


def state(tools):
    return {"question": "What is the image finding?", "generated_prefix": "Earlier text",
            "observations": [], "completed_requests": [], "available_tools": tools}


class Probe:
    def __init__(self, replies):
        self.replies, self.calls = iter(replies), []

    def generate_with_usage(self, image, prompt, **kwargs):
        self.calls.append((image, prompt, kwargs))
        return {"text": next(self.replies), "input_tokens": 20, "output_tokens": 2}


def test_action_id_prebinds_expert_capability_scope_without_roi_generation():
    tools = [tool("conch", "classification", "tissue"), tool()]
    probe = Probe(["2"])
    session = QwenCapabilitySession(probe, "image.png", "original question")
    session.control_protocol = "action_id"
    action = _parse_action(session.control(state(tools), 8), tools)
    assert (action["expert"], action["capability"], action["scope"]) == (
        "retriever", "retrieval", "related_cases"
    )
    assert action["region"] is None
    assert action["query"] == state(tools)["question"]
    assert len(probe.calls) == 1
    assert probe.calls[0][2]["allowed_texts"] == ["0", "1", "2"]
    assert probe.calls[0][2]["max_new_tokens"] == 8
    assert session.last_raw_action == "2"


def test_action_id_continue_cannot_be_misread_as_tool_request():
    probe = Probe(["0"])
    session = QwenCapabilitySession(probe, "image.png", "question")
    session.control_protocol = "action_id"
    assert json.loads(session.control(state([tool()]), 8)) == {"action": "continue"}
    assert len(probe.calls) == 1


def test_action_id_required_roi_has_separate_usage_and_validated_coordinates():
    tools = [tool("sam", "segmentation", "prompted_region", roi=True)]
    probe = Probe(["1", "[0.1, 0.2, 0.8, 0.9]"])
    session = QwenCapabilitySession(probe, "image.png", "question")
    session.control_protocol = "action_id"
    action = _parse_action(session.control(state(tools), 8), tools)
    assert action["region"] == [.1, .2, .8, .9]
    request = CapabilityRequest("id", "image", "question", "ct", "task", "d", "g",
                                "segmentation", region=tuple(action["region"]))
    assert request.region == (.1, .2, .8, .9)
    assert len(probe.calls) == 2
    assert "allowed_texts" not in probe.calls[1][2]
    assert session.last_controller_usage == {"input_tokens": 40, "output_tokens": 4}


@pytest.mark.parametrize("raw", ["[]", "[0.8, 0.2, 0.1, 0.9]", "[0, 0, 2, 1]", "null"])
def test_action_id_invalid_roi_cannot_construct_executable_request(raw):
    tools = [tool("sam", "segmentation", "prompted_region", roi=True)]
    probe = Probe(["1", raw])
    session = QwenCapabilitySession(probe, "image.png", "question")
    session.control_protocol = "action_id"
    with pytest.raises((ValueError, TypeError)):
        action = _parse_action(session.control(state(tools), 8), tools)
        CapabilityRequest("id", "image", "question", "ct", "task", "d", "g",
                          "segmentation", region=tuple(action["region"]))


def test_legacy_controller_remains_supported_without_numeric_constraint():
    raw = json.dumps({"action": "call", "expert": "retriever", "capability": "retrieval",
                      "scope": "related_cases", "query": "Related visual finding", "region": None})
    probe = Probe([raw])
    session = make_capability_session(probe, "image.png", "question")
    assert session.control_protocol == "legacy_json"
    assert session.control(state([tool()]), 160) == raw
    assert "allowed_texts" not in probe.calls[0][2]
    assert _parse_action(raw, [tool()])["scope"] == "related_cases"


def test_image_routing_never_receives_question_reference_or_domain():
    probe = Probe(["chest X-ray"])
    row = {"image": "case.png", "question": "SECRET QUESTION", "reference": "SECRET ANSWER",
           "domain": "SECRET HOSPITAL", "modality": "SECRET LABEL"}
    result = infer_image_type(probe, row)
    assert result["modality"] == "cxr"
    image, prompt, options = probe.calls[0]
    assert image == "case.png" and "SECRET" not in prompt
    assert options["allowed_texts"] == list(IMAGE_TYPES)
    assert "13" not in options["allowed_texts"]
    assert result["input_tokens"] == 20 and result["output_tokens"] == 2


def test_unknown_image_action_is_not_silently_labeled_pathology():
    assert infer_image_type(Probe(["not a choice"]), {"image": "case.png"})["modality"] == "unknown"


def test_new_evidence_rebuilds_prompt_but_preserves_exact_committed_token_ids():
    forwarded, prompts = [], []

    class AnswerSession:
        def propose(self, prefix, count, length):
            forwarded.append((prefix, count, length))
            return [Block((42,), "new", -.1, False)]

    def build(image, prompt):
        prompts.append(prompt)
        return AnswerSession()

    probe = SimpleNamespace(new_answer_session=build)
    session = QwenCapabilitySession(probe, "image.png", "original prompt")
    committed = (71, 12, 5)  # Deliberately not obtained from decoding/re-tokenizing.
    session.next_block(committed, [], 8)
    session.next_block(committed, [{"expert_id": "x", "payload": {"finding": "unknown"}}], 8)
    assert len(prompts) == 2 and prompts[0] != prompts[1]
    assert all(prefix is committed for prefix, _, _ in forwarded)
    assert all(count == 1 for _, count, _ in forwarded)
    assert committed == (71, 12, 5)


def test_default_v08_native_specs_match_real_adapter_contracts():
    config = load_yaml(Path(__file__).resolve().parents[1] / "configs/llava_med_capabilities.yaml")
    specs = config["experts"]
    assert config["generalist"]["backend"] == "llava_med"
    assert config["capability_generation"]["control_protocol"] == "action_id"
    assert specs["cxr_findings"]["adapter"] == "xrv_classification"
    assert specs["cxr_anatomy"]["adapter"] == "xrv_anatomy"
    assert specs["cxr_anatomy"]["requires_region"] is False
    generator = specs["chexagent_description"]
    assert generator["factory"] == "merit_feddg.experts.native_chexagent:CheXagentCapabilityExpert"
    assert generator["factory_kwargs"]["expert_id"] == "chexagent_description"
    assert generator["factory_kwargs"]["scope"] == generator["scope"]
    assert generator["optional"] is True
    descriptors = tool_descriptors(specs, {"modality": "cxr", "task": "open_vqa",
                                           "question": "What organ is visible?"})
    assert "cxr_anatomy" in {item["expert"] for item in descriptors}
    assert "conch_tissue" not in {item["expert"] for item in descriptors}
    assert question_type("What organ is visible?") == "anatomy"
