"""Native intervention protocol tests using deterministic, offline model doubles."""

import copy
import hashlib
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from merit_feddg.block_decode import Block
from merit_feddg.capabilities import CapabilityResult, EvidenceItem
from merit_feddg.capability_experts import encode_binary_mask
from merit_feddg.capability_features import ValueStateEncoder, action_key, state_kind
from merit_feddg.capability_runtime import (
    CapabilityRuntime,
    NativeSession,
    NativeState,
    ValueGenerationConfig,
)
from merit_feddg.capability_value import fit_value_policy
from merit_feddg.llava_generalist import LlavaMedGeneralist
from merit_feddg.open_data import INFERENCE_FIELDS


class Tokenizer:
    def decode(self, tokens, skip_special_tokens=True):
        return " ".join(f"token-{value}" for value in tokens)


class Probe:
    """Greedy next token depends on available evidence and exact prefix length."""

    def __init__(self, choices=()):
        self.processor = SimpleNamespace(tokenizer=Tokenizer())
        self.choices = iter(choices)
        self.contexts, self.proposals, self.controls = [], [], []

    def new_answer_session(self, images, prompt):
        self.contexts.append((images, prompt))
        probe = self

        class Session:
            def propose(self, prefix, count, length):
                assert count == 1
                probe.proposals.append((tuple(prefix), images, prompt, length))
                offset = 100 if "TOOL OBSERVATIONS" in prompt else 0
                size = min(length, 6 - len(prefix))
                tokens = tuple(offset + n + 1 for n in range(len(prefix), len(prefix) + size))
                return [Block(tokens, "", -0.1, len(prefix) + size == 6)]

        return Session()

    def generate_with_usage(self, images, prompt, max_new_tokens, allowed_texts):
        self.controls.append((images, prompt, tuple(allowed_texts)))
        selection = next(self.choices, "CONTINUE")
        if selection != "CONTINUE":
            selection = next(text for text in allowed_texts if text.startswith(selection + "."))
        return {"text": selection, "input_tokens": 30, "output_tokens": 2}


class Pool:
    def __init__(self, empty=()):
        self.calls, self.empty = [], set(empty)

    def infer(self, expert, request):
        self.calls.append((expert, request))
        provenance = {}
        if expert in self.empty:
            payload = {"generated_text": "  ", "empty_generation": True}
        elif request.capability == "segmentation":
            mask = np.zeros((24, 32), dtype=np.uint8)
            mask[6:18, 8:24] = 1
            payload = {"mask": encode_binary_mask(mask), "semantic_class": "unknown"}
            provenance = {"mask_resolution": "original_image", "target_mask_used": False}
        elif request.capability == "retrieval":
            payload = {"references": [{"source_id": "source-only-case", "source_question": "What tissue?",
                         "reference_applies_to": "source_image_only"}], "query_diagnosis": "not_inferred"}
        else:
            payload = {"catalog": [{"concept": "lung", "similarity": 0.1}],
                       "score_semantics": "relative_similarity", "catalog_exhaustive": False}
        item = EvidenceItem(f"{expert}:native", expert, request.capability, request.scope,
                            payload, provenance=provenance)
        return CapabilityResult(expert, request.capability, (item,))


def spec(capability="classification", **updates):
    return {"capabilities": [capability], "scope": f"native-{capability}",
            "description": "native research observations", "requires_region": False, **updates}


@pytest.fixture
def setup(tmp_path):
    image = Image.new("RGB", (32, 24), "gray")
    path = tmp_path / "query.png"
    image.save(path)
    row = {key: f"value-{key}" for key in INFERENCE_FIELDS}
    row.update(id="query-1", image=str(path), question="What tissue is shown?", modality="pathology",
               capability="generation", task="vqa", role="target", domain="held-out-hospital",
               domain_kind="hospital", group_id="patient-1")

    def build(specs=None, choices=(), empty=(), encoder=None, **config):
        config = ValueGenerationConfig(max_new_tokens=8, block_tokens=2, **config)
        probe, pool = Probe(choices), Pool(empty)
        session = NativeSession(probe, image, "Answer the medical question concisely.", row["question"], config)
        runtime = CapabilityRuntime(session, pool, dict(row), specs or {"A": spec()}, config, encoder)
        return runtime, probe, pool

    return build, row, image


def test_intervention_reuses_same_exact_committed_token_prefix(setup):
    build, _, _ = setup
    runtime, probe, pool = build()
    state = runtime.advance(NativeState(), 2)
    assert state.prefix == (1, 2)
    state, tool = runtime.execute(state, runtime.descriptors(state)[0])
    assert state.prefix == (1, 2) and tool["token_start"] == 2
    result = runtime.complete(state)
    assert result["token_ids"] == [1, 2, 103, 104, 105, 106]
    assert probe.proposals[-1][0] == (1, 2)
    assert pool.calls[0][1].generated_prefix == "token-1 token-2"
    assert len(probe.contexts) == 2  # Same committed IDs; new evidence context.


def test_paired_continuations_start_from_identical_ids_without_state_mutation(setup):
    build, _, _ = setup
    runtime, probe, _ = build()
    state = runtime.advance(NativeState(), 2)
    untouched = state
    baseline = runtime.complete(state)
    called, event = runtime.execute(state, runtime.descriptors(state)[0])
    assisted = runtime.complete(called)
    assert event["adopted"]
    assert state == untouched and state.items == () and state.history == ()
    assert baseline["token_ids"][:2] == assisted["token_ids"][:2] == [1, 2]
    assert probe.proposals[-2][0] == probe.proposals[-1][0] == state.prefix


def test_predicted_native_mask_becomes_additional_view_not_replacement(setup):
    build, _, original = setup
    before = original.tobytes()
    runtime, probe, _ = build({"segment": spec("segmentation")})
    state, event = runtime.execute(NativeState(), runtime.descriptors(NativeState())[0])
    assert event["adopted"]
    runtime.complete(state)
    images, prompt = probe.contexts[-1]
    assert isinstance(images, list) and len(images) == 2
    assert images[0] is original and images[0].tobytes() == before
    assert images[1].tobytes() != before
    assert "PREDICTED tool overlay" in prompt and "not a second patient" in prompt
    assert runtime.session.view_metadata[0]["sources"][0]["expert_id"] == "segment"
    assert state.items[0].payload["mask"]["counts"]  # Raw native geometry survives.


def test_visual_only_mask_is_adopted_when_text_budget_omits_its_record(setup):
    build, _, original = setup
    runtime, probe, _ = build({"segment": spec("segmentation")}, max_evidence_chars=2)
    state, event = runtime.execute(NativeState(), runtime.descriptors(NativeState())[0])
    assert event["adopted"]
    runtime.complete(state)
    images, prompt = probe.contexts[-1]
    assert len(images) == 2 and images[0] is original
    assert "PREDICTED tool overlay" in prompt
    assert runtime.session.view_metadata[0]["sources"][0]["expert_id"] == "segment"


def test_evidence_budget_cannot_be_smaller_than_empty_json():
    with pytest.raises(ValueError):
        ValueGenerationConfig(max_evidence_chars=1)


def test_unknown_tool_repeated_call_wrong_capability_and_unprompted_roi_are_rejected(setup):
    build, _, _ = setup
    runtime, _, pool = build({"A": spec(), "prompted": spec("segmentation", requires_region=True)})
    available = runtime.descriptors(NativeState())
    assert [descriptor["expert"] for descriptor in available] == ["A"]
    for changes in ({"expert": "unregistered"}, {"expert": "prompted", "requires_region": True},
                    {"capability": "generation"}, {"scope": "wrong-scope"}):
        with pytest.raises(ValueError, match="incompatible"):
            runtime.execute(NativeState(), {**available[0], **changes})
    assert pool.calls == []
    state, _ = runtime.execute(NativeState(), available[0])
    with pytest.raises(ValueError, match="repeated"):
        runtime.execute(state, available[0])
    assert len(pool.calls) == 1


def test_empty_generated_text_is_executed_but_not_adopted(setup):
    build, _, _ = setup
    runtime, probe, pool = build({"empty": spec("generation")}, empty=["empty"])
    result = runtime.run("all_evidence")
    events = [event for event in result["trace"] if event["event"] == "tool"]
    assert len(pool.calls) == 1
    assert events[0]["executed"] and not events[0]["adopted"]
    assert events[0]["reason"] == "empty_or_unusable_evidence"
    assert result["adopted_evidence_count"] == 0 and result["evidence"] == []
    assert "TOOL OBSERVATIONS" not in probe.contexts[-1][1]


@pytest.mark.parametrize("mode", ["agent", "all_evidence"])
def test_tool_a_then_tool_b_can_combine_before_first_answer_token(setup, mode):
    build, _, _ = setup
    runtime, _, pool = build({"A": spec(), "B": spec("retrieval")}, choices=["A", "B"])
    result = runtime.run(mode)
    events = [event for event in result["trace"] if event["event"] == "tool"]
    assert [name for name, _ in pool.calls] == ["A", "B"]
    assert [event["token_start"] for event in events] == [0, 0]
    assert events[1]["state_kind"] == "after_tool"
    assert events[1]["history_actions"] == [events[0]["action_key"]]
    assert {item["capability"] for item in result["evidence"]} == {"classification", "retrieval"}
    assert result["adopted_evidence_count"] == 2


def test_block_none_is_token_identical_to_single_greedy_generalist(setup):
    build, _, _ = setup
    base_runtime, base_probe, _ = build()
    block_runtime, block_probe, pool = build()
    base, blocks = base_runtime.run("generalist"), block_runtime.run("block_none")
    assert base["token_ids"] == blocks["token_ids"] == [1, 2, 3, 4, 5, 6]
    assert base["text"] == blocks["text"]
    assert len(base_probe.proposals) == 1 and len(block_probe.proposals) == 3
    assert [entry[0] for entry in block_probe.proposals] == [(), (1, 2), (1, 2, 3, 4)]
    assert pool.calls == [] and blocks["controller_calls"] == 0


def test_agent_can_choose_tool_after_a_committed_block_without_rewriting_it(setup):
    build, _, _ = setup
    runtime, _, pool = build(choices=["CONTINUE", "A"])
    result = runtime.run("agent")
    tool = next(event for event in result["trace"] if event["event"] == "tool")
    assert tool["token_start"] == 2 and tool["state_kind"] == "continuation"
    assert result["token_ids"][:2] == [1, 2]
    assert pool.calls[0][1].generated_prefix == "token-1 token-2"


class MinimalEncoder:
    def candidates(self, row, state, descriptors, prefix_text):
        return [{"action_key": action_key(row, descriptor), "features": [0.1],
                 "state_kind": state_kind(state), "history_actions": list(state.history)}
                for descriptor in descriptors]


def test_value_robust_without_source_support_chooses_none_and_does_not_fit(setup):
    build, _, _ = setup
    policy = fit_value_policy([], {"source_only": True})
    original = copy.deepcopy(policy)
    runtime, _, pool = build(encoder=MinimalEncoder())
    result = runtime.run("value_robust", policy=policy)
    assert pool.calls == [] and result["expert_calls"] == 0
    assert result["token_ids"] == [1, 2, 3, 4, 5, 6]
    decisions = [entry for entry in result["trace"] if entry["event"] == "controller"]
    assert all(entry["action"] == {"action": "continue"} for entry in decisions)
    assert all(not score["supported"] for entry in decisions for score in entry["scores"])
    assert any(entry["reason"] == "NONE:no_supported_positive_utility" for entry in decisions)
    assert policy == original


class FeaturePool:
    def __init__(self):
        self.images, self.texts = [], []

    def _image_feature(self, spec, image):
        self.images.append(image)
        return np.array([1, 2, 3, 4, 5, 6], dtype=float)

    def _text_vectors(self, spec, texts):
        self.texts.append(texts)
        return [np.frombuffer(hashlib.sha256(value.encode()).digest()[:6], dtype=np.uint8).astype(float)
                for _, value in texts]


def test_feature_projection_is_fixed_and_domain_role_ids_do_not_change_features(setup):
    build, row, _ = setup
    runtime, _, _ = build()
    pool = FeaturePool()
    encoder = ValueStateEncoder(pool, {"model": "frozen"}, dimensions=4, seed=71)
    descriptors = runtime.descriptors(NativeState())
    source = {**row, "role": "source", "domain": "source-hospital", "id": "source-id"}
    a = encoder.candidates(source, NativeState(), descriptors, "")
    projections = {size: values.copy() for size, values in encoder.projections.items()}
    b = encoder.candidates(row, NativeState(), descriptors, "")
    assert a == b and len(a[0]["features"]) == 8 * 4 + 5
    assert all(np.array_equal(projections[size], values) for size, values in encoder.projections.items())
    other = ValueStateEncoder(FeaturePool(), {}, dimensions=4, seed=71)
    assert other.candidates(row, NativeState(), descriptors, "") == b
    assert "held-out-hospital" not in str(pool.texts) and "source-hospital" not in str(pool.texts)
    assert not hasattr(encoder, "fit")


@pytest.mark.parametrize("field", ["label", "target_label", "answer", "reference", "references", "gain"])
def test_labels_and_outcomes_cannot_enter_runtime_or_encoder(setup, field):
    build, row, _ = setup
    runtime, _, pool = build()
    leaked = {**row, field: "secret-answer"}
    with pytest.raises(ValueError, match="label-free"):
        CapabilityRuntime(runtime.session, pool, leaked, runtime.specs, runtime.config)
    encoder_pool = FeaturePool()
    encoder = ValueStateEncoder(encoder_pool, {}, dimensions=4)
    with pytest.raises(ValueError, match="label-free"):
        encoder.candidates(leaked, NativeState(), runtime.descriptors(NativeState()), "")
    assert encoder_pool.images == [] and encoder_pool.texts == []


def test_state_features_track_prefix_history_not_future_tool_result(setup):
    build, row, _ = setup
    runtime, _, _ = build({"A": spec(), "B": spec("retrieval")})
    encoder = ValueStateEncoder(FeaturePool(), {}, dimensions=4)
    initial = NativeState()
    first = runtime.descriptors(initial)[0]
    after = replace(initial, history=(action_key(row, first),))
    continued = replace(after, prefix=(1, 2))
    assert state_kind(initial) == "initial"
    assert state_kind(after) == "after_tool"
    assert state_kind(continued) == "continuation"
    candidates = encoder.candidates(row, continued, runtime.descriptors(continued), "token-1 token-2")
    assert candidates[0]["history_actions"] == list(after.history)
    assert set(candidates[0]) == {"action_key", "features", "state_kind", "history_actions"}


def make_multiimage_backend():
    torch = pytest.importorskip("torch")
    backend = object.__new__(LlavaMedGeneralist)
    backend.torch, backend.tokenizer, backend.conv_mode = torch, Tokenizer(), "test"
    backend.image_processor = object()
    captured = {}

    class Conversation:
        roles = ("USER", "ASSISTANT")

        def copy(self):
            result = Conversation()
            result.messages = []
            return result

        def append_message(self, role, value):
            self.messages.append((role, value))

        def get_prompt(self):
            captured["prompt"] = self.messages[0][1]
            return "[INST] " + self.messages[0][1] + " [/INST]"

    def process_images(images, processor, config):
        captured["images"] = images
        return torch.zeros(len(images), 3, 4, 4)

    backend.runtime = SimpleNamespace(
        constants=SimpleNamespace(DEFAULT_IMAGE_TOKEN="<image>", IMAGE_TOKEN_INDEX=-200),
        conv_templates={"test": Conversation()},
        tokenizer_image_token=lambda prompt, *a, **kw: torch.tensor([1] + [-200] * prompt.count("<image>") + [5]),
        process_images=process_images,
    )
    tower = SimpleNamespace(device=torch.device("cpu"), dtype=torch.float32, num_patches=4)
    backend.model = SimpleNamespace(
        config=SimpleNamespace(mm_use_im_start_end=False, max_position_embeddings=100),
        get_input_embeddings=lambda: SimpleNamespace(weight=torch.zeros(1)),
        get_vision_tower=lambda: tower,
    )
    return backend, captured


def test_llava_two_native_images_have_two_tokens_preserved_order_and_sizes():
    backend, captured = make_multiimage_backend()
    original, overlay = Image.new("RGB", (20, 10), "gray"), Image.new("RGB", (20, 10), "yellow")
    single = backend._inputs(original, "<image> What organ?")
    assert captured["prompt"] == "<image>\nWhat organ?"
    assert single["images"].shape[0] == 1
    inputs = backend._inputs([original, overlay], "What organ?")
    assert captured["prompt"] == "<image>\n<image>\nWhat organ?"
    assert inputs["images"].shape[0] == 2
    assert int((inputs["inputs"] == -200).sum()) == 2
    assert inputs["image_sizes"] == [original.size, overlay.size]
    assert captured["images"][0].tobytes() == original.tobytes()
    assert captured["images"][1].tobytes() == overlay.tobytes()
    backend.model.config.max_position_embeddings = 12
    with pytest.raises(ValueError, match="exceed"):
        backend._validate_context(inputs, max_new_tokens=3)
    backend._validate_context(single, max_new_tokens=3)


@pytest.mark.parametrize("views", [[], [1, 2, 3]])
def test_llava_rejects_empty_or_unbounded_image_lists(views):
    backend, _ = make_multiimage_backend()
    with pytest.raises(ValueError, match="one original"):
        backend._inputs(views, "question")
