"""Capability-loop contracts, with synthetic native outputs, not clinical efficacy tests."""

from __future__ import annotations

import copy
import json
import math
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import pytest

from merit_feddg.block_decode import Block
from merit_feddg.capabilities import (
    CapabilityRequest,
    CapabilityResult,
    EvidenceItem,
    scoped_key,
    validate_result,
)
from merit_feddg.capability_generation import (
    CapabilityConfig,
    QwenCapabilitySession,
    generate_capabilities,
    render_memory,
)


def inference_row():
    return {
        "id": "target-case",
        "image": "unread-by-test.png",
        "image_sha256": "a" * 64,
        "question": "Describe the location and appearance.",
        "modality": "pathology",
        "capability": "generation",
        "task": "open_vqa",
        "domain": "unseen-hospital",
        "domain_kind": "real",
        "role": "target",
        "group_id": "independent-patient",
    }


def expert_spec(capabilities, scope="native", **kwargs):
    return {
        "modalities": ["pathology"],
        "tasks": ["open_vqa"],
        "capabilities": capabilities,
        "scope": scope,
        **kwargs,
    }


def call(expert, capability, scope="native", **kwargs):
    return json.dumps(
        {"action": "call", "expert": expert, "capability": capability, "scope": scope, **kwargs}
    )


class NativePool:
    def __init__(self, payload=None):
        self.calls = []
        self.resets = 0
        self.payload = payload or (lambda request: {"native_text": request.capability})

    def reset_case(self):
        self.resets += 1

    def infer(self, expert, request):
        self.calls.append((expert, request))
        return CapabilityResult(
            expert,
            request.capability,
            (
                EvidenceItem(
                    f"observation-{len(self.calls)}",
                    expert,
                    request.capability,
                    request.scope,
                    self.payload(request),
                ),
            ),
        )


class ScriptedSession:
    """Controller/decoder stand-in whose answer observes the actual evidence memory."""

    def __init__(self, actions=(), output=None):
        self.actions = iter(actions)
        self.states = []
        self.blocks = []
        self.output = output

    def control(self, state, max_tokens):
        self.states.append(copy.deepcopy(state))
        return next(self.actions, '{"action":"continue"}')

    def decode(self, tokens):
        return " ".join(str(token) for token in tokens)

    def next_block(self, prefix, memory, length):
        self.blocks.append((prefix, copy.deepcopy(memory), length))
        token = self.output(memory) if self.output else 100 + len(self.blocks)
        return Block((token,), str(token), -0.1, self.output is not None)


def test_dynamic_two_capabilities_surround_committed_generation():
    specs = {"locator": expert_spec(["detection"]), "writer": expert_spec(["generation"])}
    session = ScriptedSession(
        [call("locator", "detection"), '{"action":"continue"}', call("writer", "generation")]
    )
    pool = NativePool()
    out = generate_capabilities(
        session,
        pool,
        inference_row(),
        CapabilityConfig(max_new_tokens=2, block_tokens=1, max_calls=2),
        specs,
    )
    meaningful = [t["event"] for t in out["trace"] if t["event"] != "controller"]
    assert meaningful == ["tool", "generation", "tool", "generation"]
    assert [r.capability for _, r in pool.calls] == ["detection", "generation"]
    assert pool.calls[0][1].generated_prefix == ""
    assert pool.calls[1][1].generated_prefix == "101"
    assert [b[0] for b in session.blocks] == [(), (101,)]
    assert len(session.blocks[0][1]) == 1
    assert len(session.blocks[1][1]) == 2
    assert out["token_ids"] == [101, 102]
    assert out["expert_calls"] == 2
    assert all("answer" not in vars(request) for _, request in pool.calls)
    for state in session.states:
        assert "domain" not in state and "group_id" not in state
        assert "label" not in state and "reference" not in state


@pytest.mark.parametrize("capability", ["segmentation", "detection", "generation"])
def test_native_payload_changes_continuation_without_candidate_scores(capability):
    def payload(value):
        if capability == "segmentation":
            return {"mask": [[value, value], [0, 0]], "area_fraction": value / 2}
        if capability == "detection":
            return {"boxes_xyxy_normalized": [[0, 0, 0.25 + value / 2, 1]], "count": value}
        return {"text": "present" if value else "unknown"}

    def output(memory):
        observation = memory[0]["payload"]
        if capability == "segmentation":
            return 11 if observation["area_fraction"] else 12
        if capability == "detection":
            return 11 if observation["count"] else 12
        return 11 if observation["text"] == "present" else 12

    results = []
    for value in (0, 1):
        pool = NativePool(lambda request, value=value: payload(value))
        session = ScriptedSession(
            [
                call(
                    "native",
                    capability,
                    region=[0, 0, 1, 1] if capability == "segmentation" else None,
                )
            ],
            output=output,
        )
        results.append(
            generate_capabilities(
                session,
                pool,
                inference_row(),
                CapabilityConfig(max_calls=1),
                {"native": expert_spec([capability])},
            )
        )
        assert "candidate_support" not in json.dumps(results[-1]["evidence"])
        assert pool.calls[0][1].generated_prefix == ""
    assert results[0]["text"] != results[1]["text"]
    if capability == "segmentation":
        assert results[1]["evidence"][0]["payload"]["mask"] == [[1, 1], [0, 0]]
        assert render_memory([EvidenceItem("x", "native", capability, "native", payload(1))], 4000)[
            0
        ]["payload"]["mask"]["omitted_from_prompt"]


@pytest.mark.parametrize("field", ["answer", "references", "label", "metadata"])
def test_inference_api_rejects_reference_fields_before_controller_or_model(field):
    pool, session = NativePool(), ScriptedSession()
    with pytest.raises(ValueError, match="field|leak"):
        generate_capabilities(
            session,
            pool,
            {**inference_row(), field: "target truth"},
            CapabilityConfig(),
            {"native": expert_spec(["generation"])},
        )
    assert not session.states and not session.blocks and not pool.calls


def test_inference_api_rejects_nested_labels_in_allowed_field():
    row = {**inference_row(), "question": {"question": "q", "label": "target truth"}}
    with pytest.raises(ValueError, match="field|metadata|string|inference"):
        generate_capabilities(
            ScriptedSession(),
            NativePool(),
            row,
            CapabilityConfig(max_new_tokens=1),
            {},
        )


@pytest.mark.parametrize(
    "action",
    [
        call("unregistered", "generation"),
        call("native", "detection"),
        call("native", "generation", scope="unregistered-scope"),
        call("native", "generation", command="execute arbitrary shell"),
        call("native", "generation", region=[0, 0, float("nan"), 1]),
        call("native", "generation", region=[0, 0, 2, 1]),
        call("native", "generation", region="bad-region"),
        '[{"action":"continue"}]',
        '{"action":"continue","expert":"native"}',
        "not JSON",
    ],
)
def test_invalid_actions_never_load_tool_and_have_bounded_fallback(action):
    pool, session = NativePool(), ScriptedSession([action])
    out = generate_capabilities(
        session,
        pool,
        inference_row(),
        CapabilityConfig(max_new_tokens=2, block_tokens=1, max_controller_calls=1),
        {"native": expert_spec(["generation"])},
    )
    assert out["expert_calls"] == 0 and not pool.calls
    assert out["controller_calls"] == 1
    assert len(out["token_ids"]) == 2
    assert any(t.get("reason") == "NONE:invalid_action" for t in out["trace"])


def test_segmenter_without_box_is_not_called():
    pool = NativePool()
    out = generate_capabilities(
        ScriptedSession([call("segmenter", "segmentation")]),
        pool,
        inference_row(),
        CapabilityConfig(max_new_tokens=1),
        {"segmenter": expert_spec(["segmentation"])},
    )
    assert out["expert_calls"] == 0 and not pool.calls


def test_scope_rejection_does_not_disable_other_capability_of_same_model():
    specs = {"shared": expert_spec(["classification", "retrieval"])}
    row = inference_row()
    cards = {
        scoped_key("shared", row["modality"], row["task"], "classification", "native"): {
            "qualified": True,
            "status": "supported",
        },
        scoped_key("shared", row["modality"], row["task"], "retrieval", "native"): {
            "qualified": False,
            "status": "insufficient_support",
        },
    }
    session, pool = ScriptedSession([call("shared", "classification")]), NativePool()
    out = generate_capabilities(
        session,
        pool,
        row,
        CapabilityConfig(max_new_tokens=1, max_calls=1),
        specs,
        cards=cards,
        mode="adaptive_dg",
    )
    assert [t["capability"] for t in session.states[0]["available_tools"]] == ["classification"]
    assert out["expert_calls"] == 1
    assert pool.calls[0][1].capability == "classification"
    assert any(t.get("reason") == "NONE:insufficient_support" for t in out["trace"])


@pytest.mark.parametrize(
    "mode, max_calls", [("generalist", 3), ("adaptive_dg", 3), ("adaptive_no_dg", 0)]
)
def test_no_eligible_tools_has_no_controller_and_one_shot_budget(mode, max_calls):
    session, pool = ScriptedSession(output=lambda memory: 7), NativePool()
    out = generate_capabilities(
        session,
        pool,
        inference_row(),
        CapabilityConfig(max_new_tokens=17, block_tokens=2, max_calls=max_calls),
        {"native": expert_spec(["generation"])},
        cards={},
        mode=mode,
    )
    assert out["expert_calls"] == out["controller_calls"] == 0
    assert len(session.blocks) == 1 and session.blocks[0][2] == 17
    assert not pool.calls


def test_static_context_never_controls_and_does_not_fabricate_segmentation_box():
    session, pool = ScriptedSession(output=lambda memory: 7), NativePool()
    out = generate_capabilities(
        session,
        pool,
        inference_row(),
        CapabilityConfig(max_calls=2),
        {"writer": expert_spec(["generation"]), "segmenter": expert_spec(["segmentation"])},
        mode="all_evidence",
    )
    assert out["controller_calls"] == 0 and out["expert_calls"] == 1
    assert pool.calls[0][0] == "writer"
    assert any(t.get("reason") == "NONE:static_baseline_requires_region" for t in out["trace"])


def test_duplicate_request_is_cached_and_cannot_exhaust_controller_forever():
    session = ScriptedSession([call("native", "classification")] * 10)
    out = generate_capabilities(
        session,
        NativePool(),
        inference_row(),
        CapabilityConfig(max_new_tokens=2, block_tokens=1, max_calls=3, max_controller_calls=3),
        {"native": expert_spec(["classification"], prefix_invariant=True)},
    )
    assert out["expert_calls"] == 1
    assert out["cache_hits"] == 2
    assert out["controller_calls"] == 3
    assert out["token_ids"] == [101, 102]


def test_custom_nongenerative_adapter_is_prefix_sensitive_by_default():
    session = ScriptedSession(
        [call("native", "retrieval"), '{"action":"continue"}', call("native", "retrieval")]
    )
    pool = NativePool(lambda request: {"observed_prefix": request.generated_prefix or "empty"})
    out = generate_capabilities(
        session,
        pool,
        inference_row(),
        CapabilityConfig(max_new_tokens=2, block_tokens=1, max_calls=2),
        {"native": expert_spec(["retrieval"])},
    )
    assert out["expert_calls"] == 2
    assert [r.generated_prefix for _, r in pool.calls] == ["", "101"]
    assert [i["payload"]["observed_prefix"] for i in out["evidence"]] == ["empty", "101"]


def test_same_native_evidence_id_across_requests_is_not_silently_discarded():
    class ReusedIdsPool(NativePool):
        def infer(self, expert, request):
            result = super().infer(expert, request)
            return replace(result, items=(replace(result.items[0], evidence_id="native-id"),))

    session = ScriptedSession(
        [call("shared", "classification"), call("shared", "detection")],
        output=lambda memory: len(memory),
    )
    out = generate_capabilities(
        session,
        ReusedIdsPool(),
        inference_row(),
        CapabilityConfig(max_calls=2),
        {"shared": expert_spec(["classification", "detection"])},
    )
    assert out["text"] == "2"
    assert len({i["evidence_id"] for i in out["evidence"]}) == 2
    raw_ids = [t["result"]["items"][0]["evidence_id"] for t in out["trace"] if t["event"] == "tool"]
    assert raw_ids == ["native-id", "native-id"]


def test_rephrased_request_does_not_duplicate_identical_native_observation():
    session = ScriptedSession(
        [
            call("same", "classification", query="appearance"),
            call("same", "classification", query="identify appearance"),
        ],
        output=lambda memory: len(memory),
    )
    out = generate_capabilities(
        session,
        NativePool(),
        inference_row(),
        CapabilityConfig(max_calls=2),
        {"same": expert_spec(["classification"])},
    )
    assert out["expert_calls"] == 2  # Both actual attempts remain visible.
    assert out["text"] == "1" and len(out["evidence"]) == 1
    assert session.states[1]["completed_requests"][0]["expert"] == "same"


def test_native_result_identity_serializability_and_finite_values():
    request = CapabilityRequest(
        "sample",
        "image",
        "question",
        "pathology",
        "open_vqa",
        "domain",
        "group",
        "generation",
        scope="native",
    )
    item = EvidenceItem("i", "native", "generation", "native", {"text": "observed"})
    result = CapabilityResult("native", "generation", (item,))
    assert validate_result(result, "native", request) == result
    invalid = [
        {},
        replace(result, expert_id="other"),
        replace(result, capability="detection"),
        replace(result, items=(replace(item, expert_id="other"),)),
        replace(result, items=(replace(item, scope="other"),)),
        replace(result, items=(replace(item, confidence=float("nan")),)),
        replace(result, items=(replace(item, payload={"x": [float("inf")]}),)),
        replace(result, items=(replace(item, payload={"callback": lambda: None}),)),
        replace(result, items=(replace(item, payload={}),)),
        replace(result, items=(item, item)),
    ]
    for bad in invalid:
        with pytest.raises((ValueError, TypeError)):
            validate_result(bad, "native", request)


def test_memory_admission_never_truncates_json_and_keeps_native_artifact():
    huge = EvidenceItem("large", "native", "generation", "native", {"text": "x" * 10000})
    small = EvidenceItem("small", "native", "detection", "native", {"count": 1})
    memory = render_memory([huge, small], 600)
    assert [i["evidence_id"] for i in memory] == ["small"]
    assert json.loads(json.dumps(memory)) == memory
    assert len(huge.payload["text"]) == 10000


def test_qwen_memory_rebuild_preserves_exact_prefix_ids_without_retokenizing(monkeypatch):
    import merit_feddg.block_decode as block_module

    creations, proposals = [], []

    class FakeAnswerSession:
        def __init__(self, probe, image, prompt):
            creations.append(prompt)

        def propose(self, prefix, count, length):
            proposals.append((prefix, count, length))
            return [Block((23,), "piece", -0.2)]

    monkeypatch.setattr(block_module, "QwenBlockSession", FakeAnswerSession)
    tokenizer = SimpleNamespace(
        decode=lambda *_args, **_kwargs: pytest.fail("continuation must not decode prefix"),
        encode=lambda *_args, **_kwargs: pytest.fail("continuation must not retokenize prefix"),
    )
    probe = SimpleNamespace(processor=SimpleNamespace(tokenizer=tokenizer))
    session = QwenCapabilitySession(probe, "image", "question")
    prefix = (17, 19, 29)
    session.next_block(prefix, [], 2)
    session.next_block(prefix + (23,), [], 2)
    observation = [{"evidence_id": "e", "payload": {"text": "new evidence"}}]
    session.next_block(prefix + (23, 23), observation, 1)
    assert len(creations) == 2
    assert proposals == [(prefix, 1, 2), (prefix + (23,), 1, 2), (prefix + (23, 23), 1, 1)]
    assert "new evidence" in creations[1]


def test_generate_with_usage_counts_actual_tensors_including_eos():
    torch = pytest.importorskip("torch")
    from PIL import Image

    from merit_feddg.generalist import QwenLayerProbe

    class Inputs(dict):
        @property
        def input_ids(self):
            return self["input_ids"]

    inputs = Inputs(
        input_ids=torch.tensor([[1, 63, 60, 60, 62, 10]]),
        attention_mask=torch.ones((1, 6), dtype=torch.long),
    )
    generated_calls, decoded = [], []
    processor_object = object()

    def generate(**kwargs):
        generated_calls.append(kwargs)
        return torch.cat([kwargs["input_ids"], torch.tensor([[21, 22, 2]])], dim=1)

    def decode(values, **kwargs):
        decoded.append(([v.tolist() for v in values], kwargs))
        # EOS disappears from text but remains in the real model output token cost.
        return ['  {"action":"continue"}  ']

    probe = QwenLayerProbe.__new__(QwenLayerProbe)
    probe.torch = SimpleNamespace(inference_mode=nullcontext)
    probe.model = SimpleNamespace(generate=generate)
    probe.processor = SimpleNamespace(batch_decode=decode)
    probe._inputs = lambda *_args: inputs
    image = Image.new("RGB", (2, 2))
    result = probe.generate_with_usage(image, "question", 80, processor_object)
    assert result == {"text": '{"action":"continue"}', "input_tokens": 6, "output_tokens": 3}
    assert generated_calls[0]["max_new_tokens"] == 80  # Budget is not mistaken for actual cost.
    assert generated_calls[0]["do_sample"] is False
    assert generated_calls[0]["logits_processor"] == [processor_object]
    assert decoded[0][0] == [[21, 22, 2]]
    assert decoded[0][1]["skip_special_tokens"] is True
    assert probe.generate(image, "question", 80) == result["text"]
    assert len(generated_calls) == 2  # generate remains a one-call string wrapper.


def test_real_controller_usage_is_recorded_even_for_invalid_json():
    calls = []
    returned = iter(
        [
            {"text": "invalid JSON", "input_tokens": 123, "output_tokens": 7},
            {"text": '{"action":"continue"}', "input_tokens": 155, "output_tokens": 11},
        ]
    )

    def generate_with_usage(image, prompt, max_new_tokens):
        calls.append((image, prompt, max_new_tokens))
        return next(returned)

    probe = SimpleNamespace(
        generate_with_usage=generate_with_usage,
        processor=SimpleNamespace(
            tokenizer=SimpleNamespace(decode=lambda tokens, **_: " ".join(map(str, tokens)))
        ),
    )

    class ControllerSession(QwenCapabilitySession):
        def next_block(self, prefix, memory, length):
            return Block((77,), "77", -0.1)

    session = ControllerSession(probe, "same-image", "original-question")
    out = generate_capabilities(
        session,
        NativePool(),
        inference_row(),
        CapabilityConfig(
            max_new_tokens=2, block_tokens=1, max_controller_calls=2, controller_tokens=160
        ),
        {"native": expert_spec(["generation"])},
    )
    assert out["controller_calls"] == 2 and out["controller_output_tokens"] == 18
    events = [t for t in out["trace"] if t["event"] == "controller"]
    assert events[0]["reason"] == "NONE:invalid_action"
    assert [t["usage"] for t in events] == [
        {"input_tokens": 123, "output_tokens": 7},
        {"input_tokens": 155, "output_tokens": 11},
    ]
    assert session.last_controller_usage == events[-1]["usage"]
    assert all(image == "same-image" and budget == 160 for image, _, budget in calls)
    assert '"generated_prefix": "77"' in calls[1][1]
    assert out["token_ids"] == [77, 77]  # Controller text never enters the answer stream.


@pytest.mark.parametrize("max_calls, max_controls", [(1, 8), (3, 1)])
def test_exhausted_intervention_budget_finishes_with_remaining_greedy_budget(
    max_calls, max_controls
):
    session = ScriptedSession([call("native", "generation")], output=lambda memory: 9)
    generate_capabilities(
        session,
        NativePool(),
        inference_row(),
        CapabilityConfig(
            max_new_tokens=20,
            block_tokens=2,
            max_calls=max_calls,
            max_controller_calls=max_controls,
        ),
        {"native": expert_spec(["generation"])},
    )
    assert len(session.blocks) == 1
    assert session.blocks[0][2] == 20


def test_tiny_real_qwen_vl_controller_and_changed_memory_preserve_committed_tokens(monkeypatch):
    """Exercise real multimodal generation/mRoPE, with random weights and no downloads."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    if not transformers.utils.is_torch_available():
        pytest.skip("installed torch is below the Transformers minimum version")
    from PIL import Image
    from transformers.feature_extraction_utils import BatchFeature

    from merit_feddg.generalist import QwenLayerProbe

    torch.manual_seed(11)
    config = transformers.Qwen2_5_VLConfig(
        text_config={
            "vocab_size": 64,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "max_position_embeddings": 128,
            "rope_scaling": {"rope_type": "default", "mrope_section": [1, 1, 2]},
        },
        vision_config={
            "depth": 1,
            "hidden_size": 32,
            "out_hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 4,
            "patch_size": 2,
            "spatial_merge_size": 1,
            "temporal_patch_size": 1,
            "window_size": 4,
            "fullatt_block_indexes": [0],
        },
        image_token_id=60,
        video_token_id=61,
        vision_start_token_id=63,
        vision_end_token_id=62,
        eos_token_id=2,
        pad_token_id=0,
    )
    model = transformers.Qwen2_5_VLForConditionalGeneration(config).eval()
    model.generation_config.suppress_tokens = [0, 2, 60, 61, 62, 63]
    original_generate = model.generate
    actual_generate_inputs = []

    def record_generate(**kwargs):
        actual_generate_inputs.append(kwargs["input_ids"].detach().clone())
        return original_generate(**kwargs)

    monkeypatch.setattr(model, "generate", record_generate)
    prompt_inputs = []

    def inputs_for_prompt(image, prompt, answer=None):
        assert image.mode == "RGB" and answer is None
        # Different source lengths exercise multimodal rotary-position reset when
        # a controller call and then new evidence intervene between answer blocks.
        suffix = [15, 16, 17] if "State:\n" in prompt else []
        if "Auxiliary model observations" in prompt:
            suffix = [18, 19]
        ids = torch.tensor([[1, 63, 60, 60, 60, 60, 62, 10, *suffix]])
        prompt_inputs.append((prompt, ids.clone()))
        return BatchFeature(
            {
                "input_ids": ids,
                "attention_mask": torch.ones_like(ids),
                "pixel_values": torch.zeros((4, 12)),
                "image_grid_thw": torch.tensor([[1, 2, 2]]),
            }
        )

    def decode(tokens, **kwargs):
        return " ".join(map(str, tokens.tolist() if hasattr(tokens, "tolist") else tokens))

    probe = QwenLayerProbe.__new__(QwenLayerProbe)
    probe.torch = torch
    probe.model = model
    probe.processor = SimpleNamespace(
        tokenizer=SimpleNamespace(decode=decode),
        batch_decode=lambda sequences, **kwargs: [decode(sequence) for sequence in sequences],
    )
    probe._inputs = inputs_for_prompt
    session = QwenCapabilitySession(probe, Image.new("RGB", (4, 4)), "Clinical question.")

    first = session.next_block((), [], 3)
    control_text = session.control(
        {
            "question": "Clinical question.",
            "generated_prefix": session.decode(first.tokens),
            "available_tools": [],
        },
        4,
    )
    memory = [{"expert_id": "native", "evidence_id": "observation", "payload": {"count": 2}}]
    second = session.next_block(first.tokens, memory, 2)
    third = session.next_block(first.tokens + second.tokens, memory, 1)

    assert [len(first.tokens), len(second.tokens), len(third.tokens)] == [3, 2, 1]
    assert all(math.isfinite(block.log_probability) for block in (first, second, third))
    assert len(control_text.split()) == 4
    assert session.last_controller_usage == {"input_tokens": 11, "output_tokens": 4}
    assert len(prompt_inputs) == 3  # Reuse the answer session when evidence is unchanged.
    assert [ids.shape[1] for ids in actual_generate_inputs] == [8, 11, 13, 15]
    assert actual_generate_inputs[2][0, -3:].tolist() == list(first.tokens)
    assert actual_generate_inputs[3][0, -5:].tolist() == list(first.tokens + second.tokens)
    assert [ids.shape[1] for _, ids in prompt_inputs] == [8, 11, 10]
    assert "observation" in prompt_inputs[2][0]
