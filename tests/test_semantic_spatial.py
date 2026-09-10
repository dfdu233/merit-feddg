import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from test_capability_runtime import Probe, spec
from test_capability_runtime import setup as setup  # noqa: F401
from test_vector_gate import Candidate, Verifier
from merit_feddg.capabilities import EvidenceItem
from merit_feddg.capability_runtime import NativeSession, NativeState, ValueGenerationConfig
from merit_feddg.io import load_experiment_yaml
from merit_feddg.matched_evaluation import (
    _finalize_shards,
    experiment_arms,
    generation_prompt,
    load_manifest,
)
from merit_feddg.semantic_evidence import semantic_records, semantic_prompt
from merit_feddg.spatial_evidence import encode_soft_mask
from merit_feddg.vector_gate import VectorGateConfig, assess_visual_contrast


def measure(self, image, prompt, reserve):
    n = len(prompt)
    return {"fits": n + reserve < 20000, "remaining_tokens": 20000-n-reserve,
            "input_tokens": n, "reserved_tokens": reserve}


def test_scalar_semantics_are_preserved_and_dense_arrays_explicitly_external():
    findings = [{"finding": f"label-{i}", "score": i / 100} for i in range(18)]
    mask = encode_soft_mask(np.ones((2, 2), dtype=np.float32))
    item = EvidenceItem("x", "small", "classification", "cxr", {
        "findings": findings, "soft_mask": mask, "score_semantics": "uncalibrated_independent_sigmoid",
        "outside_crop": "unknown", "diagnosis": "not_established"})
    record = semantic_records([item])[0]
    assert record["payload"]["findings"] == findings
    assert record["payload"]["score_semantics"] == item.payload["score_semantics"]
    assert "data" not in record["payload"]["soft_mask"]
    assert not record["payload"]["soft_mask"]["dense_values_in_semantic_tokens"]
    assert item.payload["soft_mask"] == mask
    assert record["interface"]["dense_array_paths"] == ["/payload/soft_mask"]
    geometry = record["payload"]["soft_mask"]["geometry_summary"]
    assert geometry["weighted_centroid_xy"] == [.5, .5]
    assert geometry["mean_mask_value"] == 1 and not geometry["object_presence_established"]
    assert semantic_prompt("Question", []) == "Question"
    with pytest.raises(ValueError):
        semantic_records([replace(item, payload={"score": float("nan")})])


def test_native_grid_geometry_preserves_localization_without_claiming_image_coordinates():
    from merit_feddg.capability_experts import encode_binary_mask
    from merit_feddg.semantic_evidence import native_grid_summary

    mask = np.zeros((2, 4), dtype=np.uint8)
    mask[0, 3] = 1
    for encoded in (encode_binary_mask(mask), encode_soft_mask(mask)):
        summary = native_grid_summary(encoded)
        assert summary["weighted_centroid_xy"] == [.875, .25]
        assert summary["mean_mask_value"] == .125
        assert summary["coordinate_system"] == "native_grid_normalized_xy"
    assert native_grid_summary({"data_omitted_from_git": True})["status"] == "unavailable"


def test_positive_contrast_from_worse_original_image_is_not_a_correction():
    sessions = (Candidate((1,)), Candidate((2,)), Verifier((.1, .6, .3)), Verifier((.1, .8, .1)))
    old = assess_visual_contrast(*sessions, (), config=VectorGateConfig(2), remaining_tokens=2)
    new = assess_visual_contrast(*sessions, (), config=VectorGateConfig(2, require_image_gain=True), remaining_tokens=2)
    assert old["accepted"] and old["image_gain"] < 0
    assert not new["accepted"] and new["reason"] == "no_positive_original_image_gain"


def test_semantic_redundancy_skips_expensive_likelihood_scoring():
    sessions = (Candidate((1,)), Candidate((2,)), Verifier((.1, .2, .7)), Verifier((.1, .7, .2)))
    result = assess_visual_contrast(*sessions, (), config=VectorGateConfig(2), remaining_tokens=2,
                                   semantic_check=lambda a, b: {"passed": False, "equivalent": True})
    assert not result["accepted"] and result["verifier_queries"] == 0
    assert not sessions[2].calls


def test_five_arms_use_one_matched_protocol_and_no_new_trainable_bridge():
    config = load_experiment_yaml("configs/matched_semantic_spatial.yaml")
    decoder = ValueGenerationConfig(**config["capability_value"]["generation"])
    arms = experiment_arms(decoder, "semantic_spatial")
    assert set(arms) == {"generalist", "semantic_all", "hybrid_all", "hybrid_contrast", "hybrid_gate"}
    assert all(a.max_new_tokens == a.block_tokens == a.vector_gate_probe_tokens == 64 for a in arms.values())
    assert all(a.evidence_style == "semantic" and not a.request_scope_check for a in arms.values())
    assert replace(arms["hybrid_all"], semantic_spatial=False) == arms["semantic_all"]
    assert config["generalist"]["training_free_spatial"]


def test_anchor_prompt_contract_uses_only_answer_blind_task_type(tmp_path):
    config = {"prompt_contract": "anchor-ce-v1"}
    assert generation_prompt({"question": "Finding?", "answer_type": "closed"}, config) == (
        "Finding? Please answer Yes or No.")
    assert generation_prompt({"question": "Finding?", "answer_type": "open"}, config) == (
        "Finding?\nGive only the short answer. Do not explain.")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text('{"id":"x","image":"i","question":"q","image_sha256":"h",'
                        '"answer_type":"closed"}\n')
    assert load_manifest(manifest)[0]["answer_type"] == "closed"


def test_complete_disjoint_shards_are_merged_in_manifest_order(tmp_path):
    rows = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    for index, ids in enumerate((("a", "c"), ("b",))):
        root = tmp_path / "shards" / f"{index:04d}-of-0002"
        root.mkdir(parents=True)
        (root / "generalist.json").write_text(json.dumps({key: index for key in ids}))
        (root / "routing.json").write_text(json.dumps({key: index for key in ids}))
    assert _finalize_shards(tmp_path, rows, ["generalist"], {"identity": "x"}, 2)
    assert list(json.loads((tmp_path / "generalist.json").read_text())) == ["a", "b", "c"]
    assert json.loads((tmp_path / "protocol.json").read_text())["shards_complete"]


@pytest.mark.parametrize("relevant", [False, True])
def test_runtime_preserves_raw_rejected_evidence_and_gates_before_generation(setup, monkeypatch, relevant):
    monkeypatch.setattr(Probe, "context_token_budget", measure, raising=False)
    build, _, _ = setup
    runtime, probe, pool = build(specs={"A": spec()}, visual_views=0, evidence_style="semantic",
        token_budgeted_evidence=True, vector_gate="multidimensional", max_evidence_chars=20000)
    calls = []
    runtime.session.assess_relevance = lambda items: {"passed": relevant, "status": "test"}
    def gate(state, proposed):
        calls.append(proposed)
        return {"accepted": True, "reason": "test", "seconds": 0}
    runtime.session.assess_vector_evidence = gate
    state = NativeState(prefix=(9,))
    result, trace = runtime.execute(state, runtime.descriptors(state)[0])
    assert result.prefix == (9,) and bool(result.items) == relevant
    assert bool(calls) == relevant and trace["native_evidence"]
    assert trace["vector_gate"]["dimensions"]["calibration"]["status"] == "unknown"
    assert not probe.proposals


def test_hybrid_session_receives_both_semantics_and_original_native_geometry(monkeypatch):
    monkeypatch.setattr(Probe, "context_token_budget", measure, raising=False)
    probe = Probe()
    probe.tensor_bridge = SimpleNamespace(training_free=True, last_audit={})
    class Packet:
        rejected = []
        def __len__(self):
            return 0
    probe.tensor_packet = lambda items, image: Packet()
    captured = []
    probe.new_tensor_answer_session = lambda image, prompt, items, **kw: captured.append((image, prompt, items, kw))
    config = ValueGenerationConfig(evidence_style="semantic", token_budgeted_evidence=True,
        semantic_spatial=True, visual_views=0, max_evidence_chars=20000)
    session = NativeSession(probe, "original", "Question", "Question", config)
    item = EvidenceItem("a", "small", "classification", "cxr", {"finding": "rare", "score": .03})
    session._evidence_session(NativeState(items=(item,)))
    image, prompt, items, _ = captured[0]
    assert image == "original" and '"score":0.03' in prompt and items == (item,)
    assert session.last_transport == {}  # Candidate setup never mutates live audit.
