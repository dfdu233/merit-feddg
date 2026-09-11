"""Regression and mechanism checks for the negative-run failure modes."""

import json
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest

from merit_feddg.capabilities import CapabilityRequest, CapabilityResult, EvidenceItem
from merit_feddg.capability_runtime import ValueGenerationConfig
from merit_feddg.evidence_need import evidence_memory
from merit_feddg.evidence_permissions import permission_record, supported_relations
from merit_feddg.evidence_transport import pack_records
from merit_feddg.matched_evaluation import (
    SharedExpertPool,
    cache_identity,
    load_cached,
    load_manifest,
)
from merit_feddg.native_precision import fit_native_precision, precision_for
from merit_feddg.structured_evidence import _typed_record
from merit_feddg.uncertain_evidence import attach_alternatives


def test_cache_covers_nested_adapter_code_and_expert_identity(tmp_path, monkeypatch):
    import merit_feddg.matched_evaluation as module

    monkeypatch.setattr(module, "__file__", str(tmp_path / "matched_evaluation.py"))
    adapters = tmp_path / "experts"
    adapters.mkdir()
    source = adapters / "adapter.py"
    source.write_text("version = 1", encoding="utf-8")
    a = cache_identity({"expert_provenance": {"expert": "v1"}}, [], {})
    source.write_text("version = 2", encoding="utf-8")
    b = cache_identity({"expert_provenance": {"expert": "v1"}}, [], {})
    c = cache_identity({"expert_provenance": {"expert": "v2"}}, [], {})
    assert len({a, b, c}) == 3


def scores(a=0.9, b=0.1):
    return {"findings": [{"finding": "a", "score": a}, {"finding": "b", "score": b}],
            "score_semantics": "uncalibrated_independent_sigmoid"}


def item(payload=None):
    return EvidenceItem("e", "expert", "classification", "scope", payload or scores())


@pytest.mark.parametrize("style", ["native", "scoped", "graph", "uncertainty", "permissions"])
def test_retrieval_answer_policy_applies_before_every_text_compiler(style):
    payload = {"references": [{"source_id": "source", "source_question": "q",
                                "source_reference": "ANSWER_SENTINEL"}]}
    evidence = EvidenceItem("r", "retriever", "retrieval", "source", payload)
    evidence = attach_alternatives(evidence, [payload])
    cfg = ValueGenerationConfig(evidence_style=style, visual_views=0, max_evidence_chars=10000)
    assert "ANSWER_SENTINEL" not in json.dumps(evidence_memory([evidence], "q", cfg))
    assert "ANSWER_SENTINEL" in json.dumps(evidence_memory(
        [evidence], "q", replace(cfg, retrieval_answer_context=True)))
    assert evidence.payload["references"][0]["source_reference"] == "ANSWER_SENTINEL"


def measure(prompt, reserve):
    # Fake full-prompt tokenizer: evidence consumes 7 tokens per character. This
    # intentionally differs from a character budget or additive field estimate.
    count = len(prompt) * 7
    return {"fits": count + reserve <= 1400, "input_tokens": count,
            "context_limit": 1400, "reserved_tokens": reserve,
            "remaining_tokens": 1400 - count - reserve}


def test_full_prompt_token_budget_and_stable_acquisition_order():
    records = [{"expert_id": x, "evidence_id": x, "text": "z" * 55} for x in "ab"]
    render = lambda r: "Original question " + json.dumps(r)
    chosen, audit = pack_records(records, render, measure, max_chars=10000, reserve_tokens=64)
    assert chosen == records[:1]
    assert audit["omitted"] == [{"expert_id": "b", "evidence_id": "b", "reason": "token_budget"}]
    assert audit["context"]["fits"] and audit["prompt_sha256"]
    assert records[0]["text"] == "z" * 55


def test_packet_that_does_not_fit_is_not_truncated_or_silently_counted():
    record = {"expert_id": "a", "evidence_id": "b", "text": "z" * 500}
    chosen, audit = pack_records([record], json.dumps, measure, max_chars=20, reserve_tokens=64)
    assert chosen == [] and audit["presented"] == []
    assert audit["omitted"][0]["reason"] == "character_budget"
    with pytest.raises(ValueError, match="original prompt"):
        pack_records([], lambda r: "q" * 300, measure, max_chars=100, reserve_tokens=64)


def test_larger_uncertainty_removes_permission_without_creating_diagnosis():
    evidence = attach_alternatives(item(), [scores(0.85, 0.15)])
    first = permission_record(evidence, "q")["payload"]
    assert first["supported_measurement_relations"] == [{
        "predicate": "native_score_greater", "subject": "a", "object": "b", "clinical_ordering": False}]
    expanded = attach_alternatives(item(), [scores(0.85, 0.15), scores(0.05, 0.95)])
    second = permission_record(expanded, "q")["payload"]
    assert second["supported_measurement_relations"] == []
    assert not second["truth_coverage_guaranteed"]
    assert second["observation_value_ranges"]["a"] == [0.05, 0.9]


def test_duplicate_object_names_do_not_establish_correspondence():
    evidence = item()
    nodes = _typed_record(evidence, "q")["payload"]["observations"]
    assert supported_relations([nodes + [nodes[0]], nodes], [scores(), scores()]) == []


def test_spatial_permissions_require_original_image_frame_not_patient_side():
    payload = {"structures": [
        {"anatomical_structure": "A", "bbox_xyxy_normalized_original_image": [0.05, 0, 0.3, 1]},
        {"anatomical_structure": "B", "bbox_xyxy_normalized_original_image": [0.7, 0, 0.95, 1]},
    ]}
    evidence = EvidenceItem("s", "segmenter", "segmentation", "anatomy", payload)
    result = permission_record(attach_alternatives(evidence, [payload]), "q")["payload"]
    assert result["supported_measurement_relations"] == [{
        "predicate": "image_x_before", "subject": "A", "object": "B", "patient_laterality_inferred": False}]
    ambiguous = {"structures": [{"anatomical_structure": e["anatomical_structure"],
                                "bbox_xyxy_normalized": e["bbox_xyxy_normalized_original_image"]}
                               for e in payload["structures"]]}
    assert not permission_record(attach_alternatives(replace(evidence, payload=ambiguous), [ambiguous]), "q")["payload"]["supported_measurement_relations"]


def calibration_rows(n=9):
    return [{"split": "train", "loss": "native_simultaneous_score_error", "domain_kind": "hospital",
             "expert": "expert", "capability": "classification", "scope": "scope", "domain": "site",
             "group_id": str(i), "error": 0.2} for i in range(n)]


def calibrate(rows):
    return fit_native_precision(rows, [0, 0.1, 0.2, 0.5], alpha=0.1,
                                expert="expert", capability="classification", scope="scope")


def test_native_calibration_uses_finite_sample_correction_and_group_units():
    card = calibrate(calibration_rows())
    assert card["radius"] == 0.2
    assert calibrate(calibration_rows(8))["radius"] is None
    duplicated = calibration_rows() + calibration_rows()
    assert calibrate(duplicated)["domains"]["site"]["groups"] == 9
    assert not card["unseen_domain_guarantee"]


@pytest.mark.parametrize("change", [{"split": "target"}, {"domain_kind": "proxy"}, {"loss": "token_f1"}])
def test_no_test_labels_proxy_hospitals_or_answer_f1_calibration(change):
    rows = calibration_rows()
    rows[0].update(change)
    with pytest.raises(ValueError):
        calibrate(rows)


def test_domain_radius_changes_permission_and_unknown_domain_cannot_self_certify():
    card = calibrate(calibration_rows())
    evidence = attach_alternatives(item(scores(0.6, 0.4)), [scores(0.6, 0.4)])
    assert permission_record(evidence, "q")["payload"]["supported_measurement_relations"]
    known = replace(evidence, provenance={"native_precision": precision_for(card, evidence, "site")})
    assert not permission_record(known, "q")["payload"]["supported_measurement_relations"]
    unknown = replace(evidence, provenance={"native_precision": precision_for(card, evidence, "unseen")})
    assert permission_record(unknown, "q") is None


def test_cache_binds_final_generation_settings_and_full_manifest(tmp_path):
    a = cache_identity({"budget": 32}, [{"id": "a"}], {"model": "x"})
    assert a != cache_identity({"budget": 64}, [{"id": "a"}], {"model": "x"})
    assert a != cache_identity({"budget": 32}, [{"id": "a"}, {"id": "b"}], {"model": "x"})
    path = tmp_path / "cached.json"
    path.write_text(json.dumps({"identity": a, "output": {"text": "yes"}}))
    assert load_cached(path, a)["text"] == "yes"
    with pytest.raises(ValueError, match="different"):
        load_cached(path, "other")


def test_full_manifest_kept_intact_and_only_answer_blind_task_type_loaded(tmp_path):
    rows = [{"id": str(i), "image": "x.png", "question": "q", "image_sha256": "image",
             "answer_type": "closed"} for i in range(7)]
    path = tmp_path / "manifest.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    assert load_manifest(path) == [{**r, "task": "open_vqa"} for r in rows]
    altered = [{**r, "answer_type": "open"} for r in rows]
    path.write_text("\n".join(json.dumps(r) for r in altered))
    assert load_manifest(path) == [{**r, "task": "open_vqa"} for r in altered]
    rows[0]["answer"] = "no"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    with pytest.raises(ValueError, match="labels"):
        load_manifest(path)


def test_expert_outputs_are_shared_and_cached_even_after_resume(tmp_path):
    calls = []

    def infer(expert, request):
        calls.append((expert, request))
        return CapabilityResult(expert, request.capability, (item(),))

    request = CapabilityRequest("a", "image", "q", "cxr", "vqa", "test", "image", "classification", scope="scope")
    pool = SimpleNamespace(infer=infer)
    shared = SharedExpertPool(pool, tmp_path, "identity")
    first = shared.infer("expert", request)
    assert shared.last_origin == "live_native_output"
    resumed = SharedExpertPool(pool, tmp_path, "identity")
    assert asdict(resumed.infer("expert", request)) == asdict(first)
    assert resumed.last_origin == "cached_native_output" and len(calls) == 1
