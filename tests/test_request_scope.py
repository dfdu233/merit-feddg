from dataclasses import replace

import pytest
from test_capability_runtime import setup as runtime_setup
from test_capability_runtime import spec

from merit_feddg.capabilities import EvidenceItem
from merit_feddg.capability_runtime import NativeState, ValueGenerationConfig
from merit_feddg.evidence_need import evidence_memory
from merit_feddg.llava_run import experiment_config, parser
from merit_feddg.request_scope import assess_request, bind_request, focused_items


def named(**aliases):
    return {"request_contract": {"mode": "named_concepts", "concept_aliases": aliases}}


def test_finite_anatomy_cannot_answer_unrestricted_organ_question():
    card = named(Heart=["heart"], **{"Left Lung": ["left lung"]})
    assert not assess_request("Which organ is shown?", card, "segmentation")["allowed"]
    assert not assess_request("Where are the breasts?", card, "segmentation")["allowed"]
    assert assess_request("Where is the left lung?", card, "segmentation")["matched_concepts"] == ["Left Lung"]


def test_phrase_boundaries_and_aliases_do_not_assert_positive_findings():
    card = named(Cardiomegaly=["enlarged heart"])
    audit = assess_request("Is there no enlarged heart?", card, "classification")
    assert audit["matched_concepts"] == ["Cardiomegaly"]
    assert "polarity" not in audit
    assert not assess_request("cardiomegalyx", card, "classification")["allowed"]
    assert not assess_request("What organ?", {}, "classification")["allowed"]
    with pytest.raises(ValueError):
        assess_request("q", {"request_contract": {"mode": "free_query"}}, "classification")


def test_requested_low_score_survives_unrelated_high_scores():
    item = EvidenceItem("a", "xrv", "classification", "findings", {
        "findings": [{"finding": "Infiltration", "score": .99},
                     {"finding": "Cardiomegaly", "score": .01}],
        "score_semantics": "uncalibrated_independent_sigmoid",
        "unlisted_findings": "unknown"})
    audit = assess_request("Is cardiomegaly present?", named(Cardiomegaly=[]), "classification")
    bound = bind_request((item,), audit)
    result = evidence_memory(bound, "Is cardiomegaly present?", ValueGenerationConfig(evidence_style="focused"))
    assert result[0]["payload"]["findings"] == [{"finding": "Cardiomegaly", "score": .01}]
    assert result[0]["payload"]["unlisted_findings"] == "unknown"
    assert len(item.payload["findings"]) == 2
    assert focused_items((item,)) == ()


@pytest.fixture
def setup(tmp_path):
    return runtime_setup.__wrapped__(tmp_path)


def test_scope_rejects_before_model_or_domain_gate(setup):
    build, _, _ = setup
    runtime, _, pool = build({"A": spec("segmentation", **named(Heart=[]))},
                             request_scope_check=True, evidence_style="focused", visual_views=0)
    class Gate:
        def decide(self, *args):
            pytest.fail("out-of-scope tool must not reach domain gate")
    result = runtime.run("agent", applicability=Gate())
    assert not pool.calls and result["expert_calls"] == 0
    assert result["trace"][0]["reason"] == "outside_explicit_request_coverage"


def test_domain_gate_can_still_reject_task_compatible_expert(setup):
    build, _, _ = setup
    runtime, _, pool = build({"A": spec(**named(lung=[]))}, request_scope_check=True,
                             evidence_style="focused", visual_views=0)
    runtime.row["question"] = "What is the lung appearance?"
    class Gate:
        def decide(self, row, descriptor):
            return {"allowed": False, "reason": "domain_unsupported"}
    result = runtime.run("agent", applicability=Gate())
    assert not pool.calls
    assert any(e.get("reason") == "domain_unsupported" for e in result["trace"])


def test_supported_request_reaches_evidence_without_second_image(setup):
    build, _, _ = setup
    runtime, probe, pool = build({"A": spec(**named(lung=[]))}, request_scope_check=True,
                                 evidence_style="focused", visual_views=0)
    runtime.row["question"] = "Is lung tissue visible?"
    state, event = runtime.execute(NativeState(), runtime.descriptors(NativeState())[0])
    assert event["adopted"] and event["request_scope"]["matched_concepts"] == ["lung"]
    runtime.complete(state)
    assert not isinstance(probe.contexts[-1][0], list)
    assert len(pool.calls) == 1


def test_unavailable_named_output_has_no_high_score_fallback():
    item = EvidenceItem("a", "x", "classification", "s", {
        "catalog": [{"concept": "heart", "similarity": .9}]})
    audit = assess_request("lung", named(lung=[]), "classification")
    assert focused_items(bind_request((item,), audit)) == ()
    assert replace(item, payload=item.payload) == item


def test_new_pilot_is_original_image_and_focused_with_legacy_unchanged():
    args = parser().parse_args(["--study", "diagnose", "--config", "configs/request_scoped_pilot.yaml"])
    config = experiment_config(args)
    generation = config["capability_value"]["generation"]
    assert generation["visual_views"] == 0 and generation["request_scope_check"]
    assert config["capability_diagnostics"]["contract_comparison"]
    assert config["experts"]["cxr_findings"]["adapter"] == "xrv_classification"
    assert ValueGenerationConfig().request_scope_check is False
