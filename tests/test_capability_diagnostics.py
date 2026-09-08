"""Offline protocol doubles, not evidence of clinical performance."""

import copy
import json
from dataclasses import asdict, replace

import pytest
from test_capability_runtime import setup as runtime_setup
from test_capability_runtime import spec

from merit_feddg.capabilities import CapabilityResult, EvidenceItem
from merit_feddg.capability_diagnostics import (
    _DuplicateSession,
    collect_diagnostic_case,
    diagnostic_summary,
    write_diagnostics,
)
from merit_feddg.capability_runtime import NativeState, ValueGenerationConfig
from merit_feddg.evidence_need import evidence_memory, evidence_need, scoped_items
from merit_feddg.llava_run import experiment_config, parser


@pytest.fixture
def setup(tmp_path):
    return runtime_setup.__wrapped__(tmp_path)


def metric(row, output, references):
    assert row["role"] == "source" and references == ["unused-source-only-reference"]
    return 0.5 if "token-10" in output["text"] else 0.25


def source_runtime(setup, specs=None, **config):
    build, _, _ = setup
    runtime, probe, pool = build(specs, **config)
    runtime.row["role"] = "source"
    return runtime, probe, pool


def collect(runtime, **kwargs):
    return collect_diagnostic_case(runtime, ["unused-source-only-reference"], metric, **kwargs)


def test_operator_branches_reuse_exact_native_evidence(setup):
    runtime, _, pool = source_runtime(setup, {"A": spec("segmentation")})
    result = collect(runtime, continuations=False, evidence_operators=True)
    branches = {b["name"].split(":")[-1]: b for b in result["branches"]}
    assert len(pool.calls) == 1
    assert branches["scoped_crop"]["native_evidence"] == branches["scoped_control_crop"]["native_evidence"]
    assert branches["scoped_crop"]["presented_memory"] == branches["scoped_control_crop"]["presented_memory"]
    assert branches["scoped_crop"]["with"]["visual_evidence"][0]["view_kind"] == "predicted_region_crop"


def test_operator_pilot_inherits_models_and_is_audit_only():
    args = parser().parse_args(["--study", "diagnose", "--config", "configs/evidence_operators_pilot.yaml"])
    config = experiment_config(args)
    assert "cxr_anatomy" in config["experts"]
    assert config["capability_value"]["generation"]["behavior_probe"] == "audit"
    assert config["capability_diagnostics"]["evidence_operators"] is True


def test_contract_comparison_uses_one_tool_and_no_panel(setup):
    runtime, probe, pool = source_runtime(setup, {"A": spec("segmentation", request_contract={
        "mode": "named_concepts", "concept_aliases": {"Heart": []}})},
        request_scope_check=True, evidence_style="focused", visual_views=0)
    result = collect(runtime, continuations=False, contract_comparison=True, spatial_diagnostics=False)
    branch = next(b for b in result["branches"] if b["name"].endswith(":focused_text"))
    assert branch["request_scope"]["allowed"] is False
    assert branch["with"]["token_ids"] == result["baseline"]["token_ids"]
    assert len(pool.calls) == 1
    assert not any(isinstance(images, list) for images, _ in probe.contexts)


def test_scope_compiler_preserves_native_scores_and_raw_items():
    raw = EvidenceItem("a", "conch", "classification", "tissue", {
        "catalog": [{"concept": "lung", "similarity": 0.8},
                    {"concept": "heart", "similarity": -0.2},
                    {"concept": "kidney", "similarity": 0.7}],
        "score_semantics": "relative_similarity", "unlisted_concepts": "unknown",
    })
    before = copy.deepcopy(asdict(raw))
    result = scoped_items([raw], "What is the heart appearance?", top_k=1)[0]
    assert result.payload["catalog"] == [{"concept": "heart", "similarity": -0.2}]
    assert result.payload["unlisted_concepts"] == "unknown"
    assert asdict(raw) == before
    # Overlap is presentation relevance, not a positive diagnosis threshold.
    assert "diagnosis" not in result.payload and result.payload["catalog"][0]["similarity"] < 0


def test_retrieval_answer_is_an_explicit_ablation_not_a_current_patient_fact():
    item = EvidenceItem("r", "retriever", "retrieval", "cases", {
        "references": [{"source_question": "Which organ?", "source_reference": "SECRET_REF",
                        "reference_applies_to": "source_image_only"}],
        "source_answers_included": True,
    })
    config = ValueGenerationConfig(evidence_style="scoped")
    hidden = evidence_memory([item], "Which organ?", config)
    visible = evidence_memory([item], "Which organ?", replace(config, retrieval_answer_context=True))
    assert "SECRET_REF" not in json.dumps(hidden)
    assert "SECRET_REF" in json.dumps(visible)
    assert item.payload["source_answers_included"] is True
    assert hidden[0]["payload"]["references"][0]["reference_applies_to"] == "source_image_only"


def test_generated_observation_is_not_truncated_into_a_false_positive():
    item = EvidenceItem("g", "generative", "generation", "describe", {
        "generated_text": "Pleural effusion is not demonstrated. Findings remain uncertain."
    })
    presented = scoped_items([item], "Is there an effusion?", top_k=1)[0]
    assert presented.payload["generated_text"] == item.payload["generated_text"]


@pytest.mark.parametrize("capability", ["classification", "segmentation", "detection", "retrieval", "generation"])
def test_typed_need_has_no_label_roi_or_generated_answer_dependency(capability):
    descriptor = {"capability": capability, "scope": "native"}
    result = evidence_need("What organ is shown?", descriptor)
    assert result.question_type == "anatomy" and result.region is None
    assert "What organ is shown?" in result.query
    if capability == "retrieval":
        assert result.query == "What organ is shown?"
    assert set(asdict(result)) == {"capability", "scope", "question_type", "query", "region", "source"}


def test_diagnostic_no_target_queries_or_policy_encoder(setup):
    runtime, probe, pool = source_runtime(setup)
    runtime.row["role"] = "target"
    with pytest.raises(ValueError, match="source"):
        collect(runtime)
    assert not probe.proposals and not pool.calls


def test_same_native_output_is_reused_across_presentations(setup):
    runtime, _, pool = source_runtime(setup)
    result = collect(runtime, continuations=False)
    assert len(pool.calls) == 1  # one classification call, not two presentation calls
    native = next(b for b in result["branches"] if b["name"].endswith(":native_text"))
    scoped = next(b for b in result["branches"] if b["name"].endswith(":scoped_text"))
    assert native["native_evidence"] == scoped["native_evidence"]
    assert native["without"]["token_ids"] == scoped["without"]["token_ids"]
    assert result["target_generations"] == 0 and result["block_none_exact"]


def test_continuation_diagnostics_use_exact_baseline_prefix(setup):
    runtime, _, _ = source_runtime(setup)
    result = collect(runtime)
    assert {b["state_kind"] for b in result["branches"]} == {"initial", "continuation"}
    for branch in result["branches"]:
        prefix = branch["prefix_tokens"]
        assert branch["without"]["token_ids"][:len(prefix)] == prefix
        assert branch["with"]["token_ids"][:len(prefix)] == prefix


def test_mask_text_overlay_and_duplicate_controls_preserve_original(setup):
    runtime, probe, pool = source_runtime(setup, {"A": spec("segmentation")})
    original = runtime.session.image
    before = original.tobytes()
    result = collect(runtime, continuations=False)
    branches = {branch["name"].rsplit(":", 1)[-1]: branch for branch in result["branches"]}
    assert len(pool.calls) == 1
    assert branches["native_overlay"]["with"]["visual_evidence"][0]["sources"]
    assert branches["scoped_text_duplicate"]["with"]["visual_evidence"][0]["sources"] == []
    assert branches["native_text"]["with"]["visual_evidence"] == []
    assert original.tobytes() == before
    assert all(images[0] is original for images, _ in probe.contexts if isinstance(images, list))


def test_duplicate_control_has_identical_pixels_and_honest_label(setup):
    runtime, _, _ = source_runtime(setup)
    old = runtime.session
    session = _DuplicateSession(old.probe, old.image, old.prompt, old.question,
                                replace(runtime.config, visual_views=0))
    images, prompt = session.context(NativeState())
    assert images[0].tobytes() == images[1].tobytes()
    assert "identical copies" in prompt and "model-predicted anatomical regions" not in prompt


def test_pairs_measure_joint_and_conditional_gain_without_claiming_roi_transfer(setup):
    runtime, _, pool = source_runtime(setup, {"A": spec(), "B": spec("retrieval")})
    result = collect(runtime, pairs=[["A", "B"]], continuations=False)
    pair = result["compositions"][0]
    assert pair["joint_gain"] == pytest.approx(pair["first_gain"] + pair["second_given_first_gain"])
    assert pair["interaction"] == pytest.approx(
        pair["joint_gain"] - pair["first_gain"] - pair["second_alone_gain"])
    assert "NOT ROI handoff" in pair["interpretation"]
    assert len(pool.calls) == 3  # A and B alone, B after A; A reused
    assert all(request.region is None for _, request in pool.calls)


def test_incompatible_predeclared_pair_is_reported_not_forced(setup):
    runtime, _, pool = source_runtime(setup, {"A": spec(), "B": spec(modalities=["cxr"])})
    result = collect(runtime, pairs=[["A", "B"]], continuations=False)
    assert not result["compositions"] and result["skipped_pairs"]
    assert {name for name, _ in pool.calls} == {"A"}


@pytest.mark.parametrize("pairs", [[["A", "A"]], [["A", "missing"]], [["A"]]])
def test_invalid_pair_plan_fails_before_inference(setup, pairs):
    runtime, probe, pool = source_runtime(setup)
    with pytest.raises(ValueError, match="pairs"):
        collect(runtime, pairs=pairs)
    assert not pool.calls and not probe.proposals


def test_tool_runtime_error_stops_source_experiment_not_saved_as_zero_gain(setup):
    runtime, _, pool = source_runtime(setup)
    def broken(*args):
        raise ValueError("native tensor shape mismatch")
    pool.infer = broken
    with pytest.raises(RuntimeError, match="tensor shape mismatch"):
        collect(runtime)


def test_generation_subrequest_comparison_executes_two_real_requests(setup):
    runtime, _, pool = source_runtime(setup, {"G": spec("generation")})
    def infer(name, request):
        pool.calls.append((name, request))
        return CapabilityResult(name, "generation", (EvidenceItem(
            "g", name, "generation", request.scope, {"generated_text": "No effusion is visible."}
        ),))
    pool.infer = infer
    result = collect(runtime, continuations=False)
    assert len(pool.calls) == 2
    assert pool.calls[0][1].query == runtime.row["question"]
    assert "at most two short sentences" in pool.calls[1][1].query
    assert {b["request_style"] for b in result["branches"]} == {"need", "question"}


def test_summary_counts_independent_groups_not_repeated_prefixes(setup, tmp_path):
    runtime, _, _ = source_runtime(setup)
    case = collect(runtime)
    summary = diagnostic_summary([case])
    rows = summary["presentation_by_domain"]
    assert all(row["independent_groups"] == 1 and row["branch_count"] == 1 for row in rows)
    assert {row["state_kind"] for row in rows} == {"initial", "continuation"}
    assert not summary["policy_fitted"]
    write_diagnostics(tmp_path, [case], [runtime.row], {})
    annotation = json.loads((tmp_path / "evidence-audit.json").read_text())
    assert annotation and all(row["answer_factuality"] is None for row in annotation)
    assert "unused-source-only-reference" not in json.dumps(annotation)
    routing = json.loads((tmp_path / "routing-audit.json").read_text())
    assert routing[0]["audited_image_type"] is None and "question" not in routing[0]


def test_legacy_profile_stays_default_scoped_is_explicit_and_no_extra_models():
    legacy = experiment_config(parser().parse_args(["--study", "value"]))
    scoped = experiment_config(parser().parse_args(["--study", "value", "--evidence-profile", "scoped"]))
    assert legacy["experts"] == scoped["experts"]
    assert "evidence_style" not in legacy["capability_value"]["generation"]
    assert scoped["capability_value"]["generation"]["request_style"] == "need"
    assert scoped["capability_value"]["generation"]["visual_views"] == 0
    diagnostics = experiment_config(parser().parse_args(["--study", "diagnose", "--chexagent", "off"]))
    assert diagnostics["capability_diagnostics"]["pairs"] == [["conch_tissue", "source_cases"]]


@pytest.mark.parametrize("options", [{"evidence_top_k": 0}, {"evidence_top_k": True},
                                   {"evidence_style": "made_up"}, {"request_style": "label"},
                                   {"retrieval_answer_context": "false"}])
def test_bad_presentation_configuration_is_rejected(options):
    with pytest.raises(ValueError):
        ValueGenerationConfig(**options)
