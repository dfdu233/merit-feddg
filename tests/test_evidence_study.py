"""Offline orchestration tests with explicit artificial model doubles."""

import json
from dataclasses import asdict

import pytest
import yaml
from test_evidence_decode import TinyProbe

from merit_feddg.capabilities import CapabilityResult, EvidenceItem
from merit_feddg.capability_runtime import CapabilityRuntime, NativeSession, ValueGenerationConfig
from merit_feddg.capability_value_study import InterventionScorer
from merit_feddg.evidence_decode import GuidanceConfig
from merit_feddg.evidence_study import compare_source_case, run_evidence_study
from merit_feddg.llava_run import experiment_config, parser
from merit_feddg.open_data import INFERENCE_FIELDS


class TinyPool:
    def __init__(self, *args, **kwargs):
        self.calls = 0

    def reset_case(self):
        pass

    def clear(self):
        pass

    def infer(self, name, request):
        self.calls += 1
        return CapabilityResult(name, "classification", (
            EvidenceItem("item", name, "classification", "scope",
                         {"catalog": [{"concept": "lung", "similarity": 0.7}]}),))


def make_row(role, index):
    row = {key: f"{role}-{index}-{key}" for key in INFERENCE_FIELDS}
    row.update(role=role, question="What organ?", modality="cxr", task="open_vqa",
               domain=f"{role}-domain", domain_kind="proxy")
    return row


SPECS = {"tool": {"id": "test/tool", "capabilities": ["classification"], "scope": "scope",
                  "modalities": ["cxr"], "tasks": ["open_vqa"], "description": "test only"}}


def test_source_native_tool_executed_once_across_strengths():
    probe, pool = TinyProbe(), TinyPool()
    cfg = ValueGenerationConfig(visual_views=0, evidence_style="scoped")
    row = make_row("source", 1)
    engine = CapabilityRuntime(NativeSession(probe, row["image"], "q", row["question"], cfg),
                               pool, row, SPECS, cfg)
    result = compare_source_case(engine, ["evidence"], InterventionScorer({}), GuidanceConfig(), [0.25, 0.5])
    assert pool.calls == 1 and len(result["records"]) == 2
    assert all("guidance_trace" in b["guided"] for b in result["branches"])
    assert "replay" in result["latency_note"]
    engine.row["role"] = "target"
    with pytest.raises(ValueError):
        compare_source_case(engine, [], InterventionScorer({}), GuidanceConfig(), [0.5])


def test_opt_in_configuration_preserves_old_default():
    old = experiment_config(parser().parse_args([]))
    new = experiment_config(parser().parse_args(["--study", "evidence"]))
    assert "bounded_evidence" not in old
    cfg = new["bounded_evidence"]
    assert cfg["generation"]["visual_views"] == 0
    assert cfg["generation"]["request_style"] == "question"
    assert cfg["generation"]["evidence_style"] == "scoped"
    assert "capability_value" not in new
    with pytest.raises(ValueError):
        experiment_config(parser().parse_args(["--study", "evidence", "--generalist", "openmed"]))


def test_real_runner_staging_cache_and_target_reference_isolation(tmp_path, monkeypatch):
    import merit_feddg.capability_experts as experts
    import merit_feddg.evidence_study as module
    import merit_feddg.generalist_factory as factory

    source = [make_row("source", i) for i in range(2)]
    target = [make_row("target", 0)]
    monkeypatch.setattr(module, "read_manifest", lambda path, role: source if role == "source" else target)
    monkeypatch.setattr(module, "audit_open_split", lambda a, b: None)
    monkeypatch.setattr(module, "_filter_optional_experts", lambda specs, artifacts: (specs, {}))
    monkeypatch.setattr(module, "_route_records", lambda rows, *args: (rows, {}))
    monkeypatch.setattr(module, "model_provenance", lambda *args: {"test": True})
    monkeypatch.setattr(factory, "generalist_provenance", lambda *args: {"test": True})
    probe = TinyProbe()
    monkeypatch.setattr(factory, "load_generalist", lambda *args: probe)
    monkeypatch.setattr(experts, "CapabilityPool", TinyPool)
    config = {"generalist": {"backend": "llava_med"}, "experts": SPECS,
              "bounded_evidence": {"generation": asdict(ValueGenerationConfig(visual_views=0)),
                                   "strengths": [0.5]}}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    ref_path = tmp_path / "refs.json"
    refs = {r["id"]: ["evidence"] for r in source}
    # Deliberately omit target answers: source stage must not access them.
    ref_path.write_text(json.dumps(refs), encoding="utf-8")
    args = ("source", "target", ref_path, path, tmp_path, tmp_path / "runs")
    report = run_evidence_study(*args)
    assert report["target_generations"] == 0
    assert all(not c["qualified"] for c in report["cards"].values())
    calls = len(probe.calls)
    assert run_evidence_study(*args)["records"] == report["records"]
    assert len(probe.calls) == calls
    refs[target[0]["id"]] = ["base"]
    ref_path.write_text(json.dumps(refs), encoding="utf-8")
    evaluated = run_evidence_study(*args, stage="evaluate")
    assert "source_calibrated" in evaluated["results"]
    assert evaluated["results"]["source_calibrated"]["mean_expert_calls"] == 0
    assert evaluated["results"]["bounded:tool"]["mean_expert_calls"] == 1
    assert report["bridge_summary"]
    policy_path = tmp_path / "runs" / evaluated["source_key"][:16] / "evidence-policy.json"
    saved = json.loads(policy_path.read_text(encoding="utf-8"))
    saved["cards"][next(iter(saved["cards"]))]["qualified"] = True
    policy_path.write_text(json.dumps(saved), encoding="utf-8")
    with pytest.raises(ValueError, match="saved source policy"):
        run_evidence_study(*args, stage="evaluate")


def test_evaluation_cannot_start_without_source_policy(tmp_path, monkeypatch):
    # Stage validation is performed before loading any backend/model.
    with pytest.raises(ValueError, match="stage"):
        run_evidence_study("s", "t", "r", "c", tmp_path, tmp_path, stage="all")
