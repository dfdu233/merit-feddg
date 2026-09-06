"""Synthetic runtime fixtures test wiring, never claim real-model medical gains."""

from __future__ import annotations

import copy
import json
import sys
from dataclasses import replace
from types import ModuleType, SimpleNamespace

import pytest

from merit_feddg.capabilities import EvidenceItem
from merit_feddg.capability_features import action_key, state_kind
from merit_feddg.capability_runtime import ValueGenerationConfig
from merit_feddg.capability_value_study import InterventionScorer, collect_source_case


def make_row(identity="case", role="source", domain="source-a"):
    return {
        "id": identity, "role": role, "image": f"{identity}.png", "question": "What is shown?",
        "group_id": f"image-{identity}", "domain": domain, "domain_kind": "proxy",
        "modality": "pathology", "task": "open_vqa",
    }


class BranchRuntime:
    """Deterministic, prefix-preserving mock with a genuine conditional A->B label."""

    def __init__(self, *, role="source", tools=("A", "B"), answer_length=6, row=None):
        self.row = row or make_row(role=role)
        self.config = ValueGenerationConfig(max_new_tokens=12, block_tokens=2, max_expert_calls=2)
        self.tools, self.answer_length = tools, answer_length
        self.execution_states, self.feature_states, self.complete_states = [], [], []

    def descriptors(self, state):
        if len(state.history) >= self.config.max_expert_calls:
            return []
        result = []
        for name in self.tools:
            descriptor = {"expert": name, "capability": "classification", "scope": "tissue",
                          "description": name, "requires_region": False}
            if action_key(self.row, descriptor) not in state.history:
                result.append(descriptor)
        return result

    def candidates(self, state, descriptors):
        self.feature_states.append(state)
        return [{
            "action_key": action_key(self.row, descriptor), "state_kind": state_kind(state),
            "history_actions": list(state.history),
            "features": [float(len(state.prefix)), float(len(state.items)), float(i)],
        } for i, descriptor in enumerate(descriptors)]

    def execute(self, state, descriptor):
        assert descriptor in self.descriptors(state)
        self.execution_states.append((state, descriptor["expert"]))
        name = descriptor["expert"]
        executed, adopted = name != "FAILED", name not in {"EMPTY", "FAILED"}
        item = EvidenceItem(name, name, "classification", "tissue", {"label": name})
        after = replace(state, items=((item,) + state.items if adopted else state.items),
                        history=(*state.history, action_key(self.row, descriptor)))
        return after, {"expert": name, "executed": executed, "adopted": adopted,
                       "seconds": 0.25, "reason": "mock_observation" if adopted else name}

    def complete(self, state):
        self.complete_states.append(state)
        names = tuple(sorted(item.expert_id for item in state.items))
        quality = {(): 0.2, ("A",): 0.4, ("B",): 0.1, ("A", "B"): 0.9}.get(names, 0.5)
        first = 10 if not names else 20 if names == ("A",) else 30
        remaining = max(0, self.answer_length - len(state.prefix))
        tokens = [*state.prefix, *range(first, first + remaining)]
        return {"text": "+".join(names) or "base", "quality": quality,
                "token_ids": tokens, "seconds": 1.0 + quality, "finished": True}


def quality_scorer(row, output, references):
    assert row["role"] == "source" and references == ["source-only-reference"]
    return output["quality"]


def collect(runtime, **kwargs):
    return collect_source_case(runtime, ["source-only-reference"], quality_scorer, **kwargs)


def test_initial_same_prefix_pairs_and_conditional_second_tool_not_sum_of_single_gains():
    runtime = BranchRuntime()
    result = collect(runtime, pair_first_tools=("A",), collect_continuations=False)
    initial = [branch for branch in result["branches"] if branch["state_kind"] == "initial"]
    after_a = [branch for branch in result["branches"] if branch["state_kind"] == "after_tool"]
    assert len(initial) == 2 and len(after_a) == 1
    assert {branch["with"]["text"] for branch in initial} == {"A", "B"}
    assert all(branch["without"]["text"] == "base" for branch in initial)
    assert after_a[0]["without"]["text"] == "A"
    assert after_a[0]["with"]["text"] == "A+B"
    assert after_a[0]["gain"] == pytest.approx(0.9 - 0.4)
    assert sum(branch["gain"] for branch in initial) == pytest.approx(0.1)
    assert after_a[0]["gain"] != pytest.approx(sum(branch["gain"] for branch in initial))
    assert after_a[0]["history_actions"] == [initial[0]["action_key"]]
    # The first A intervention was cached, not executed again for its own initial pair.
    assert sum(not state.history and expert == "A"
               for state, expert in runtime.execution_states) == 1


def test_continuations_preserve_exact_baseline_token_prefix_in_both_branches():
    runtime = BranchRuntime(answer_length=6)
    result = collect(runtime, pair_first_tools=("A",))
    later = [branch for branch in result["branches"] if branch["state_kind"] == "continuation"]
    assert later and {tuple(branch["prefix_tokens"]) for branch in later} == {(10, 11), (20, 21)}
    for branch in result["branches"]:
        prefix = branch["prefix_tokens"]
        assert branch["without"]["token_ids"][:len(prefix)] == prefix
        assert branch["with"]["token_ids"][:len(prefix)] == prefix
    with_history = [branch for branch in later if branch["history_actions"]]
    assert len(with_history) == 1 and with_history[0]["with"]["text"] == "A+B"


@pytest.mark.parametrize("length", [0, 1, 2])
def test_short_answers_do_not_invent_later_generation_states(length):
    result = collect(BranchRuntime(answer_length=length), pair_first_tools=("A",))
    assert all(branch["state_kind"] != "continuation" for branch in result["branches"])


def test_continuation_collection_can_be_disabled():
    result = collect(BranchRuntime(answer_length=8), collect_continuations=False)
    assert {row["state_kind"] for row in result["records"]} == {"initial"}


def test_pre_action_features_contain_old_evidence_not_the_new_observation():
    runtime = BranchRuntime()
    result = collect(runtime, pair_first_tools=("A",), collect_continuations=False)
    assert all(record["features"][1] == 0 for record in result["records"]
               if record["state_kind"] == "initial")
    after = next(row for row in result["records"] if row["state_kind"] == "after_tool")
    assert after["features"][1] == 1 and after["history_actions"]
    assert all(len(state.items) < 2 for state in runtime.feature_states)


def test_empty_executed_and_runtime_failed_outcomes_are_not_dropped_or_pair_roots():
    runtime = BranchRuntime(tools=("A", "EMPTY", "FAILED"))
    result = collect(runtime, pair_first_tools=("EMPTY", "FAILED"), collect_continuations=False)
    empty = next(row for row in result["records"] if row["action_key"].startswith("EMPTY|"))
    failed = next(row for row in result["records"] if row["action_key"].startswith("FAILED|"))
    assert empty["executed"] and not empty["adopted"] and empty["gain"] == 0
    assert not failed["executed"] and not failed["adopted"]
    assert empty["cost"] == pytest.approx(0.25)
    assert all(row["state_kind"] == "initial" for row in result["records"])
    assert len(result["records"]) == 3


def test_no_target_collection_and_no_references_in_feature_record():
    runtime = BranchRuntime(role="target")
    with pytest.raises(ValueError, match="source"):
        collect(runtime)
    assert not runtime.execution_states and not runtime.complete_states
    result = collect(BranchRuntime(), collect_continuations=False)
    assert all(row["role"] == "source" for row in result["records"])
    assert "source-only-reference" not in json.dumps(result["records"])


def test_declared_pair_roots_obey_tool_call_budget_and_root_limit():
    runtime = BranchRuntime()
    runtime.config = replace(runtime.config, max_expert_calls=1)
    result = collect(runtime, pair_first_tools=("A", "B"), collect_continuations=False)
    assert {row["state_kind"] for row in result["records"]} == {"initial"}
    result = collect(BranchRuntime(), pair_first_tools=("A", "B"),
                     max_pair_first_tools=1, collect_continuations=False)
    assert result["pair_roots"] == ["A"]


def install_custom_scorer(monkeypatch, value):
    module = ModuleType("test_local_quality_hook")
    calls = []

    def factory(offset=0.0):
        def evaluate(**kwargs):
            calls.append(kwargs)
            return value + offset if type(value) in (int, float) else value
        return evaluate

    module.factory = factory
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module.__name__, calls


def test_custom_scorer_receives_only_explicit_scoring_inputs_and_uses_kwargs(monkeypatch):
    module, calls = install_custom_scorer(monkeypatch, 0.35)
    scorer = InterventionScorer({"name": "clinical_quality", "factory": f"{module}:factory",
                                 "fingerprint": "fixed-implementation-sha", "kwargs": {"offset": 0.1}})
    assert scorer(make_row(), {"text": "prediction", "internal": "not passed"}, ["answer"]) == pytest.approx(0.45)
    assert calls == [{"question": "What is shown?", "prediction": "prediction", "references": ["answer"]}]


@pytest.mark.parametrize("value", [True, "0.5", -0.1, 1.1, float("nan"), float("inf")])
def test_custom_scorer_rejects_binary_nonnumeric_nonfinite_or_out_of_range(value, monkeypatch):
    module, _ = install_custom_scorer(monkeypatch, value)
    scorer = InterventionScorer({"name": "custom", "factory": f"{module}:factory", "fingerprint": "v1"})
    with pytest.raises(ValueError, match="quality scorer"):
        scorer(make_row(), {"text": "prediction"}, ["reference"])


@pytest.mark.parametrize("config", [
    {"name": "custom"}, {"name": "custom", "factory": "some:factory"},
    {"name": "custom", "fingerprint": "v1"},
])
def test_custom_scorer_requires_declared_factory_and_fingerprint(config):
    with pytest.raises(ValueError, match="factory and fingerprint"):
        InterventionScorer(config)


def test_default_lexical_scorer_is_not_silently_binary():
    scorer = InterventionScorer({"name": "token_f1"})
    assert scorer(make_row(), {"text": "renal"}, ["renal cell carcinoma"]) == pytest.approx(0.5)


def test_source_stage_freezes_before_target_and_target_reference_changes_only_evaluation(tmp_path, monkeypatch):
    from merit_feddg import capability_experts, generalist_factory
    from merit_feddg import capability_value_study as study

    source = [make_row(f"{domain}-{i}", domain=domain) for domain in ("d1", "d2") for i in range(2)]
    target = [make_row("target", role="target", domain="held-out")]
    refs = {row["id"]: ["A"] for row in source + target}
    reference_path = tmp_path / "references.json"
    reference_path.write_text(json.dumps(refs), encoding="utf-8")
    config = {
        "generalist": {"backend": "mock"}, "experts": {"source_cases": {}},
        "capability_value": {
            "methods": ["generalist", "value_mean"], "single_tools": False,
            "collection": {"collect_continuations": False},
            "fit": {"min_cases_per_domain": 2},
        },
    }
    monkeypatch.setattr(study, "read_manifest", lambda path, role: copy.deepcopy(source if role == "source" else target))
    monkeypatch.setattr(study, "audit_open_split", lambda src, dst: None)
    monkeypatch.setattr(study, "load_yaml", lambda path: copy.deepcopy(config))
    monkeypatch.setattr(study, "_filter_optional_experts", lambda specs, artifacts: (specs, []))
    monkeypatch.setattr(study, "_extraction_runtime_provenance", lambda: {"code": "fixed"})
    monkeypatch.setattr(study, "hardware_provenance", lambda: {"device": "mock"})
    monkeypatch.setattr(study, "model_provenance", lambda spec, artifacts: {"model": "fixed"})
    monkeypatch.setattr(study, "inference_identity", lambda row: dict(row))
    monkeypatch.setattr(study, "_write_annotations", lambda *args: None)
    routed = []

    def route(rows, *args):
        routed.extend(row["role"] for row in rows)
        return rows, {}

    monkeypatch.setattr(study, "_route_records", route)
    monkeypatch.setattr(generalist_factory, "resolve_generalist_spec", lambda spec: spec)
    monkeypatch.setattr(generalist_factory, "generalist_provenance", lambda *args: {"model": "fixed"})
    probe = SimpleNamespace(torch=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))
    monkeypatch.setattr(generalist_factory, "load_generalist", lambda *args: probe)
    banks, generated = [], []

    class Pool:
        def __init__(self, specs, artifacts, *, source_records, source_references):
            banks.append((source_records, source_references))

        def reset_case(self):
            pass

        def clear(self):
            pass

    class Runtime(BranchRuntime):
        def __init__(self, session, pool, row, specs, config, encoder):
            super().__init__(row=row)
            self.session, self.config = session, config

        def run(self, mode, *, policy, forced_expert):
            assert list(tmp_path.rglob("value-policy.json")), "target generated before source policy was saved"
            assert "target" not in policy["source_summary"]["sample_ids"]
            generated.append((self.row["id"], mode))
            return {"text": "A", "expert_calls": int(mode == "value_mean"), "controller_calls": 0,
                    "controller_output_tokens": 0, "trace": [], "evidence": [], "seconds": 0.1}

    monkeypatch.setattr(capability_experts, "CapabilityPool", Pool)
    monkeypatch.setattr(study, "ValueStateEncoder", lambda *args, **kwargs: object())
    monkeypatch.setattr(study, "NativeSession", lambda actual_probe, *args: SimpleNamespace(probe=actual_probe))
    monkeypatch.setattr(study, "CapabilityRuntime", Runtime)
    args = ("source.jsonl", "target.jsonl", reference_path, "config.yaml", tmp_path, tmp_path / "runs")
    result = study.run_value_study(*args, stage="source")
    assert result["target_generations"] == 0 and not generated and set(routed) == {"source"}
    assert all(row["role"] == "source" for bank, _ in banks for row in bank)
    assert all("target" not in bank_refs for _, bank_refs in banks)

    first = study.run_value_study(*args, stage="evaluate")
    assert len(generated) == 2
    assert first["results"]["generalist"]["token_f1"] == 1.0
    policy_before = next(tmp_path.rglob("value-policy.json")).read_text(encoding="utf-8")
    refs["target"] = ["different reference"]
    reference_path.write_text(json.dumps(refs), encoding="utf-8")
    second = study.run_value_study(*args, stage="evaluate")
    assert len(generated) == 2, "reference-only change must not regenerate target predictions"
    assert second["results"]["generalist"]["token_f1"] == 0
    assert first["result"] != second["result"]
    assert next(tmp_path.rglob("value-policy.json")).read_text(encoding="utf-8") == policy_before

    # Diagnosis is a separate source-only lifecycle: no target routing, fitting,
    # encoder features or policy requirement. Repeated commands reuse case caches.
    from merit_feddg import capability_diagnostics as diagnostics

    collected = []
    def diagnose(engine, actual_references, scorer, **kwargs):
        assert engine.row["role"] == "source" and actual_references == ["A"]
        collected.append(engine.row["id"])
        return {"role": "source", "sample_id": engine.row["id"],
                "group_id": engine.row["group_id"], "domain": engine.row["domain"],
                "domain_kind": "proxy", "modality": "pathology", "branches": [],
                "compositions": [], "baseline": {"text": "A"}}

    def forbidden(*args, **kwargs):
        raise AssertionError("diagnosis must not fit a policy or construct its encoder")

    monkeypatch.setattr(diagnostics, "collect_diagnostic_case", diagnose)
    monkeypatch.setattr(study, "ValueStateEncoder", forbidden)
    monkeypatch.setattr(study, "fit_value_policy", forbidden)
    # A diagnostic-only tool menu need not contain the policy's feature encoder.
    config["experts"] = {"native_tool": {}}
    previous_routed, previous_generated = len(routed), len(generated)
    diagnosed = study.run_value_study(*args, stage="diagnose")
    assert diagnosed["target_generations"] == 0 and not diagnosed["policy_fitted"]
    assert len(collected) == 4 and set(routed[previous_routed:]) == {"source"}
    assert len(generated) == previous_generated
    study.run_value_study(*args, stage="diagnose")
    assert len(collected) == 4
