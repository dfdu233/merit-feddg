from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from merit_feddg.capability_study import (
    capability_summary,
    run_capability_study,
    scope_qualification,
)
from merit_feddg.io import save_yaml
from merit_feddg.open_data import INFERENCE_FIELDS, pixel_digest


def gains(values):
    return [
        {"role": "source", "domain": domain, "base_f1": 0.0, "guided_f1": value}
        for domain in ("a", "b")
        for value in values
    ]


def test_scope_qualification_distinguishes_missing_support_from_negative_gain():
    assert scope_qualification(gains([0.25]), min_per_domain=2)["status"] == "insufficient_support"
    positive = scope_qualification(gains([0.25, 0.25]), min_per_domain=2)
    assert positive["qualified"] and positive["status"] == "qualified"
    missing = scope_qualification(
        gains([0.25, 0.25]), expected_domains=["a", "b", "never-called-c"], min_per_domain=2
    )
    assert not missing["qualified"] and missing["status"] == "insufficient_support"
    assert missing["missing_intervention_domains"] == ["never-called-c"]
    unproven = scope_qualification(gains([0.0, 0.5]), min_per_domain=2)
    assert not unproven["qualified"] and unproven["status"] == "unproven_gain"
    harmful = [{**row, "base_f1": 0.5, "guided_f1": 0.0} for row in gains([0.0, 0.0])]
    assert scope_qualification(harmful, min_per_domain=2)["status"] == "observed_negative_gain"
    with pytest.raises(ValueError, match="target"):
        scope_qualification([{**harmful[0], "role": "target"}])


def test_summary_reports_actual_generation_calls_and_lexical_limits():
    rows = [{"id": "a", "domain": "target-a"}, {"id": "b", "domain": "target-b"}]
    base = {key: {"text": "wrong", "expert_calls": 0, "seconds": 0.1} for key in ("a", "b")}
    outputs = {
        "a": {"text": "tumor", "expert_calls": 2, "seconds": 0.3},
        "b": {"text": "wrong", "expert_calls": 0, "seconds": 0.2},
    }
    result = capability_summary(outputs, base, {"a": ["tumor"], "b": ["normal"]}, rows)
    assert result["token_f1"] == 0.5
    assert result["f1_improved"] == 1 and result["f1_harmed"] == 0
    assert result["mean_expert_calls"] == 1.0
    assert result["expert_call_fraction"] == 0.5
    assert result["worst_domain_f1"] == 0
    assert "NOT medical hallucination" in result["metric_warning"]
    assert result["mean_controller_tokens"] is None  # Unmeasured is not zero.


def make_case(tmp_path, sample_id, role, value):
    image = tmp_path / f"{sample_id}.png"
    Image.new("RGB", (4, 4), (value, value, value)).save(image)
    return {
        "id": sample_id,
        "image": str(image),
        "question": "What is the tissue and related appearance?",
        "modality": "pathology",
        "capability": "classification",
        "task": "open_vqa",
        "domain": f"{role}-{value % 2}",
        "domain_kind": "proxy",
        "role": role,
        "group_id": sample_id,
        "image_sha256": pixel_digest(image),
    }


@pytest.mark.parametrize("real_engine", [False, True])
def test_real_runner_keeps_target_labels_out_of_tools_and_binds_source_index(
    tmp_path, monkeypatch, real_engine
):
    from merit_feddg import capability_experts, capability_generation, generalist
    from merit_feddg import capability_study as study

    source = [make_case(tmp_path, f"s{i}", "source", i) for i in range(4)]
    target = [make_case(tmp_path, "t", "target", 12)]
    for name, rows in (("source", source), ("target", target)):
        (tmp_path / f"{name}.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    references = {row["id"]: ["tumor"] for row in source + target}
    refs_path = tmp_path / "refs.json"
    refs_path.write_text(json.dumps(references))
    config_path = tmp_path / "config.yaml"
    config = {
        "generalist": {"id": "tiny-generalist"},
        "qualification": {"min_per_domain": 2},
        "experts": {
            "tissue": {
                "id": "tiny-classifier",
                "modalities": ["pathology"],
                "tasks": ["open_vqa"],
                "capabilities": ["classification"],
                "scope": "tissue",
            },
            "cases": {
                "id": "tiny-retriever",
                "modalities": ["pathology"],
                "tasks": ["open_vqa"],
                "capabilities": ["retrieval"],
                "scope": "cases",
            },
        },
    }
    save_yaml(config_path, config)
    monkeypatch.setattr(study, "model_provenance", lambda *_: {"test": "mock"})
    monkeypatch.setattr(study, "_extraction_runtime_provenance", lambda: {"test": "mock"})
    monkeypatch.setattr(study, "hardware_provenance", lambda: {"test": "mock"})
    monkeypatch.setattr(
        generalist,
        "QwenLayerProbe",
        lambda *a, **k: SimpleNamespace(
            torch=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        ),
    )
    counters, pools = [], []

    class Session:
        def __init__(self, probe, image, prompt):
            counters.append((image, prompt))
            self.is_source = Path(image).stem.startswith("s")

        def decode(self, tokens):
            return " ".join("tumor" if token == 2 else "wrong" for token in tokens)

        def control(self, state, max_tokens):
            observed = {item["expert_id"] for item in state["observations"]}
            for tool in state["available_tools"]:
                if tool["expert"] in observed:
                    continue
                if self.is_source and tool["capability"] == "retrieval":
                    continue
                return json.dumps(
                    {
                        "action": "call",
                        **{k: tool[k] for k in ("expert", "capability", "scope")},
                        "query": "Provide a scoped native observation",
                        "region": None,
                    }
                )
            return '{"action":"continue"}'

        def next_block(self, prefix, memory, length):
            from merit_feddg.block_decode import Block

            token = 2 if memory else 1
            return Block((token,), self.decode((token,)), -0.1, True)

    monkeypatch.setattr(
        capability_generation,
        "QwenCapabilitySession",
        Session if real_engine else lambda *_: object(),
    )

    class Pool:
        def __init__(self, specs, artifacts, *, source_records, source_references):
            assert all(row["role"] == "source" for row in source_records)
            assert set(source_references) == {row["id"] for row in source_records}
            assert "t" not in source_references
            pools.append(source_references)

        def reset_case(self):
            pass

        def clear(self):
            pass

        def infer(self, expert_id, request):
            from merit_feddg.capabilities import CapabilityResult, EvidenceItem

            assert not hasattr(request, "reference") and not hasattr(request, "answer")
            item = EvidenceItem(
                evidence_id=expert_id,
                expert_id=expert_id,
                capability=request.capability,
                scope=request.scope,
                payload={"fixture_native_observation": "tumor-like morphology"},
            )
            return CapabilityResult(expert_id, request.capability, (item,))

    def generate(session, pool, row, decoder, specs, *, cards, mode, allowed_pairs):
        assert set(row) == INFERENCE_FIELDS
        counters.append((row["id"], mode, allowed_pairs))
        if mode == "generalist":
            calls = 0
        elif allowed_pairs is not None:
            # Even a legacy classification row can request retrieval. The test
            # gives that scope no calls, which must not qualify the whole tool.
            calls = int(next(iter(allowed_pairs))[1] == "classification")
        elif mode == "adaptive_dg":
            calls = sum(bool(card["qualified"]) for card in cards.values())
        else:
            calls = 2
        return {
            "text": "tumor" if calls else "wrong",
            "expert_calls": calls,
            "controller_calls": 1,
            "trace": [{"event": "generation"}],
            "evidence": [{"test": "native"}] * calls,
        }

    monkeypatch.setattr(capability_experts, "CapabilityPool", Pool)
    if not real_engine:
        monkeypatch.setattr(capability_generation, "generate_capabilities", generate)
    args = (
        tmp_path / "source.jsonl",
        tmp_path / "target.jsonl",
        refs_path,
        config_path,
        tmp_path,
        tmp_path / "out",
    )
    first = run_capability_study(*args)
    assert first["results"]["adaptive_dg"]["token_f1"] == 1.0
    assert set(first["results"]) == {"generalist", "all_evidence", "adaptive_no_dg", "adaptive_dg"}
    cards = first["qualification"]["cards"]
    assert cards["tissue|pathology|open_vqa|classification|tissue"]["qualified"]
    assert not cards["cases|pathology|open_vqa|retrieval|cases"]["qualified"]
    assert first["results"]["adaptive_no_dg"]["mean_expert_calls"] == 2
    if real_engine:
        assert first["results"]["adaptive_no_dg"]["tool_calls_by_capability"] == {
            "classification": 1,
            "retrieval": 1,
        }
        assert first["results"]["adaptive_no_dg"]["multi_expert_case_fraction"] == 1.0
        assert first["results"]["adaptive_no_dg"]["multi_capability_case_fraction"] == 1.0
        sequence = first["results"]["adaptive_no_dg"]["expert_sequences"][0]
        assert sequence["n"] == 1
        assert [event["expert"] for event in sequence["sequence"]] == ["tissue", "cases"]
    assert (Path(first["run_dir"]) / "result.md").exists()
    count = len(counters)

    references["t"] = ["changed target label"]
    refs_path.write_text(json.dumps(references))
    second = run_capability_study(*args)
    assert len(counters) == count
    assert first["qualification"] == second["qualification"]
    assert first["run_dir"] != second["run_dir"]
    assert second["results"]["adaptive_dg"]["token_f1"] == 0.0

    references["s0"] = ["different source retrieval reference"]
    refs_path.write_text(json.dumps(references))
    third = run_capability_study(*args)
    assert len(counters) > count
    assert third["qualification"]["source_data_key"] != first["qualification"]["source_data_key"]
    assert pools[-1]["s0"] == references["s0"]
    count = len(counters)
    monkeypatch.setattr(study, "_extraction_runtime_provenance", lambda: {"test": "changed code"})
    run_capability_study(*args)
    assert len(counters) > count

    # Factuality annotation templates do not expose methods or answers to raters.
    annotations = json.loads((Path(first["run_dir"]) / "annotation-blinded.json").read_text())
    assert all("method" not in item and "reference" not in item for item in annotations)
