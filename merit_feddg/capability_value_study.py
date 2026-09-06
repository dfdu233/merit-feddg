"""Real paired source interventions and prefix-conditioned tool value evaluation.

This does not fabricate an expert outcome when no tool was executed. Source
branches share exact token IDs and a no-further-tools continuation policy.
Target references are only accessed after the source policy has been frozen.
"""

from __future__ import annotations

import importlib
import json
import math
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

from .capability_features import ValueStateEncoder, action_key, state_kind
from .capability_runtime import (
    CapabilityRuntime,
    NativeSession,
    NativeState,
    ValueGenerationConfig,
)
from .capability_study import (
    _filter_optional_experts,
    _route_records,
    _write_annotations,
    capability_summary,
)
from .capability_value import fit_value_policy
from .contribution import answer_metrics
from .extract import _extraction_runtime_provenance
from .io import load_yaml
from .open_data import audit_open_split, read_manifest
from .open_study import (
    atomic_json,
    fingerprint,
    hardware_provenance,
    inference_identity,
    model_provenance,
)


class InterventionScorer:
    """A continuous [0,1] outcome, evaluated after inference, never as a feature.

    A custom local factory returns callable(question, prediction, references).
    Its implementation version/hash must be declared in the config. Neither
    this hook nor lexical F1 is itself a validated hallucination evaluator.
    """

    def __init__(self, config):
        self.config = config
        self.function = None
        if config.get("name", "token_f1") != "token_f1":
            if not config.get("factory") or not config.get("fingerprint"):
                raise ValueError("custom continuous scorers require factory and fingerprint")
            module, name = config["factory"].split(":", 1)
            self.function = getattr(importlib.import_module(module), name)(**config.get("kwargs", {}))

    def __call__(self, row, output, references):
        value = (
            self.function(question=row["question"], prediction=output["text"], references=references)
            if self.function else answer_metrics(output["text"], references)["token_f1"]
        )
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("quality scorer must return a finite continuous number")
        if not 0 <= value <= 1:
            raise ValueError("quality scorer must return a value in [0,1]")
        return float(value)


def collect_source_case(runtime, references, scorer, *, collect_continuations=True,
                        pair_first_tools=(), max_pair_first_tools=2, verify_block_none=False):
    """Collect all legal actions, including negative/empty outcomes, at fixed states.

    The configurable pair roots are predeclared tool names, never selected with
    target answers. A->B is a conditional intervention, not sum(single gains).
    The first action still has a myopic label; we do not claim global planning.
    """
    row = runtime.row
    if row["role"] != "source":
        raise ValueError("only real source cases may train tool value")
    if type(max_pair_first_tools) is not int or max_pair_first_tools < 0:
        raise ValueError("max_pair_first_tools must be a nonnegative integer")
    if (not isinstance(pair_first_tools, (list, tuple))
            or any(not isinstance(name, str) or not name for name in pair_first_tools)):
        raise ValueError("pair_first_tools must be a list of explicit expert names")
    initial = NativeState()
    states = [initial]
    completed, executions, records, branches = {}, {}, [], []

    def identity(state):
        return fingerprint(asdict(state))

    def finish(state):
        key = identity(state)
        if key not in completed:
            completed[key] = runtime.complete(state)
        return completed[key]

    def execute(state, descriptor):
        key = (identity(state), action_key(row, descriptor))
        if key not in executions:
            executions[key] = runtime.execute(state, descriptor)
        return executions[key]

    if verify_block_none:
        plain, blocked = finish(initial), runtime.run("block_none")
        if plain["token_ids"] != blocked["token_ids"]:
            raise RuntimeError(
                f"Source {row['id']}: block-NONE differs from single-pass generation. "
                "Audit deterministic image preprocessing and prefix decoding before fitting utility."
            )

    def continuation(state):
        output = finish(state)
        end = len(state.prefix) + runtime.config.block_tokens
        # If the model already finished in this block, inventing a later state
        # would turn short-answer VQA into a spurious repeated controller task.
        if len(output["token_ids"]) > end:
            return replace(state, prefix=tuple(output["token_ids"][:end]))
        return None

    roots = runtime.descriptors(initial)
    selected = [d for d in roots if d["expert"] in pair_first_tools][:max_pair_first_tools]
    adopted_roots = []
    if runtime.config.max_expert_calls > 1:
        for descriptor in selected:
            after, event = execute(initial, descriptor)
            if event["executed"] and event["adopted"]:
                states.append(after)
                adopted_roots.append(descriptor["expert"])
    if collect_continuations:
        states += [later for state in list(states) if (later := continuation(state)) is not None]
    for state in states:
        descriptors = runtime.descriptors(state)
        candidates = runtime.candidates(state, descriptors)
        baseline = finish(state)
        # All feature extraction above precedes observing the new tool result.
        base_score = scorer(row, baseline, references)
        for descriptor, candidate in zip(descriptors, candidates, strict=True):
            after, event = execute(state, descriptor)
            guided = finish(after)
            gain = scorer(row, guided, references) - base_score
            incremental = max(0.0, event["seconds"] + guided["seconds"] - baseline["seconds"])
            record = {
                **candidate, "role": "source", "sample_id": row["id"],
                "group_id": row["group_id"], "domain": row["domain"],
                "state_id": identity(state), "gain": gain, "cost": incremental,
                "executed": event["executed"], "adopted": event["adopted"],
            }
            records.append(record)
            branches.append({
                "state_id": identity(state), "state_kind": state_kind(state),
                "history_actions": list(state.history), "prefix_tokens": list(state.prefix),
                "action_key": candidate["action_key"], "without": baseline, "with": guided,
                "tool": event, "gain": gain,
            })
    return {"records": records, "branches": branches, "baseline": finish(initial),
            "requested_pair_roots": [d["expert"] for d in selected], "pair_roots": adopted_roots,
            "cost_note": "one measured tool plus continuation time difference; cold/warm mixed"}


def run_value_study(source_path, target_path, references_path, config_path, artifacts, output,
                    *, stage="all"):
    from .capability_experts import CapabilityPool
    from .generalist_factory import (
        generalist_provenance,
        load_generalist,
        resolve_generalist_spec,
    )

    if stage not in {"all", "source", "evaluate"}:
        raise ValueError("stage must be all, source or evaluate")
    source, target = read_manifest(source_path, "source"), read_manifest(target_path, "target")
    audit_open_split(source, target)
    config = load_yaml(config_path)
    config["generalist"] = resolve_generalist_spec(config["generalist"])
    value = config.get("capability_value", {})
    decoder = ValueGenerationConfig(**value.get("generation", {}))
    specs, excluded = _filter_optional_experts(config["experts"], artifacts)
    encoder_options = dict(value.get("encoder", {}))
    encoder_tool = encoder_options.pop("expert", "source_cases")
    if encoder_tool not in specs:
        raise ValueError("value encoder requires a configured local contrastive expert")
    refs = json.loads(Path(references_path).read_text(encoding="utf-8"))
    source_refs = {row["id"]: refs[row["id"]] for row in source}
    scorer_config = value.get("quality", {"name": "token_f1"})
    scorer = InterventionScorer(scorer_config)
    probe = None

    def ensure_probe():
        nonlocal probe
        if probe is None:
            probe = load_generalist(config["generalist"], artifacts)
        return probe

    runtime_identity = {
        "generalist": generalist_provenance(config["generalist"], artifacts),
        "generalist_settings": config["generalist"],
        "runtime": _extraction_runtime_provenance(), "hardware": hardware_provenance(),
    }
    source, source_routes = _route_records(
        source, config.get("routing", {}), ensure_probe, runtime_identity, output
    )
    provenance = {
        "schema": "native-capability-value-v09", "config": config, **runtime_identity,
        "experts": {name: model_provenance(spec, artifacts) for name, spec in specs.items()},
        "excluded_tools": excluded,
        "source": [inference_identity(row) for row in source], "source_references": source_refs,
        "quality": scorer_config,
        "lodo_scope": "policy_regression_only_shared_source_retrieval_bank",
    }
    run_key = fingerprint(provenance)
    root = Path(output) / run_key[:16]
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(root / "provenance.json", provenance)
    pool = CapabilityPool(specs, artifacts, source_records=source, source_references=source_refs)
    encoder = ValueStateEncoder(pool, specs[encoder_tool], max_tokens=decoder.max_new_tokens,
                                max_calls=decoder.max_expert_calls, **encoder_options)

    def make_runtime(row):
        pool.reset_case()
        prompt = row["question"] + "\n" + config.get("prompt_suffix", "Answer concisely from the image.")
        session = NativeSession(ensure_probe(), row["image"], prompt, row["question"], decoder)
        return CapabilityRuntime(session, pool, row, specs, decoder, encoder)

    def cached_case(row, method, callback, *, policy_key=None):
        key = fingerprint({"run": run_key, "input": inference_identity(row),
                           "method": method, "policy": policy_key})
        path = Path(output) / "value-case-cache" / f"{key}.json"
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            if saved.get("cache_key") == key:
                return saved
        print(f"Value study {row['role']} {row['id']} {method}", flush=True)
        engine = make_runtime(row)
        cuda = engine.session.probe.torch.cuda
        if cuda.is_available():
            for device in range(cuda.device_count()):
                cuda.synchronize(device)
                cuda.reset_peak_memory_stats(device)
        started = perf_counter()
        result = callback(engine)
        if cuda.is_available():
            for device in range(cuda.device_count()):
                cuda.synchronize(device)
            result["peak_allocated_gib"] = sum(
                cuda.max_memory_allocated(i) for i in range(cuda.device_count())
            ) / 2**30
        result.update(cache_key=key, generation_seconds=perf_counter() - started)
        atomic_json(path, result)
        return result

    policy_path = root / "value-policy.json"
    try:
        if stage != "evaluate":
            records = []
            for row in source:
                data = cached_case(row, "source-paired", lambda engine, row=row: collect_source_case(
                    engine, source_refs[row["id"]], scorer, **value.get("collection", {})
                ))
                records.extend(data["records"])
            atomic_json(root / "source-interventions.json", records)
            policy = fit_value_policy(records, {"source_run_key": run_key, "quality": scorer_config},
                                      **value.get("fit", {}))
            atomic_json(policy_path, policy)
        else:
            if not policy_path.exists():
                raise FileNotFoundError("matching source-fitted policy missing; run --value-stage source first")
            policy = json.loads(policy_path.read_text(encoding="utf-8"))
            if policy["provenance"].get("source_run_key") != run_key:
                raise ValueError("policy fingerprint does not match model/source/config")
        if stage == "source":
            report = {"stage": "source", "policy": str(policy_path), "source": policy["source_summary"],
                      "target_generations": 0, "target_labels_used": False}
            atomic_json(root / "source-result.json", report)
            return report

        # Freeze policy first, then route/generate targets. No target features
        # normalize/refit it, and target labels never enter the retrieval bank.
        target, target_routes = _route_records(
            target, config.get("routing", {}), ensure_probe, runtime_identity, output
        )
        policy_key = fingerprint(policy)
        predictions = {}
        modes = value.get("methods", ["generalist", "block_none", "all_evidence", "agent",
                                      "value_mean", "value_robust"])
        if "generalist" not in modes:
            raise ValueError("the unmodified generalist baseline is required")
        methods = [(mode, mode, None) for mode in modes]
        if value.get("single_tools", True):
            methods += [(f"single:{name}", "all_evidence", name) for name in specs]
        for name, mode, expert in methods:
            outputs = {}
            for row in target:
                result = cached_case(row, name, lambda engine, mode=mode, expert=expert: engine.run(
                    mode, policy=policy, forced_expert=expert
                ), policy_key=policy_key if mode.startswith("value_") else None)
                routing_seconds = 0 if mode in {"generalist", "block_none"} else target_routes.get(
                    row["id"], {}).get("seconds", 0)
                outputs[row["id"]] = {**result, "routing_seconds": routing_seconds,
                                      "seconds": result["generation_seconds"] + routing_seconds}
            predictions[name] = outputs
        references = {row["id"]: refs[row["id"]] for row in target}
        results = {name: capability_summary(outputs, predictions["generalist"], references, target)
                   for name, outputs in predictions.items()}
        for name, outputs in predictions.items():
            quality = [scorer(row, outputs[row["id"]], references[row["id"]]) for row in target]
            results[name]["configured_quality_mean"] = sum(quality) / len(quality)
        report = {
            "run_key": run_key, "policy_key": policy_key, "results": results,
            "quality": scorer_config, "source_summary": policy["source_summary"],
            "conditions": policy["conditions"], "source_count": len(source), "target_count": len(target),
            "domain_kinds": sorted({row["domain_kind"] for row in source + target}),
            "excluded_tools": excluded,
            "protocol_audit": {
                "baseline_empty_answers": sum(not output["text"].strip()
                                              for output in predictions["generalist"].values()),
                "block_none_same_tokens_fraction": sum(
                    predictions["block_none"][row["id"]].get("token_ids")
                    == predictions["generalist"][row["id"]].get("token_ids") for row in target
                ) / len(target) if "block_none" in predictions else None,
                "deterministic_image_padding": config["generalist"].get("deterministic_image_padding", False),
                "answer_length_limit": decoder.max_new_tokens,
                "original_image_always_preserved": True,
            },
            "limitations": [
                "Token-F1/EM and trained lexical utility are not clinical factuality or hallucination rates.",
                "Proxy/hash groups do not establish hospital-domain generalization.",
                "LODO is policy-only: the retrieval bank is shared source data, not fold-isolated pipeline data.",
                "Empirical source overprediction penalties do not guarantee safety under arbitrary target shift.",
                "Greedy next-tool value can miss pairs whose first tool has no standalone benefit.",
                "Re-prefill and predicted-image encoding cost time; no KV-cache decoding speedup is claimed.",
                "Detection/ROI plugin contracts exist but require validated adapters/data for efficacy claims.",
            ],
        }
        evaluation_key = fingerprint({"run": run_key, "policy": policy_key,
                                      "target": [inference_identity(row) for row in target],
                                      "references": references})
        evaluation_root = root / "evaluations" / evaluation_key[:16]
        evaluation_root.mkdir(parents=True, exist_ok=True)
        atomic_json(evaluation_root / "result.json", report)
        atomic_json(evaluation_root / "predictions.json", predictions)
        atomic_json(evaluation_root / "routing.json", {**source_routes, **target_routes})
        _write_annotations(evaluation_root, predictions, target)
        lines = ["# Native capability value study", "", "Lexical metrics are NOT hallucination rates.", "",
                 "| Method | Token-F1 | Delta | Calls/case | Seconds/case |",
                 "| --- | ---: | ---: | ---: | ---: |"]
        for name, result in results.items():
            lines.append(f"| {name} | {result['token_f1']:.5f} | {result['paired_f1_gain']:+.5f} | "
                         f"{result['mean_expert_calls']:.3f} | {result['mean_seconds']:.3f} |")
        lines += ["", "## Boundaries", "", *[f"- {v}" for v in report["limitations"]]]
        (evaluation_root / "result.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        atomic_json(Path(output) / "latest-value.json", {"result": str(evaluation_root / "result.json"),
                                                       "policy": str(policy_path)})
        return {"result": str(evaluation_root / "result.json"), "results": results}
    finally:
        pool.clear()
