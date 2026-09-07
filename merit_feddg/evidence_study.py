"""Real source diagnostics and separately unlocked target evidence experiments.

Initial-scope bridge study: compare the SAME native tool result at the SAME
prefix, not an untrained agent or counterfactual cached candidate answer scores.
"""

from __future__ import annotations

import gc
import hashlib
import json
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import numpy as np

from .bounded_session import BoundedNativeSession
from .capabilities import scoped_key
from .capability_runtime import CapabilityRuntime, NativeSession, NativeState, ValueGenerationConfig
from .capability_study import (
    _filter_optional_experts,
    _route_records,
    _write_annotations,
    capability_summary,
)
from .capability_value_study import InterventionScorer
from .evidence_calibration import fit_evidence_calibration
from .evidence_decode import GuidanceConfig
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


def descriptor_scope(row, descriptor):
    return scoped_key(descriptor["expert"], row["modality"], row["task"],
                      descriptor["capability"], descriptor["scope"])


def compare_source_case(runtime, references, scorer, guidance, strengths):
    """Tools executed once/case/scope, then reused for presentation diagnostics.

    Replay time is explicitly NOT online end-to-end method latency.
    Empty tool output remains a zero-gain observation, not a missing record.
    """
    if runtime.row["role"] != "source":
        raise ValueError("source diagnostics require a source row")
    baseline = runtime.complete(NativeState())
    token_session = runtime.session.probe.new_answer_session(runtime.row["image"], runtime.session.prompt)
    # Verify the score adapter against actual production generation before any
    # evidence is introduced. Zero-strength parity alone is tautological because
    # it deliberately bypasses the new token loop.
    for position, token in enumerate(baseline["token_ids"][:guidance.evidence_tokens]):
        scores = token_session.next_scores(tuple(baseline["token_ids"][:position]))
        if int(np.argmax(scores)) != token:
            raise RuntimeError("next-token score adapter differs from production baseline")
    records, branches = [], []
    for descriptor in runtime.descriptors(NativeState()):
        after, tool = runtime.execute(NativeState(), descriptor)
        if not tool["executed"]:
            raise RuntimeError(f"source tool failure is not zero clinical utility: {tool['reason']}")
        # Acquire positions are deterministic even for the direct-context runtime.
        after = replace(after, items=tuple(replace(item, provenance={**item.provenance,
                        "merit_acquired_token": 0}) for item in after.items))
        direct = runtime.complete(after)
        base_quality = scorer(runtime.row, baseline, references)
        for strength in strengths:
            session = BoundedNativeSession(
                runtime.session.probe, runtime.row["image"], runtime.session.prompt,
                runtime.row["question"], runtime.config, replace(guidance, strength=strength),
            )
            engine = CapabilityRuntime(session, runtime.pool, runtime.row, runtime.specs, runtime.config)
            guided = engine.complete(after)
            control_session = BoundedNativeSession(
                runtime.session.probe, runtime.row["image"], runtime.session.prompt,
                runtime.row["question"], runtime.config,
                replace(guidance, strength=strength, control="format_only"),
            )
            control = CapabilityRuntime(control_session, runtime.pool, runtime.row,
                                        runtime.specs, runtime.config).complete(after)
            quality = {"baseline": base_quality, "direct_context": scorer(runtime.row, direct, references),
                       "guided": scorer(runtime.row, guided, references),
                       "format_only": scorer(runtime.row, control, references)}
            gain = quality["guided"] - base_quality
            record = {"role": "source", "scope": descriptor_scope(runtime.row, descriptor),
                      "sample_id": runtime.row["id"], "group_id": runtime.row["group_id"],
                      "domain": runtime.row["domain"], "domain_kind": runtime.row["domain_kind"],
                      "strength": strength, "gain": gain}
            records.append(record)
            branches.append({**record, "baseline": baseline, "direct_context": direct,
                             "guided": guided, "format_only": control, "tool": tool, "quality": quality})
    return {"records": records, "branches": branches, "baseline": baseline,
            "tokenwise_baseline_checked": min(len(baseline["token_ids"]), guidance.evidence_tokens),
            "latency_note": "source evidence replay; not online end-to-end latency"}


def source_bridge_summary(branches):
    """Independent-group domain means, including harmful and empty observations."""
    groups = defaultdict(lambda: defaultdict(list))
    for row in branches:
        for arm in ("direct_context", "guided", "format_only"):
            key = (row["scope"], row["domain"], row["strength"], arm)
            groups[key][row["group_id"]].append(row["quality"][arm] - row["quality"]["baseline"])
    result = []
    for (scope, domain, strength, arm), values in sorted(groups.items()):
        gains = [float(np.mean(samples)) for samples in values.values()]
        result.append({"scope": scope, "domain": domain, "strength": strength, "arm": arm,
                       "independent_groups": len(gains), "mean_gain": float(np.mean(gains)),
                       "positive_groups": sum(v > 0 for v in gains),
                       "negative_groups": sum(v < 0 for v in gains)})
    return result


def run_evidence_study(source_path, target_path, references_path, config_path, artifacts, output,
                       *, stage="source"):
    from .capability_experts import CapabilityPool
    from .generalist_factory import generalist_provenance, load_generalist, resolve_generalist_spec

    if stage not in {"source", "evaluate"}:
        raise ValueError("evidence stage must be source or evaluate")
    config = load_yaml(config_path)
    config["generalist"] = resolve_generalist_spec(config["generalist"])
    if config["generalist"].get("backend") != "llava_med":
        raise ValueError("bounded evidence backend currently supports official LLaVA-Med only")
    options = config.get("bounded_evidence", {})
    decoder = ValueGenerationConfig(**options.get("generation", {}))
    if decoder.visual_views != 0:
        raise ValueError("bounded evidence study requires visual_views=0")
    guidance = GuidanceConfig(**options.get("guidance", {}))
    strengths = options.get("strengths", [0.25, 0.5])
    if (not strengths or len(set(strengths)) != len(strengths)
            or any(isinstance(s, bool) or not isinstance(s, (int, float)) or not 0 < s <= 1
                   for s in strengths)):
        raise ValueError("provide distinct source strengths in (0,1]")
    if guidance.control != "real":
        raise ValueError("study guidance control must be real; controls are generated separately")
    source, target = read_manifest(source_path, "source"), read_manifest(target_path, "target")
    audit_open_split(source, target)
    refs = json.loads(Path(references_path).read_text(encoding="utf-8"))
    source_refs = {r["id"]: refs[r["id"]] for r in source}
    source_specs, excluded = _filter_optional_experts(config["experts"], artifacts)
    # A retrieval bank containing the held-out calibration groups would defeat
    # independent confirmation. Keep retrieval outside this source-fit profile.
    specs = {k: v for k, v in source_specs.items() if "retrieval" not in v["capabilities"]}
    scorer = InterventionScorer(options.get("quality", {"name": "token_f1"}))
    probe = None

    def ensure_probe():
        nonlocal probe
        if probe is None:
            probe = load_generalist(config["generalist"], artifacts)
        return probe

    code = {p.relative_to(Path(__file__).parent).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(__file__).parent.rglob("*.py"))}
    identity = {"schema": "bounded-evidence-v011", "config": config, "code": code,
                "runtime": _extraction_runtime_provenance(), "hardware": hardware_provenance(),
                "generalist": generalist_provenance(config["generalist"], artifacts),
                "experts": {k: model_provenance(v, artifacts) for k, v in specs.items()}}
    source, source_routes = _route_records(source, config.get("routing", {}), ensure_probe, identity, output)
    source_key = fingerprint({**identity, "source": [inference_identity(r) for r in source],
                              "source_references": source_refs})
    root = Path(output) / source_key[:16]
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(root / "provenance.json", {**identity, "source_key": source_key,
                                          "excluded_optional_experts": excluded})
    pool = CapabilityPool(specs, artifacts, source_records=[], source_references={})
    policy_path = root / "evidence-policy.json"

    def engine(row, setting=None):
        prompt = row["question"] + "\n" + config.get("prompt_suffix", "Answer concisely from the image.")
        if setting is None:
            session = NativeSession(ensure_probe(), row["image"], prompt, row["question"], decoder)
        else:
            session = BoundedNativeSession(ensure_probe(), row["image"], prompt, row["question"], decoder, setting)
        return CapabilityRuntime(session, pool, row, specs, decoder)

    def cached(row, name, callback, *, policy_key=None):
        key = fingerprint([source_key, inference_identity(row), name, policy_key])
        path = root / "case-cache" / f"{key}.json"
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            if saved.get("cache_key") == key:
                return saved
        pool.reset_case()
        print(f"Evidence study {row['role']} {row['id']} {name}", flush=True)
        cuda = ensure_probe().torch.cuda
        try:
            if cuda.is_available():
                for device in range(cuda.device_count()):
                    cuda.synchronize(device)
                    cuda.reset_peak_memory_stats(device)
            started = perf_counter()
            result = callback()
            if cuda.is_available():
                for device in range(cuda.device_count()):
                    cuda.synchronize(device)
                result["peak_allocated_gib"] = sum(cuda.max_memory_allocated(i)
                                                    for i in range(cuda.device_count())) / 2**30
                result["peak_reserved_gib"] = sum(cuda.max_memory_reserved(i)
                                                   for i in range(cuda.device_count())) / 2**30
            result.update(cache_key=key, seconds=perf_counter() - started)
            atomic_json(path, result)
            return result
        finally:
            pool.reset_case()
            gc.collect()
            if probe is not None and probe.torch.cuda.is_available():
                probe.torch.cuda.empty_cache()

    try:
        if stage == "source":
            records, cases, branches = [], [], []
            for row in source:
                def diagnose(row=row):
                    runtime = engine(row)
                    plain = runtime.complete(NativeState())
                    blocked = runtime.run("block_none")
                    zero = engine(row, replace(guidance, strength=0.0)).complete(NativeState())
                    if not plain["token_ids"] == blocked["token_ids"] == zero["token_ids"]:
                        raise RuntimeError("baseline token parity failed; stop source calibration")
                    return compare_source_case(runtime, source_refs[row["id"]], scorer, guidance, strengths)
                result = cached(row, "source-bridges", diagnose)
                records.extend(result["records"])
                branches.extend(result["branches"])
                cases.append({"id": row["id"], "cache_key": result["cache_key"]})
            policy = fit_evidence_calibration(records, **options.get("calibration", {}))
            policy["source_key"] = source_key
            policy["source_interventions_sha256"] = fingerprint(records)
            atomic_json(policy_path, policy)
            atomic_json(root / "source-interventions.json", records)
            report = {"stage": stage, "policy": str(policy_path), "source_cases": len(source),
                      "records": len(records), "target_generations": 0, "cards": policy["cards"],
                      "quality_metric": scorer.config, "bridge_summary": source_bridge_summary(branches),
                      "domain_kinds": sorted({r["domain_kind"] for r in source}), "cases": cases,
                      "limitations": policy["limitations"] + ["Token-F1 is not clinical factuality."]}
            atomic_json(root / "source-result.json", report)
            lines = ["# Source-only bounded evidence diagnosis", "",
                     "Target generations: 0. Quality metric: " + scorer.config.get("name", "token_f1") + ".",
                     "Token-F1 and positive/negative groups do not establish clinical benefit or harm.", "",
                     "| Scope | Source domain | Strength | Arm | Groups | Mean gain |",
                     "| --- | --- | ---: | --- | ---: | ---: |"]
            for item in report["bridge_summary"]:
                lines.append(f"| {item['scope'].replace('|', '/')} | {item['domain']} | {item['strength']} | "
                             f"{item['arm']} | {item['independent_groups']} | {item['mean_gain']:+.5f} |")
            lines += ["", "## Calibration (not a statistical safety guarantee)", ""]
            lines += [f"- {scope}: {card['status']}" for scope, card in policy["cards"].items()]
            lines += ["", *[f"- {v}" for v in report["limitations"]], "",
                      "Source timings reuse native evidence and are not end-to-end method latency."]
            (root / "source-result.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
            atomic_json(Path(output) / "latest-evidence.json", {"result": str(root / "source-result.json")})
            return report
        if not policy_path.exists():
            raise FileNotFoundError("matching source calibration missing; run evidence-stage source first")
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
        if policy.get("source_key") != source_key:
            raise ValueError("source/model/config/code calibration fingerprint mismatch")
        records = json.loads((root / "source-interventions.json").read_text(encoding="utf-8"))
        expected_policy = fit_evidence_calibration(records, **options.get("calibration", {}))
        expected_policy.update(source_key=source_key, source_interventions_sha256=fingerprint(records))
        if policy != expected_policy:
            raise ValueError("saved source policy differs from its recorded calibration; rerun source stage")
        target, target_routes = _route_records(target, config.get("routing", {}), ensure_probe, identity, output)
        predictions = {}
        # Default single-tool and format controls share the same configured tool
        # identity across all three arms. This is not target-oracle routing.
        methods = ["generalist", "source_calibrated"]
        methods += [f"{arm}:{name}" for name in specs for arm in ("direct", "bounded", "format")]
        for method in methods:
            predictions[method] = {}
            for row in target:
                def predict(row=row, method=method):
                    runtime = engine(row)
                    descriptors = runtime.descriptors(NativeState())
                    choice, setting = None, None
                    if method == "source_calibrated":
                        eligible = [(d, policy["cards"].get(descriptor_scope(row, d), {})) for d in descriptors]
                        eligible = [(d, c) for d, c in eligible if c.get("qualified")]
                        if eligible:
                            choice, card = max(eligible, key=lambda dc: min(dc[1]["selection_gain"].values()))
                            setting = replace(guidance, strength=card["strength"])
                    elif ":" in method:
                        arm, name = method.split(":", 1)
                        choice = next((d for d in descriptors if d["expert"] == name), None)
                        if arm != "direct":
                            setting = replace(guidance, control="format_only" if arm == "format" else "real")
                    runtime = engine(row, setting)
                    state, events = NativeState(), []
                    if choice is not None:
                        state, event = runtime.execute(state, choice)
                        if not event["executed"]:
                            raise RuntimeError(f"target tool failed: {event['reason']}")
                        events.append(event)
                    result = runtime.complete(state)
                    return {**result, "expert_calls": len(events), "controller_calls": 0,
                            "controller_output_tokens": 0, "trace": events,
                            "adopted_evidence_count": sum(bool(e["adopted"]) for e in events),
                            "evidence": [asdict(i) for i in state.items],
                            "selection": choice, "guidance": asdict(setting) if setting else None}
                result = cached(row, method, predict, policy_key=fingerprint(policy))
                route_time = target_routes.get(row["id"], {}).get("seconds", 0) if method != "generalist" else 0
                predictions[method][row["id"]] = {**result, "seconds": result["seconds"] + route_time,
                                                   "generation_seconds": result["seconds"],
                                                   "routing_seconds": route_time}
        target_refs = {r["id"]: refs[r["id"]] for r in target}
        for outputs in predictions.values():
            for row in target:
                outputs[row["id"]]["quality"] = scorer(row, outputs[row["id"]], target_refs[row["id"]])
        results = {name: capability_summary(outputs, predictions["generalist"], target_refs, target)
                   for name, outputs in predictions.items()}
        for name, outputs in predictions.items():
            results[name]["configured_quality_mean"] = float(np.mean([o["quality"] for o in outputs.values()]))
            results[name]["configured_quality_gain"] = float(np.mean([
                o["quality"] - predictions["generalist"][sample]["quality"] for sample, o in outputs.items()]))
        evaluation_key = fingerprint([source_key, [inference_identity(r) for r in target], target_refs])
        evaluation_root = root / "evaluations" / evaluation_key[:16]
        evaluation_root.mkdir(parents=True, exist_ok=True)
        report = {"results": results, "source_cases": len(source), "target_cases": len(target),
                  "quality_metric": scorer.config,
                  "source_key": source_key, "domain_kinds": sorted({r["domain_kind"] for r in source + target}),
                  "limitations": policy["limitations"] + [
                      "Initial-scope single-tool bridge study; no agent/multi-tool composition claim.",
                      "Token-F1 is not hallucination rate. Format-only is not exact length-matched.",
                      ("Guided scoring replays each prefix through two production KV paths; "
                       "no cross-branch KV sharing.")]}
        atomic_json(evaluation_root / "result.json", report)
        atomic_json(evaluation_root / "predictions.json", predictions)
        _write_annotations(evaluation_root, predictions, target)
        atomic_json(evaluation_root / "routes.json", {"source": source_routes, "target": target_routes})
        lines = ["# Bounded evidence study", "", "Lexical F1 is NOT clinical factuality.", "",
                 "| Method | F1 | Delta | Calls/case | Seconds/case |",
                 "| --- | ---: | ---: | ---: | ---: |"]
        for name, result in results.items():
            lines.append(f"| {name} | {result['token_f1']:.5f} | {result['paired_f1_gain']:+.5f} | "
                         f"{result['mean_expert_calls']:.2f} | {result['mean_seconds']:.3f} |")
        lines += ["", *[f"- {v}" for v in report["limitations"]]]
        (evaluation_root / "result.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        atomic_json(Path(output) / "latest-evidence.json", {"result": str(evaluation_root / "result.json")})
        return report
    finally:
        pool.clear()
