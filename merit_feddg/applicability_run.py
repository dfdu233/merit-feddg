"""Offline native capability collaboration with source-calibrated applicability.

collect: forced source interventions + frozen native features, no selective-label bias.
evaluate: existing dynamic agent with four optional input-admission ablations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from .applicability import ApplicabilityConfig, ApplicabilityGate, build_memory
from .capability_experts import CapabilityPool
from .capability_features import action_key
from .capability_runtime import CapabilityRuntime, NativeSession, NativeState, ValueGenerationConfig
from .capability_study import _filter_optional_experts, _route_records, capability_summary
from .capability_value_study import InterventionScorer
from .generalist_factory import generalist_provenance, load_generalist, resolve_generalist_spec
from .io import load_experiment_yaml
from .open_data import audit_open_split, pixel_digest, read_manifest
from .open_study import atomic_json, fingerprint, model_provenance


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def code_identity():
    return {str(p.relative_to(Path(__file__).parent)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(__file__).parent.rglob("*.py")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["collect", "evaluate"], required=True)
    parser.add_argument("--config", default="configs/applicability_pilot.yaml")
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--query-manifest", help="separate target manifest; absent = source cross-fit")
    parser.add_argument("--references", required=True)
    parser.add_argument("--native-losses", help="source-only ID -> scope_key -> measured [0,1] error")
    parser.add_argument("--memory", help="frozen memory.json for evaluate")
    parser.add_argument("--output", default="runs/applicability-pilot")
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--model-path")
    parser.add_argument("--llava-source")
    parser.add_argument("--vision-tower-path")
    parser.add_argument("--limit", type=int, default=0, help="0 = all; fixed ID order, not answer-selected")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("limit cannot be negative")
    if args.stage == "collect" and (args.query_manifest or args.memory):
        parser.error("collect accepts source only, no query or old memory")
    if args.stage == "evaluate" and (not args.memory or args.native_losses):
        parser.error("evaluate requires frozen --memory; native losses are collect-only")
    config = load_experiment_yaml(args.config)
    spec = resolve_generalist_spec(config["generalist"])
    for name, field in (("model_path", "checkpoint_path"), ("llava_source", "source_path"),
                        ("vision_tower_path", "vision_tower_path")):
        if getattr(args, name):
            spec[field] = getattr(args, name)
    spec["deterministic_image_padding"] = True
    options = config["applicability"]
    admission_config = ApplicabilityConfig(**options.get("admission", {}))
    decoder = ValueGenerationConfig(**options.get("generation", {}))
    sources = read_manifest(args.source_manifest, "source")
    queries = read_manifest(args.query_manifest, "target") if args.query_manifest else sources
    if args.query_manifest:
        audit_open_split(sources, queries)
    for rows in (sources, queries):
        if len({r["id"] for r in rows}) != len(rows):
            raise ValueError("duplicate IDs")
        if len({r["group_id"] for r in rows}) != len(rows):
            raise ValueError("one question per independent patient group required")
        if len({r["image_sha256"] for r in rows}) != len(rows):
            raise ValueError("duplicate RGB pixels")
        for row in rows:
            if pixel_digest(row["image"]) != row["image_sha256"]:
                raise ValueError("image fingerprint changed")
    if not admission_config.allow_proxy and any(r["domain_kind"] != "independent" for r in sources):
        raise ValueError("source domains must be independent; proxy pilot needs explicit allow_proxy")
    specs, excluded = _filter_optional_experts(config["experts"], args.artifacts)
    selected_tools = options.get("experts")
    if selected_tools:
        missing = set(selected_tools)-set(specs)
        if missing:
            raise ValueError(f"requested applicability experts unavailable: {sorted(missing)}")
        specs = {k: v for k, v in specs.items() if k in selected_tools}
    if not specs:
        raise ValueError("no experts configured")
    active = sorted(sources if args.stage == "collect" else queries, key=lambda r: r["id"])
    if args.limit:
        active = active[:args.limit]
    if args.check_only:
        print(json.dumps({"stage": args.stage, "cases": len(active), "tools": list(specs),
              "domains": sorted({r["domain"] for r in sources}), "config": asdict(admission_config),
              "note": "metadata check only; no native embedding or medical inference tested"}, indent=2))
        return
    provenance = {"experts": {k: model_provenance(v, args.artifacts) for k, v in specs.items()},
                  "expert_settings": specs, "generalist": generalist_provenance(spec, args.artifacts),
                  "decoder": asdict(decoder), "suffix": config.get("prompt_suffix", ""),
                  "routing": config.get("routing", {}), "code": code_identity()}
    references = read_json(args.references)
    source_refs = {r["id"]: references[r["id"]] for r in sources}
    native = read_json(args.native_losses) if args.native_losses else None
    if native is not None and set(native)-{r["id"] for r in sources}:
        raise ValueError("native losses contain non-source IDs")
    scorer = InterventionScorer(options.get("quality", {"name": "token_f1"}))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    probe = None

    def ensure_probe():
        nonlocal probe
        if probe is None:
            probe = load_generalist(spec, args.artifacts)
        return probe

    sources, source_routes = _route_records(sources, provenance["routing"], ensure_probe, provenance, out)
    if args.query_manifest:
        queries, query_routes = _route_records(queries, provenance["routing"], ensure_probe, provenance, out)
    else:
        queries, query_routes = sources, source_routes
    atomic_json(out / "routing.json", {"source": source_routes, "query": query_routes})
    active = sorted(sources if args.stage == "collect" else queries, key=lambda r: r["id"])
    if args.limit:
        active = active[:args.limit]
    pool = CapabilityPool(specs, args.artifacts, source_records=sources, source_references=source_refs)

    def runtime(row):
        prompt = row["question"]+"\n"+provenance["suffix"]
        session = NativeSession(ensure_probe(), row["image"], prompt, row["question"], decoder)
        return CapabilityRuntime(session, pool, row, specs, decoder)

    try:
        if args.stage == "collect":
            records, collection = [], []
            for row in active:
                identity = fingerprint({"provenance": provenance, "row": row,
                    "source_index": sources, "source_refs": source_refs,
                    "quality": scorer.config, "native": native})
                path = out / "source-cache" / f"{identity}.json"
                if path.exists():
                    case = read_json(path)
                    print(f"Source cache hit: {row['id']}", flush=True)
                else:
                    pool.reset_case()
                    run = runtime(row)
                    baseline = run.run("generalist")
                    entries, failures, outputs = [], [], {}
                    for descriptor in run.descriptors(NativeState()):
                        key = action_key(row, descriptor)
                        try:
                            feature = pool.domain_embedding(descriptor["expert"], row["image"]).tolist()
                        except (ValueError, NotImplementedError) as exc:
                            failures.append({"scope_key": key, "reason": "unsupported_native_embedding",
                                             "detail": str(exc)})
                            continue
                        # Force each eligible tool exactly once, independent of controller selection.
                        state, event = run.execute(NativeState(), descriptor)
                        if "runtime_error" in str(event.get("reason", "")):
                            raise RuntimeError(f"source tool failure: {event}")
                        generated = run.complete(state)
                        before = scorer(row, baseline, source_refs[row["id"]])
                        after = scorer(row, generated, source_refs[row["id"]])
                        loss = native[row["id"]][key] if native is not None else max(0., before-after)
                        entries.append({"id": row["id"], "role": "source", "scope_key": key,
                            "domain": row["domain"], "domain_kind": row["domain_kind"],
                            "group_id": row["group_id"], "image_sha256": row["image_sha256"],
                            "feature": feature, "loss": loss, "base_quality": before,
                            "tool_quality": after, "gain": after-before})
                        outputs[key] = {"output": generated, "event": event}
                    case = {"id": row["id"], "baseline": baseline, "outputs": outputs,
                            "records": entries, "failures": failures}
                    path.parent.mkdir(parents=True, exist_ok=True)
                    atomic_json(path, case)
                    print(f"Source {row['id']}: {len(entries)} measured scopes", flush=True)
                records.extend(case["records"])
                collection.append(case)
            metric = ({"name": "native_task_loss", "file_sha256": hashlib.sha256(
                Path(args.native_losses).read_bytes()).hexdigest()} if native is not None else
                {"name": "paired_generation_harm", "quality": scorer.config,
                 "warning": "bridge-specific answer harm, NOT expert native task error"})
            # Preserve diagnostics even if no expert has usable native support.
            atomic_json(out / "source-interventions.json", collection)
            atomic_json(out / "source-summary.json", {
                "cases": len(collection), "records": len(records), "metric": metric,
                "target_generations": 0,
                "failures": [failure for case in collection for failure in case["failures"]],
            })
            memory = build_memory(records, admission_config, provenance, metric)
            memory["source_manifest"] = sources
            memory["source_references_fingerprint"] = fingerprint(source_refs)
            atomic_json(out / "memory.json", memory)
            print(f"Saved {out / 'memory.json'}; target generations=0", flush=True)
        else:
            memory = read_json(args.memory)
            if memory["provenance"] != provenance or memory["config"] != asdict(admission_config):
                raise ValueError("model/code/interface/config changed: recollect source memory")
            if memory["source_manifest"] != sources or memory["source_references_fingerprint"] != fingerprint(source_refs):
                raise ValueError("source/retrieval bank changed since calibration")
            if args.query_manifest and any(r["domain"] in {s["domain"] for s in sources} for r in queries):
                raise ValueError("target source-domain names overlap; not a held-out-source experiment")
            modes = ["generalist", "agent", "global", "distance", "local", "robust"]
            predictions = {mode: {} for mode in modes}
            memory_key = fingerprint(memory)
            for row in active:
                for mode in modes:
                    identity = fingerprint({"memory": memory_key, "row": row, "mode": mode,
                        "source_evaluation": not bool(args.query_manifest)})
                    path = out / "prediction-cache" / f"{identity}.json"
                    if path.exists():
                        result = read_json(path)
                    else:
                        pool.reset_case()
                        run = runtime(row)
                        gate = None if mode in {"generalist", "agent"} else ApplicabilityGate(
                            memory, pool, mode=mode, exclude_domain=not bool(args.query_manifest))
                        result = run.run("generalist" if mode == "generalist" else "agent",
                                         applicability=gate)
                        path.parent.mkdir(parents=True, exist_ok=True)
                        atomic_json(path, result)
                    predictions[mode][row["id"]] = result
                    print(f"{row['id']} {mode}: calls={result['expert_calls']}", flush=True)
            # Target reference values are used here only, after all model inference.
            results = {mode: capability_summary(outputs, predictions["generalist"], references, active)
                       for mode, outputs in predictions.items()}
            report = {"results": results, "memory_fingerprint": memory_key,
                "source_crossfit": not bool(args.query_manifest), "excluded_experts": excluded,
                "real_source_domains": memory["real_source_domains"], "metric": memory["metric"],
                "real_query_domains": all(r["domain_kind"] == "independent" for r in active),
                "domain_generalization_verified": False,
                "limitations": ["No guarantee under arbitrary unseen shift.",
                    "Native feature extraction has expert model loading/forward cost.",
                    "Initial single-tool calibration is not conditional multi-tool benefit.",
                    ("Source cross-fit excludes query domain from neighbors and residuals, but does not "
                     "isolate the entire upstream retrieval/model pipeline."),
                    "Token-F1 is not clinical factuality; zero calls is not demonstrated DG benefit."]}
            atomic_json(out / "result.json", report)
            atomic_json(out / "predictions.json", predictions)
            lines = ["# Input-aware native collaboration", "", "| Method | F1 | Gain | Calls/case |",
                     "| --- | ---: | ---: | ---: |"]
            for mode, value in results.items():
                lines.append(f"| {mode} | {value['token_f1']:.4f} | {value['paired_f1_gain']:+.4f} | "
                             f"{value['mean_expert_calls']:.2f} |")
            lines += ["", *report["limitations"]]
            (out / "result.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    finally:
        pool.clear()


if __name__ == "__main__":
    main()
