"""Real generation evaluation of heterogeneous capability acquisition.

Inference manifests never contain reference answers. Source references have two
explicit uses: source-only retrieval and after-generation utility calibration.
Neither target references nor target-derived concept vocabularies reach a model.
"""

from __future__ import annotations

import gc
import json
from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter

import numpy as np

from .capabilities import CAPABILITIES, scoped_key, tool_descriptors
from .contribution import answer_metrics, qualify_contribution
from .extract import _extraction_runtime_provenance, _local_or_remote
from .io import load_yaml
from .open_data import audit_open_split, read_manifest
from .open_study import (
    atomic_json,
    fingerprint,
    hardware_provenance,
    inference_identity,
    model_provenance,
)


def scope_qualification(rows, *, expected_domains=None, attempts=None, **options):
    """A negative lower margin is insufficient evidence, not proof of harm."""
    card = qualify_contribution(rows, **options)
    expected = sorted(set(expected_domains or card["domains"]))
    missing = sorted(set(expected) - set(card["domains"]))
    if missing:
        card.update(qualified=False, support_sufficient=False)
    native_support = Counter(
        attempt["domain"]
        for attempt in attempts or []
        if attempt.get("intervention_status") == "native_evidence_adopted"
    )
    # Keep every successful execution (including empty observations) in the
    # utility estimate, but require actual nonempty evidence support separately.
    # Dropping empty results from the mean would condition on a success outcome.
    if attempts is not None and any(
        native_support[domain] < options.get("min_per_domain", 8) for domain in expected
    ):
        card.update(qualified=False, support_sufficient=False)
    attempt_counts = Counter(attempt.get("intervention_status", "unknown") for attempt in attempts or [])
    if not rows and attempt_counts.get("runtime_error", 0):
        status = "runtime_error"
    elif not card["support_sufficient"]:
        status = "insufficient_support"
    elif card["qualified"]:
        status = "qualified"
    elif any(d["mean_gain"] < 0 for d in card["domains"].values()):
        status = "observed_negative_gain"
    else:
        status = "unproven_gain"
    return {
        **card,
        "expected_domains": expected,
        "missing_intervention_domains": missing,
        "status": status,
        "attempt_status_counts": dict(attempt_counts),
        "native_evidence_support_by_domain": dict(native_support),
        "unit": "full_answer_utility_of_forced_single_expert_scope_evidence",
        "policy_calibrated": False,
        "warning": "Lexical utility is not clinical factuality; margins are heuristic.",
    }


def capability_summary(outputs, baseline, references, rows):
    """Paired real-output metrics, with independent groups audited at input."""
    if not rows:
        raise ValueError("cannot evaluate an empty target")
    ids = [row["id"] for row in rows]
    values = [answer_metrics(outputs[i]["text"], references[i]) for i in ids]
    controls = [answer_metrics(baseline[i]["text"], references[i]) for i in ids]
    delta = np.asarray([v["token_f1"] - c["token_f1"] for v, c in zip(values, controls)])
    domains = defaultdict(list)
    for index, row in enumerate(rows):
        domains[row["domain"]].append(index)
    rng = np.random.default_rng(42)
    boot = np.zeros(2000)
    for indices in domains.values():
        boot += delta[rng.choice(indices, size=(2000, len(indices)), replace=True)].sum(axis=1)
    boot /= len(rows)
    domain_f1 = {
        domain: float(np.mean([values[i]["token_f1"] for i in indices]))
        for domain, indices in domains.items()
    }
    calls = [outputs[i]["expert_calls"] for i in ids]
    per_expert, per_capability, none_reasons = Counter(), Counter(), Counter()
    sequences = Counter()
    multi_expert = multi_capability = 0
    for sample_id in ids:
        sequence = []
        for event in outputs[sample_id].get("trace", []):
            if event.get("event") == "tool" and "request" in event:
                per_expert[event["expert"]] += 1
                per_capability[event["request"]["capability"]] += 1
                sequence.append(
                    (event["expert"], event["request"]["capability"], event["request"]["scope"])
                )
            if str(event.get("reason", "")).startswith("NONE"):
                none_reasons[event["reason"]] += 1
            elif event.get("event") == "controller" and event.get("action") == {
                "action": "continue"
            }:
                none_reasons["controller_continue"] += 1
        sequences[tuple(sequence)] += 1
        multi_expert += len({event[0] for event in sequence}) > 1
        multi_capability += len({event[1] for event in sequence}) > 1
    controller_tokens = [outputs[i].get("controller_output_tokens") for i in ids]
    controller_events = [
        event
        for sample_id in ids
        for event in outputs[sample_id].get("trace", [])
        if event.get("event") == "controller"
    ]
    invalid_actions = sum(event.get("reason") == "NONE:invalid_action" for event in controller_events)
    tool_runtime_errors = sum(
        "runtime_error" in str(event.get("reason", ""))
        for sample_id in ids
        for event in outputs[sample_id].get("trace", [])
        if event.get("event") == "tool"
    )
    answer_tokens = [outputs[i].get("token_ids") for i in ids]
    return {
        "n": len(ids),
        "control": "generalist",
        "exact_match": float(np.mean([v["exact_match"] for v in values])),
        "token_f1": float(np.mean([v["token_f1"] for v in values])),
        "paired_f1_gain": float(delta.mean()),
        "paired_f1_gain_bootstrap95": np.quantile(boot, [0.025, 0.975]).tolist(),
        "f1_improved": int((delta > 0).sum()),
        "f1_harmed": int((delta < 0).sum()),
        "lexical_improved": int((delta > 0).sum()),
        "lexical_decreased": int((delta < 0).sum()),
        "same_text_as_control": float(
            np.mean([outputs[i]["text"] == baseline[i]["text"] for i in ids])
        ),
        "mean_expert_calls": float(np.mean(calls)),
        "expert_call_fraction": float(np.mean(np.asarray(calls) > 0)),
        "mean_controller_calls": float(
            np.mean([outputs[i].get("controller_calls", 0) for i in ids])
        ),
        "mean_controller_tokens": float(np.mean(controller_tokens))
        if all(value is not None for value in controller_tokens)
        else None,
        "invalid_controller_actions": invalid_actions,
        "invalid_controller_action_fraction": invalid_actions / len(controller_events)
        if controller_events
        else None,
        "tool_runtime_errors": tool_runtime_errors,
        "mean_answer_tokens": float(np.mean([len(tokens) for tokens in answer_tokens]))
        if all(tokens is not None for tokens in answer_tokens)
        else None,
        "mean_cached_requests": float(np.mean([outputs[i].get("cache_hits", 0) for i in ids])),
        "mean_adopted_evidence": float(
            np.mean(
                [
                    outputs[i].get("adopted_evidence_count", len(outputs[i].get("evidence", [])))
                    for i in ids
                ]
            )
        ),
        "tool_calls_by_expert": dict(per_expert),
        "tool_calls_by_capability": dict(per_capability),
        "expert_sequences": [
            {
                "sequence": [
                    {"expert": expert, "capability": capability, "scope": scope}
                    for expert, capability, scope in sequence
                ],
                "n": count,
            }
            for sequence, count in sorted(sequences.items(), key=lambda item: (-item[1], item[0]))
        ],
        "multi_expert_case_fraction": multi_expert / len(ids),
        "multi_capability_case_fraction": multi_capability / len(ids),
        "none_reason_counts": dict(none_reasons),
        "mean_seconds": float(np.mean([outputs[i]["seconds"] for i in ids])),
        "mean_routing_seconds": float(np.mean([outputs[i].get("routing_seconds", 0) for i in ids])),
        "mean_generation_seconds": float(
            np.mean([outputs[i].get("generation_seconds", outputs[i]["seconds"]) for i in ids])
        ),
        "p95_seconds": float(np.quantile([outputs[i]["seconds"] for i in ids], 0.95)),
        "peak_allocated_gib": max(outputs[i].get("peak_allocated_gib", 0) for i in ids),
        "domain_f1": domain_f1,
        "worst_domain_f1": min(domain_f1.values()),
        "metric_warning": "EM/token-F1 are lexical measures, NOT medical hallucination rates.",
    }


def _write_annotations(root, predictions, rows):
    lookup = {row["id"]: row for row in rows}
    blinded, key = [], []
    for method, outputs in predictions.items():
        for sample_id, result in outputs.items():
            annotation_id = fingerprint({"case": sample_id, "method": method})[:20]
            row = lookup[sample_id]
            blinded.append(
                {
                    "annotation_id": annotation_id,
                    "image": row["image"],
                    "question": row["question"],
                    "response": result["text"],
                    "supported_claims": None,
                    "unsupported_claims": None,
                    "contradicted_claims": None,
                    "clinically_important_omissions": None,
                }
            )
            key.append({"annotation_id": annotation_id, "sample_id": sample_id, "method": method})
    atomic_json(root / "annotation-blinded.json", sorted(blinded, key=lambda r: r["annotation_id"]))
    atomic_json(root / "annotation-key-private.json", key)


def _compatible_pairs(specs, row):
    """The old manifest's single capability must not disable other abilities."""
    for descriptor in tool_descriptors(specs, row):
        yield descriptor["expert"], descriptor["capability"], descriptor["scope"]


def _filter_optional_experts(specs, artifacts):
    """Only an explicitly optional, missing local checkpoint is omitted.

    Present-but-invalid checkpoints, missing dependencies and required tools are
    not swallowed. Qualification never influences model availability filtering.
    """
    active, excluded = {}, {}
    for name, spec in specs.items():
        optional = spec.get("optional", False)
        if not isinstance(optional, bool):
            raise TypeError(f"expert {name}: optional must be an explicit boolean")
        if optional:
            expected = spec.get("checkpoint_path") or _local_or_remote(spec["id"], artifacts)
            if not spec.get("checkpoint_path") and artifacts and not Path(expected).exists():
                expected = Path(artifacts) / "models" / spec["id"].replace("/", "--")
            path = Path(expected).expanduser().resolve()
            if not path.exists():
                excluded[name] = {
                    "reason": "optional_checkpoint_missing",
                    "expected_checkpoint": str(path),
                    "id": spec["id"],
                    "capabilities": list(spec.get("capabilities", [])),
                    "scope": spec.get("scope"),
                    "download_attempted": False,
                }
                continue
        active[name] = spec
    return active, excluded


def _scope_key(name, row, capability, scope):
    return scoped_key(name, row["modality"], row["task"], capability, scope)


def _route_records(rows, routing, ensure_probe, identity, output):
    """Image-only routing cached without questions, references or domain labels."""
    if not routing.get("enabled", False):
        return rows, {}
    from .capability_routing import infer_image_type

    routed, metadata = [], {}
    for row in rows:
        if not routing.get("override_dataset_modality", False) and row["modality"] not in {
            "mixed", "unknown"
        }:
            routed.append(row)
            continue
        key = fingerprint(
            {"model_runtime": identity, "image_sha256": row["image_sha256"], "routing": routing}
        )
        path = Path(output) / "routing-cache" / f"{key}.json"
        result = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        if result is None or result.get("cache_key") != key:
            result = dict(infer_image_type(ensure_probe(), row))
            result["cache_key"] = key
            if not isinstance(result.get("modality"), str) or not result["modality"].strip():
                raise ValueError("image router must return a nonempty modality")
            atomic_json(path, result)
        metadata[row["id"]] = {**result, "dataset_modality": row["modality"], "role": row["role"]}
        routed.append({**row, "modality": result["modality"]})
    return routed, metadata


def _write_markdown(root, report):
    rows = [
        "# Native capability generation study",
        "",
        "Actual autoregressive outputs; frozen models, source-only scope qualification.",
        "Lexical EM/F1 are **not** medical hallucination/factuality measures.",
        "",
        "| Method | N | EM | Token-F1 | Calls/case | Controller calls/case | Seconds/case |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method, result in report["results"].items():
        rows.append(
            f"| {method} | {result['n']} | {result['exact_match']:.4f} | "
            f"{result['token_f1']:.4f} | {result['mean_expert_calls']:.2f} | "
            f"{result['mean_controller_calls']:.2f} | {result['mean_seconds']:.3f} |"
        )
    rows += [
        "",
        "## Source capability-scope cards",
        "",
        "| Scope key | Status | Robust margin | Missing intervention domains |",
        "| --- | --- | ---: | --- |",
    ]
    for key, card in report["qualification"]["cards"].items():
        rows.append(
            f"| {key.replace('|', '/')} | {card['status']} | {card['robust_gain']:.4f} | "
            f"{', '.join(card['missing_intervention_domains']) or 'none'} |"
        )
    rows += ["", "## Interpretation boundaries", ""]
    rows += [f"- {limitation}" for limitation in report["limitations"]]
    if report.get("excluded_tools"):
        rows += ["", "## Optional tools not available locally", ""]
        rows += [
            f"- {name}: {record['reason']} (no automatic download)."
            for name, record in report["excluded_tools"].items()
        ]
    rows += [
        "",
        "Full generated answers and tool/generation traces: [predictions.json](predictions.json).",
        "Source qualification: [qualification.json](qualification.json).",
        "Use [annotation-blinded.json](annotation-blinded.json) for medical factuality assessment.",
    ]
    (root / "result.md").write_text("\n".join(rows) + "\n", encoding="utf-8")


def run_capability_study(source_path, target_path, references_path, config_path, artifacts, output):
    """Calibrate single-scope interventions, then run real target collaboration.

    Reference-changing target evaluations reuse identical model generations.
    Source reference changes invalidate generations because retrieval can use
    them. Each source retrieval request must exclude its entire source domain.
    """
    from .capability_experts import CapabilityPool
    from .capability_generation import (
        CapabilityConfig,
        generate_capabilities,
    )
    from .generalist_factory import (
        generalist_provenance as backend_provenance,
    )
    from .generalist_factory import (
        load_generalist,
        make_capability_session,
        resolve_generalist_spec,
    )

    source, target = read_manifest(source_path, "source"), read_manifest(target_path, "target")
    audit_open_split(source, target)
    config = load_yaml(config_path)
    config["generalist"] = resolve_generalist_spec(config["generalist"])
    decoder = CapabilityConfig(**config.get("capability_generation", config.get("generation", {})))
    specs = config["experts"]
    if not specs:
        raise ValueError("at least one configured capability expert is required")
    for name, spec in specs.items():
        values = [name, *spec.get("modalities", []), *spec.get("tasks", [])]
        capabilities = spec.get("capabilities", [])
        values += [*capabilities, spec.get("scope", "default")]
        if not capabilities or not spec.get("modalities") or not spec.get("tasks"):
            raise ValueError("expert modalities/tasks/capabilities must be explicit")
        if any(capability not in CAPABILITIES for capability in capabilities):
            raise ValueError("unknown configured capability")
        if any(not isinstance(value, str) or not value or "|" in value for value in values):
            raise ValueError("expert identity/scope fields must be nonempty and contain no '|'")
    specs, excluded_tools = _filter_optional_experts(specs, artifacts)
    for name, record in excluded_tools.items():
        print(f"Skipping optional tool {name}: {record['reason']}", flush=True)
    probe = None

    def ensure_probe():
        nonlocal probe
        if probe is None:
            probe = load_generalist(config["generalist"], artifacts)
        return probe

    runtime, hardware = _extraction_runtime_provenance(), hardware_provenance()
    generalist_provenance = (
        backend_provenance(config["generalist"], artifacts)
        if config["generalist"].get("backend") == "llava_med"
        else model_provenance(config["generalist"], artifacts)
    )
    routing_identity = {
        "generalist": generalist_provenance,
        "generalist_settings": config["generalist"],
        "runtime": runtime,
        "hardware": hardware,
    }
    source, source_routes = _route_records(
        source, config.get("routing", {}), ensure_probe, routing_identity, output
    )
    target, target_routes = _route_records(
        target, config.get("routing", {}), ensure_probe, routing_identity, output
    )
    routes = {**source_routes, **target_routes}
    refs = json.loads(Path(references_path).read_text(encoding="utf-8"))
    for row in source + target:
        answer_metrics("", refs[row["id"]])
    source_refs = {row["id"]: refs[row["id"]] for row in source}
    source_identity = {
        "rows": [inference_identity(row) for row in source],
        "references": source_refs,
        "retrieval_exclusion": "same_domain_same_group_same_image",
    }
    source_key = fingerprint(source_identity)
    provenance = {
        "config": config,
        "runtime": runtime,
        "hardware": hardware,
        "generalist": generalist_provenance,
        "experts": {name: model_provenance(spec, artifacts) for name, spec in specs.items()},
        "excluded_tools": excluded_tools,
        "source_data_key": source_key,
    }
    run_key = fingerprint(provenance)
    evaluation_key = fingerprint(
        {
            "generation": run_key,
            "target": [inference_identity(row) for row in target],
            "target_references": {row["id"]: refs[row["id"]] for row in target},
        }
    )
    root = Path(output) / evaluation_key[:16]
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(root / "provenance.json", provenance)
    atomic_json(root / "routing.json", routes)
    pool = CapabilityPool(specs, artifacts, source_records=source, source_references=source_refs)
    case_index = []

    def generate(row, mode, *, allowed_pairs=None, cards=None):
        nonlocal probe
        key = fingerprint(
            {
                "runtime": run_key,
                "input": inference_identity(row),
                "mode": mode,
                "allowed_pairs": sorted(allowed_pairs) if allowed_pairs is not None else None,
                "cards": cards,
                "retrieval_index": source_key,
            }
        )
        path = Path(output) / "case-cache" / run_key[:16] / f"{key}.json"
        case_index.append(
            {
                "id": row["id"],
                "role": row["role"],
                "method": mode,
                "allowed_pairs": sorted(allowed_pairs) if allowed_pairs is not None else None,
                "file": str(path.resolve()),
            }
        )
        if path.exists():
            cached = json.loads(path.read_text(encoding="utf-8"))
            if cached.get("cache_key") == key and "trace" in cached:
                print(f"Reusing {row['role']} {row['id']} {mode}", flush=True)
                return cached
        print(f"Generating {row['role']} {row['id']} {mode}", flush=True)
        ensure_probe()
        pool.reset_case()
        cuda = probe.torch.cuda
        if cuda.is_available():
            for device in range(cuda.device_count()):
                cuda.synchronize(device)
                cuda.reset_peak_memory_stats(device)
        suffix = config.get(
            "prompt_suffix",
            "Answer concisely from the image. Use expert observations only within their scope.",
        )
        prompt = row["question"] + "\n" + suffix
        started = perf_counter()
        result = generate_capabilities(
            make_capability_session(probe, row["image"], prompt, config=decoder),
            pool,
            row,
            decoder,
            specs,
            cards=cards,
            mode=mode,
            allowed_pairs=allowed_pairs,
        )
        if cuda.is_available():
            for device in range(cuda.device_count()):
                cuda.synchronize(device)
            result["peak_allocated_gib"] = (
                sum(cuda.max_memory_allocated(i) for i in range(cuda.device_count())) / 2**30
            )
        generation_seconds = perf_counter() - started
        routing_seconds = 0.0 if mode == "generalist" else routes.get(row["id"], {}).get("seconds", 0.0)
        result.update(
            {
                "cache_key": key,
                "sample_id": row["id"],
                "method": mode,
                "seconds": generation_seconds + routing_seconds,
                "generation_seconds": generation_seconds,
                "routing_seconds": routing_seconds,
                "routing_cost_accounting": "one image-only route per tool-assisted inference; cached timing reused",
                "source_data_key": source_key,
            }
        )
        atomic_json(path, result)
        return result

    gains = defaultdict(list)
    attempts = defaultdict(list)
    adaptive_audits = defaultdict(list)
    expected_domains = defaultdict(set)
    for row in source:
        baseline = generate(row, "generalist")
        base_f1 = answer_metrics(baseline["text"], refs[row["id"]])["token_f1"]
        for name, capability, scope in _compatible_pairs(specs, row):
            key = _scope_key(name, row, capability, scope)
            expected_domains[key].add(row["domain"])
            gains[key]  # Persist insufficient scope cards rather than silently omit.
            pair = {(name, capability, scope)}
            # Tool utility must not depend on whether the online controller can
            # express a legal action. Force one task-compatible native tool;
            # this qualifies the evidence setup, not the multi-tool policy.
            guided = generate(row, "all_evidence", allowed_pairs=pair)
            adopted = guided.get("adopted_evidence_count", len(guided.get("evidence", [])))
            tool_failed = any(
                "runtime_error" in str(event.get("reason", ""))
                for event in guided.get("trace", [])
            )
            intervention_status = (
                "runtime_error"
                if tool_failed
                else "native_evidence_adopted"
                if adopted > 0
                else "empty_or_unusable_evidence"
                if guided["expert_calls"] > 0
                else "no_executable_request"
            )
            record = {
                "role": "source",
                "id": row["id"],
                "domain": row["domain"],
                "base_f1": base_f1,
                "guided_f1": answer_metrics(guided["text"], refs[row["id"]])["token_f1"],
                "expert_calls": guided["expert_calls"],
                "adopted_evidence": adopted,
                "intervention_status": intervention_status,
                "calibration_mode": "forced_single_tool",
            }
            attempts[key].append(record)
            # Preserve empty returns in the utility estimate. Native-evidence
            # support is counted independently by scope_qualification.
            if guided["expert_calls"] > 0 and not tool_failed:
                gains[key].append(record)
            if config.get("evaluation", {}).get("source_adaptive_audit", False):
                adaptive = generate(row, "adaptive_no_dg", allowed_pairs=pair)
                adaptive_audits[key].append(
                    {
                        "id": row["id"],
                        "domain": row["domain"],
                        "expert_calls": adaptive["expert_calls"],
                        "base_f1": base_f1,
                        "guided_f1": answer_metrics(adaptive["text"], refs[row["id"]])["token_f1"],
                    }
                )
    cards = {
        key: scope_qualification(
            rows,
            expected_domains=expected_domains[key],
            attempts=attempts[key],
            **config.get("qualification", {}),
        )
        for key, rows in gains.items()
    }
    qualification = {
        "cards": cards,
        "source_paired_results": dict(gains),
        "source_attempts_including_none": dict(attempts),
        "source_adaptive_audit": dict(adaptive_audits),
        "provenance": run_key,
        "source_data_key": source_key,
        "domain_kind": sorted({row["domain_kind"] for row in source}),
        "target_labels_used": False,
        "calibration_policy": "forced_all_evidence_with_one_allowed_expert_capability_scope",
        "online_policy_calibrated": False,
        "interaction_warning": (
            "Forced single-scope evidence gains do not validate online tool selection, "
            "later-prefix interventions, or multi-tool composition."
        ),
    }
    atomic_json(root / "qualification.json", qualification)
    pool.clear()
    gc.collect()
    if probe is not None and probe.torch.cuda.is_available():
        probe.torch.cuda.empty_cache()
    predictions = defaultdict(dict)
    evaluation = config.get("evaluation", {})
    modes = evaluation.get("modes", ["generalist", "all_evidence", "adaptive_no_dg", "adaptive_dg"])
    valid_modes = {"generalist", "all_evidence", "adaptive_no_dg", "adaptive_dg"}
    if not isinstance(modes, list) or not modes or any(mode not in valid_modes for mode in modes):
        raise ValueError("evaluation.modes must be a nonempty list of supported generation modes")
    modes = list(dict.fromkeys(["generalist", *modes]))
    single_pairs = sorted(
        {pair for row in target for pair in _compatible_pairs(specs, row)}
        if evaluation.get("single_tools", True)
        else []
    )
    for row in target:
        for mode in modes:
            predictions[mode][row["id"]] = generate(
                row, mode, cards=cards if mode == "adaptive_dg" else None
            )
        for pair in single_pairs:
            label = "single_tool::" + "::".join(pair)
            predictions[label][row["id"]] = generate(row, "all_evidence", allowed_pairs={pair})
    report = {
        "run_dir": str(root.resolve()),
        "method": "on-demand native capability acquisition and evidence-conditioned generation",
        "qualification": qualification,
        "active_tools": sorted(specs),
        "excluded_tools": excluded_tools,
        "results": {
            method: capability_summary(outputs, predictions["generalist"], refs, target)
            for method, outputs in predictions.items()
        },
        "real_domain_metadata": all(row["domain_kind"] == "independent" for row in source + target),
        "routing": {
            "enabled": bool(config.get("routing", {}).get("enabled", False)),
            "source_cases_routed": len(source_routes),
            "target_cases_routed": len(target_routes),
            "predicted_target_modalities": dict(Counter(row["modality"] for row in target)),
            "warning": "Image type is model-inferred, not validated clinical domain metadata.",
            "timing": "Tool-method seconds include each case's measured routing cost; generalist excludes routing.",
        },
        "limitations": [
            "No proven medical hallucination reduction: automatic metrics are lexical EM/F1.",
            "Default PathVQA source groups are hash proxies, not independent hospitals.",
            "Source cards qualify forced single-tool evidence, not the online controller policy.",
            "Single-scope source qualification does not establish joint tool-composition safety.",
            "Cold/warm model and retrieval-index costs are mixed; wall time is measured, not inferred.",
            "Image-only inferred modality is a routing prediction, not hospital/domain ground truth.",
            "No shuffled-evidence control here: fixed-request replay is required to isolate evidence.",
            "all_evidence is bounded by the same configured call budget, not an unlimited oracle.",
        ],
    }
    atomic_json(root / "predictions.json", dict(predictions))
    _write_annotations(root, predictions, target)
    atomic_json(root / "result.json", report)
    _write_markdown(root, report)
    atomic_json(root / "case-index.json", case_index)
    atomic_json(Path(output) / "latest.json", {"run_dir": str(root.resolve())})
    pool.clear()
    return report
