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

from .capabilities import CAPABILITIES, scoped_key
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


def scope_qualification(rows, *, expected_domains=None, **options):
    """A negative lower margin is insufficient evidence, not proof of harm."""
    card = qualify_contribution(rows, **options)
    expected = sorted(set(expected_domains or card["domains"]))
    missing = sorted(set(expected) - set(card["domains"]))
    if missing:
        card.update(qualified=False, support_sufficient=False)
    if not card["support_sufficient"]:
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
        "unit": "full_answer_utility_of_one_expert_scope_intervention",
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
    return {
        "n": len(ids),
        "control": "generalist",
        "exact_match": float(np.mean([v["exact_match"] for v in values])),
        "token_f1": float(np.mean([v["token_f1"] for v in values])),
        "paired_f1_gain": float(delta.mean()),
        "paired_f1_gain_bootstrap95": np.quantile(boot, [0.025, 0.975]).tolist(),
        "f1_improved": int((delta > 0).sum()),
        "f1_harmed": int((delta < 0).sum()),
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
    for name, spec in specs.items():
        if row["modality"] not in spec["modalities"] or row["task"] not in spec["tasks"]:
            continue
        for capability in spec["capabilities"]:
            yield name, capability, spec.get("scope", capability)


def _scope_key(name, row, capability, scope):
    return scoped_key(name, row["modality"], row["task"], capability, scope)


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
        QwenCapabilitySession,
        generate_capabilities,
    )

    source, target = read_manifest(source_path, "source"), read_manifest(target_path, "target")
    audit_open_split(source, target)
    config = load_yaml(config_path)
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
        "runtime": _extraction_runtime_provenance(),
        "hardware": hardware_provenance(),
        "generalist": model_provenance(config["generalist"], artifacts),
        "experts": {name: model_provenance(spec, artifacts) for name, spec in specs.items()},
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
    pool = CapabilityPool(specs, artifacts, source_records=source, source_references=source_refs)
    probe = None
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
        if probe is None:
            from .generalist import QwenLayerProbe

            generalist = config["generalist"]
            probe = QwenLayerProbe(
                generalist.get("checkpoint_path") or _local_or_remote(generalist["id"], artifacts),
                layers=[-1],
                dtype=generalist.get("dtype", "bfloat16"),
                device_map=generalist.get("device_map", "auto"),
            )
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
            QwenCapabilitySession(probe, row["image"], prompt),
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
        result.update(
            {
                "cache_key": key,
                "sample_id": row["id"],
                "method": mode,
                "seconds": perf_counter() - started,
                "source_data_key": source_key,
            }
        )
        atomic_json(path, result)
        return result

    gains = defaultdict(list)
    attempts = defaultdict(list)
    expected_domains = defaultdict(set)
    for row in source:
        baseline = generate(row, "generalist")
        base_f1 = answer_metrics(baseline["text"], refs[row["id"]])["token_f1"]
        for name, capability, scope in _compatible_pairs(specs, row):
            key = _scope_key(name, row, capability, scope)
            expected_domains[key].add(row["domain"])
            gains[key]  # Persist insufficient scope cards rather than silently omit.
            guided = generate(row, "adaptive_no_dg", allowed_pairs={(name, capability, scope)})
            record = {
                "role": "source",
                "id": row["id"],
                "domain": row["domain"],
                "base_f1": base_f1,
                "guided_f1": answer_metrics(guided["text"], refs[row["id"]])["token_f1"],
                "expert_calls": guided["expert_calls"],
                "adopted_evidence": len(guided.get("evidence", [])),
            }
            attempts[key].append(record)
            # NONE does not establish the usefulness of this expert's capability.
            if guided["expert_calls"] > 0:
                gains[key].append(record)
    cards = {
        key: scope_qualification(
            rows, expected_domains=expected_domains[key], **config.get("qualification", {})
        )
        for key, rows in gains.items()
    }
    qualification = {
        "cards": cards,
        "source_paired_results": dict(gains),
        "source_attempts_including_none": dict(attempts),
        "provenance": run_key,
        "source_data_key": source_key,
        "domain_kind": sorted({row["domain_kind"] for row in source}),
        "target_labels_used": False,
        "calibration_policy": "adaptive_no_dg_with_one_allowed_expert_capability_scope",
        "interaction_warning": "Single-scope gains do not guarantee positive multi-tool interactions.",
    }
    atomic_json(root / "qualification.json", qualification)
    pool.clear()
    gc.collect()
    if probe is not None and probe.torch.cuda.is_available():
        probe.torch.cuda.empty_cache()
    predictions = defaultdict(dict)
    for row in target:
        for mode in ("generalist", "all_evidence", "adaptive_no_dg", "adaptive_dg"):
            predictions[mode][row["id"]] = generate(
                row, mode, cards=cards if mode == "adaptive_dg" else None
            )
    report = {
        "run_dir": str(root.resolve()),
        "method": "on-demand native capability acquisition and evidence-conditioned generation",
        "qualification": qualification,
        "results": {
            method: capability_summary(outputs, predictions["generalist"], refs, target)
            for method, outputs in predictions.items()
        },
        "real_domain_metadata": all(row["domain_kind"] == "independent" for row in source + target),
        "limitations": [
            "No proven medical hallucination reduction: automatic metrics are lexical EM/F1.",
            "Default PathVQA source groups are hash proxies, not independent hospitals.",
            "Single-scope source qualification does not establish joint tool-composition safety.",
            "Cold/warm model and retrieval-index costs are mixed; wall time is measured, not inferred.",
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
