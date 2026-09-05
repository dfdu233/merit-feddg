"""Actual source and target autoregressive runs, not a frozen-score simulation."""

from __future__ import annotations

import gc
import hashlib
import inspect
import json
import platform
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np

from .block_decode import BlockConfig, QwenBlockSession, decode_blocks
from .contribution import answer_metrics, qualify_contribution
from .extract import _extraction_runtime_provenance, _local_or_remote, _snapshot_provenance
from .io import load_yaml, save_json
from .open_data import audit_open_split, read_manifest
from .open_experts import OpenExpertPool


def fingerprint(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    save_json(temporary, payload)
    temporary.replace(path)


def hardware_provenance():
    result = {"platform": platform.platform(), "processor": platform.processor()}
    try:
        import torch
    except ImportError:
        return result
    result["cuda_devices"] = [
        torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
    ]
    return result


def compatible(spec, row):
    return (
        row["modality"] in spec["modalities"]
        and row["capability"] in spec["capabilities"]
        and row["task"] in spec["tasks"]
    )


def card_key(name, row):
    return "|".join((name, row["modality"], row["capability"], row["task"]))


def inference_identity(row):
    # Moving the same verified image does not require repeating expensive inference.
    return {key: value for key, value in row.items() if key != "image"}


def choose_expert(specs, cards, row):
    available = [
        (cards.get(card_key(name, row), {}).get("robust_gain", -1), name)
        for name, spec in specs.items()
        if compatible(spec, row) and cards.get(card_key(name, row), {}).get("qualified", False)
    ]
    return max(available, default=(0, None))[1]


def shuffled_image_donors(rows):
    """Deterministic within-task derangement; no answers and no self-image donors."""
    groups = defaultdict(list)
    for row in rows:
        groups[(row["modality"], row["capability"], row["task"])].append(row)
    donors = {}
    for group in groups.values():
        ordered = sorted(group, key=lambda row: row["image_sha256"])
        if len(ordered) > 1:
            for index, row in enumerate(ordered):
                donors[row["id"]] = ordered[(index + 1) % len(ordered)]
    return donors


def model_provenance(spec, artifacts):
    custom = bool(spec.get("checkpoint_path"))
    path = Path(spec["checkpoint_path"] if custom else _local_or_remote(spec["id"], artifacts))
    if not path.exists() or (not custom and not path.is_dir()):
        raise FileNotFoundError(f"download the model first; local snapshot required: {spec['id']}")
    if custom:
        result = {
            "id": spec["id"],
            "checkpoint_path": str(path.resolve()),
            "snapshot": "custom-local",
        }
    else:
        result = _snapshot_provenance(spec, artifacts)
        snapshot = result.get("snapshot")
        if not isinstance(snapshot, dict) or snapshot.get("revision") != spec.get("revision"):
            raise ValueError(f"missing or revision-mismatched download marker: {spec['id']}")
    files = sorted(path.rglob("*")) if path.is_dir() else [path]
    result["file_stats"] = [
        (
            p.relative_to(path).as_posix() if path.is_dir() else p.name,
            p.stat().st_size,
            p.stat().st_mtime_ns,
        )
        for p in files
        if p.is_file() and ".cache" not in p.relative_to(path.parent).parts
    ]
    if not result["file_stats"]:
        raise ValueError("empty local checkpoint")
    if spec.get("factory"):
        import importlib

        module, attribute = spec["factory"].split(":")
        factory = getattr(importlib.import_module(module), attribute)
        source = inspect.getsourcefile(factory)
        if source is None:
            raise ValueError("plugin factory must expose a source file for provenance")
        result["factory_source_sha256"] = hashlib.sha256(Path(source).read_bytes()).hexdigest()
    return result


def paired_summary(outputs, base, references, rows, control_name="beam_only"):
    ids = [r["id"] for r in rows]
    values = [answer_metrics(outputs[i]["text"], references[i]) for i in ids]
    controls = [answer_metrics(base[i]["text"], references[i]) for i in ids]
    delta = np.asarray([v["token_f1"] - b["token_f1"] for v, b in zip(values, controls)])
    # One patient/image per row audited at input; stratify independent target domains.
    domains = defaultdict(list)
    for i, row in enumerate(rows):
        domains[row["domain"]].append(i)
    rng = np.random.default_rng(42)
    boot = np.zeros(2000)
    for indices in domains.values():
        boot += delta[rng.choice(indices, size=(2000, len(indices)), replace=True)].sum(axis=1)
    boot /= len(delta)
    return {
        "n": len(ids),
        "control": control_name,
        "exact_match": float(np.mean([v["exact_match"] for v in values])),
        "token_f1": float(np.mean([v["token_f1"] for v in values])),
        "paired_f1_gain": float(delta.mean()),
        "paired_f1_gain_bootstrap95": np.quantile(boot, [0.025, 0.975]).tolist(),
        "f1_improved": int((delta > 0).sum()),
        "f1_harmed": int((delta < 0).sum()),
        "same_text_as_control": float(
            np.mean([outputs[i]["text"] == base[i]["text"] for i in ids])
        ),
        "mean_expert_calls": float(np.mean([outputs[i]["expert_calls"] for i in ids])),
        "mean_changed_blocks": float(
            np.mean(
                [sum(t["selected"] != t["base_selected"] for t in outputs[i]["trace"]) for i in ids]
            )
        ),
        "mean_seconds": float(np.mean([outputs[i]["seconds"] for i in ids])),
        "p95_seconds": float(np.quantile([outputs[i]["seconds"] for i in ids], 0.95)),
        "peak_allocated_gib": max(outputs[i].get("peak_allocated_gib", 0) for i in ids),
        "domain_f1": {
            d: float(np.mean([values[i]["token_f1"] for i in idx])) for d, idx in domains.items()
        },
        "metric_warning": "Lexical EM/F1 are NOT medical hallucination or factuality rates.",
    }


def run_open_study(source_path, target_path, references_path, config_path, artifacts, output):
    source, target = read_manifest(source_path, "source"), read_manifest(target_path, "target")
    audit_open_split(source, target)
    config = load_yaml(config_path)
    decoder = BlockConfig(**config["decoding"])
    specs = config["experts"]
    for name, spec in specs.items():
        if "|" in name or not spec.get("tasks") or not spec.get("capabilities"):
            raise ValueError("expert names/tasks/capabilities must be explicit")
        if float(spec.get("temperature", 0.07)) <= 0:
            raise ValueError("expert temperature must be positive")
    refs = json.loads(Path(references_path).read_text(encoding="utf-8"))
    for row in source + target:
        answer_metrics("", refs[row["id"]])  # Validate, never pass references into generation.
    provenance = {
        "config": config,
        "runtime": _extraction_runtime_provenance(),
        "hardware": hardware_provenance(),
        "generalist": model_provenance(config["generalist"], artifacts),
        "experts": {n: model_provenance(s, artifacts) for n, s in specs.items()},
    }
    run_key = fingerprint(provenance)
    evaluation_key = fingerprint(
        {
            "generation": run_key,
            "source": [inference_identity(r) for r in source],
            "target": [inference_identity(r) for r in target],
            "references": {r["id"]: refs[r["id"]] for r in source + target},
        }
    )
    root = Path(output) / evaluation_key[:16]
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(root / "provenance.json", provenance)
    case_index = []
    probe = None
    pool = OpenExpertPool(specs, artifacts)

    def generate(row, method, expert=None, reverse=False, donor=None):
        nonlocal probe
        key = fingerprint(
            {
                "runtime": run_key,
                "input": inference_identity(row),
                "method": method,
                "expert": expert,
                "reverse": reverse,
                "evidence_image": donor["image_sha256"] if donor else row["image_sha256"],
            }
        )
        path = Path(output) / "case-cache" / run_key[:16] / f"{key}.json"
        case_index.append(
            {
                "role": row["role"],
                "id": row["id"],
                "method": method,
                "expert": expert,
                "file": str(path.resolve()),
            }
        )
        if path.exists():
            cached = json.loads(path.read_text(encoding="utf-8"))
            if cached.get("cache_key") == key and "trace" in cached:
                print(f"Reusing {row['role']} {row['id']} {method}", flush=True)
                return cached
        print(
            f"Generating {row['role']} {row['id']} {method} expert={expert or 'NONE'}", flush=True
        )
        if probe is None:
            from .generalist import QwenLayerProbe

            spec = config["generalist"]
            probe = QwenLayerProbe(
                spec.get("checkpoint_path") or _local_or_remote(spec["id"], artifacts),
                layers=[-1],
                dtype=spec.get("dtype", "bfloat16"),
                device_map=spec.get("device_map", "auto"),
            )
        pool.reset_case()
        cuda = probe.torch.cuda
        if cuda.is_available():
            for device in range(cuda.device_count()):
                cuda.synchronize(device)
                cuda.reset_peak_memory_stats(device)
        prompt = row["question"] + "\n" + config["prompt_suffix"]
        resident_before = sorted(pool.models)
        started = perf_counter()
        if method == "greedy":
            text = probe.generate(row["image"], prompt, max_new_tokens=decoder.max_new_tokens)
            result = {
                "text": text,
                "trace": [],
                "expert_calls": 0,
                "seconds": perf_counter() - started,
            }
        else:
            result = decode_blocks(
                QwenBlockSession(probe, row["image"], prompt),
                question=row["question"],
                modality=row["modality"],
                capability=row["capability"],
                config=replace(decoder),
                expert_id=expert,
                reverse_scores=reverse,
                evidence=pool.evidence_function(expert, donor["image"] if donor else row["image"])
                if expert
                else None,
            )
        if cuda.is_available():
            for device in range(cuda.device_count()):
                cuda.synchronize(device)
            result["peak_allocated_gib"] = (
                sum(cuda.max_memory_allocated(i) for i in range(cuda.device_count())) / 2**30
            )
        # Include input preparation and any expert loading, consistently across methods.
        result["seconds"] = perf_counter() - started
        result.update(
            {
                "cache_key": key,
                "sample_id": row["id"],
                "method": method,
                "expert_selected": expert,
                "evidence_donor_id": donor["id"] if donor else None,
                "resident_experts_before": resident_before,
                "resident_experts_after": sorted(pool.models),
            }
        )
        atomic_json(path, result)
        return result

    gains = defaultdict(list)
    for row in source:
        baseline = generate(row, "beam_only")
        base_metric = answer_metrics(baseline["text"], refs[row["id"]])["token_f1"]
        for name, spec in specs.items():
            if not compatible(spec, row):
                continue
            guided = generate(row, "expert_ungated", name)
            gains[card_key(name, row)].append(
                {
                    "role": "source",
                    "domain": row["domain"],
                    "id": row["id"],
                    "base_f1": base_metric,
                    "guided_f1": answer_metrics(guided["text"], refs[row["id"]])["token_f1"],
                }
            )
    cards = {
        key: qualify_contribution(rows, **config.get("qualification", {}))
        for key, rows in gains.items()
    }
    qualification = {
        "cards": cards,
        "source_paired_results": dict(gains),
        "provenance": run_key,
        "source_data_key": fingerprint(
            {"rows": source, "references": {r["id"]: refs[r["id"]] for r in source}}
        ),
        "domain_kind": sorted({r["domain_kind"] for r in source}),
        "target_labels_used": False,
    }
    atomic_json(root / "qualification.json", qualification)
    # Calibration loaded experts; remove them before target NONE/lazy-load checks.
    pool.models.clear()
    pool.reset_case()
    gc.collect()
    if probe is not None and probe.torch.cuda.is_available():
        probe.torch.cuda.empty_cache()
    predictions = defaultdict(dict)
    donors = shuffled_image_donors(target)
    for row in target:
        for method in ("greedy", "beam_only"):
            predictions[method][row["id"]] = generate(row, method)
        selected = choose_expert(specs, cards, row)
        predictions["robust"][row["id"]] = generate(row, "robust", selected)
        predictions["reversed_support"][row["id"]] = generate(
            row, "reversed_support", selected, True
        )
        if row["id"] in donors:
            predictions["shuffled_image"][row["id"]] = generate(
                row, "shuffled_image", selected, donor=donors[row["id"]]
            )
        for name, spec in specs.items():
            if compatible(spec, row):
                predictions[f"ungated:{name}"][row["id"]] = generate(row, "expert_ungated", name)
    report = {
        "run_dir": str(root.resolve()),
        "qualification": qualification,
        "method": "source-qualified bounded short-block expert reranking",
        "results": {},
        "limitations": [
            "No proven hallucination reduction: EM/F1 only.",
            "No cross-block KV reuse; cold/warm model costs are mixed and traced.",
            "Reversed support is candidate-score corruption, not shuffled images.",
            "Generalization to unseen domains is a hypothesis, not guaranteed by qualification.",
        ],
        "real_domain_metadata": all(r["domain_kind"] == "independent" for r in source + target),
    }
    for method, outputs in predictions.items():
        subset = [r for r in target if r["id"] in outputs]
        report["results"][method] = paired_summary(outputs, predictions["beam_only"], refs, subset)
    if "shuffled_image" in predictions:
        subset = [r for r in target if r["id"] in predictions["shuffled_image"]]
        report["robust_vs_shuffled_image"] = paired_summary(
            predictions["robust"], predictions["shuffled_image"], refs, subset, "shuffled_image"
        )
    # Human assessment template: no method names, reference answers or numeric claims.
    blinded, key = [], []
    for method, outputs in predictions.items():
        for sample_id, result in outputs.items():
            annotation_id = fingerprint({"case": sample_id, "method": method})[:20]
            row = next(r for r in target if r["id"] == sample_id)
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
    atomic_json(root / "result.json", report)
    atomic_json(root / "case-index.json", case_index)
    atomic_json(Path(output) / "latest.json", {"run_dir": str(root.resolve())})
    return report
