#!/usr/bin/env python3
"""Resume the training-free new method on the complete official VQA-RAD test split."""

from __future__ import annotations

import gc
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from merit_feddg.capability_experts import CapabilityPool
from merit_feddg.capability_runtime import CapabilityRuntime, NativeSession, ValueGenerationConfig
from merit_feddg.capability_study import _filter_optional_experts, _route_records
from merit_feddg.contribution import answer_metrics
from merit_feddg.generalist_factory import load_generalist
from merit_feddg.io import load_experiment_yaml
from merit_feddg.open_study import atomic_json, fingerprint


def main() -> None:
    root = Path("runs/vqarad-official-full-test").resolve()
    manifest = [
        json.loads(line)
        for line in (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    references = json.loads((root / "references.json").read_text(encoding="utf-8"))
    # Use the tracked profile rather than a generated run-local config so this
    # completed evaluation remains reproducible from a fresh checkout.
    config = load_experiment_yaml("configs/uncertainty_source_pilot.yaml")
    config["generalist"]["deterministic_image_padding"] = True
    specs, excluded = _filter_optional_experts(config["experts"], "artifacts")
    specs.pop("source_cases", None)
    generation = dict(config["capability_value"]["generation"])
    # The 4,000-character diagnostic budget overflowed LLaVA-Med when multiple
    # observations were combined. This transport cap changes no medical score,
    # routing threshold, or evidence value; whole packets that do not fit remain
    # omitted rather than silently truncated.
    generation["max_evidence_chars"] = 2600
    decoder = ValueGenerationConfig(**generation)
    probe = None

    def ensure_probe():
        nonlocal probe
        if probe is None:
            probe = load_generalist(config["generalist"], "artifacts")
        return probe

    rows = [
        {
            "id": row["id"],
            "image": row["image"],
            "question": row["question"],
            "modality": "mixed",
            "capability": "classification",
            "task": "open_vqa",
            "domain": "vqarad-official-test",
            "domain_kind": "official_dataset_split",
            "role": "target",
            "group_id": row["image_sha256"],
            "image_sha256": row["image_sha256"],
        }
        for row in manifest
    ]
    method_root = root / "new-method-uncertainty"
    identity = {
        "protocol": "official_vqa_rad_full_test_new_method",
        "config": fingerprint(config),
        "source_retrieval": False,
    }
    rows, routes = _route_records(
        rows, config.get("routing", {}), ensure_probe, identity, method_root
    )
    atomic_json(method_root / "routing.json", routes)
    pool = CapabilityPool(specs, "artifacts", source_records=())
    cache = method_root / "case-cache"
    cache.mkdir(parents=True, exist_ok=True)
    outputs = {}
    started_all = time.perf_counter()
    for index, row in enumerate(rows, 1):
        path = cache / f"{row['id']}.json"
        if path.exists():
            output = json.loads(path.read_text(encoding="utf-8"))
        else:
            pool.reset_case()
            prompt = row["question"] + "\n" + config.get(
                "prompt_suffix", "Answer concisely from the image."
            )
            session = NativeSession(
                ensure_probe(), row["image"], prompt, row["question"], decoder
            )
            engine = CapabilityRuntime(session, pool, row, specs, decoder, None)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            output = engine.run("all_evidence")
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                output["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
            output["generation_seconds"] = time.perf_counter() - started
            atomic_json(path, output)
            del engine, session
            pool.reset_case()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        outputs[row["id"]] = output
        if index == 1 or index % 10 == 0 or index == len(rows):
            print(
                f"official VQA-RAD new method {index}/{len(rows)} {row['id']} "
                f"calls={output['expert_calls']} seconds={output['generation_seconds']:.3f}",
                flush=True,
            )

    def summarize(answer_type=None):
        selected = [
            (row, original)
            for row, original in zip(rows, manifest, strict=True)
            if answer_type is None or original["answer_type"] == answer_type
        ]
        values = [
            answer_metrics(outputs[row["id"]]["text"], references[row["id"]])
            for row, _ in selected
        ]
        return {
            "n": len(values),
            "exact_match": float(np.mean([value["exact_match"] for value in values])),
            "token_f1": float(np.mean([value["token_f1"] for value in values])),
            "empty_answers": sum(not outputs[row["id"]]["text"].strip() for row, _ in selected),
            "mean_seconds": float(
                np.mean(
                    [
                        outputs[row["id"]]["generation_seconds"] + routes[row["id"]]["seconds"]
                        for row, _ in selected
                    ]
                )
            ),
            "mean_expert_calls": float(
                np.mean([outputs[row["id"]]["expert_calls"] for row, _ in selected])
            ),
            "mean_probe_calls": float(
                np.mean([outputs[row["id"]]["probe_model_calls"] for row, _ in selected])
            ),
        }

    report = {
        "protocol": "official_vqa_rad_full_test_new_method_only",
        "dataset": "flaviagiammarino/vqa-rad",
        "method": "uncertainty-preserving all-evidence",
        "custom_source_target_split": False,
        "official_train_used": False,
        "source_retrieval_enabled": False,
        "test_answers_used_for_generation": False,
        "baseline_regenerated": False,
        "config": {**config["capability_value"], "generation": generation},
        "active_experts": list(specs),
        "excluded_tools": excluded,
        "metrics": {
            "overall": summarize(),
            "closed": summarize("closed"),
            "open": summarize("open"),
        },
        "routing_modalities": dict(Counter(row["modality"] for row in rows)),
        "wall_seconds": time.perf_counter() - started_all,
        "peak_allocated_gib": max(
            output.get("peak_allocated_gib", 0) for output in outputs.values()
        ),
        "limitations": [
            "No source split means source-fitted value policies and source retrieval are omitted.",
            "Exact match and token-F1 are lexical metrics, not clinical factuality.",
            "Existing baseline must be joined separately; it was not regenerated.",
        ],
    }
    atomic_json(method_root / "predictions.json", outputs)
    atomic_json(method_root / "result.json", report)
    print(json.dumps(report, indent=2), flush=True)
    pool.clear()


if __name__ == "__main__":
    main()
