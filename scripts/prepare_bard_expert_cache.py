"""Prepare BARD native expert caches one model at a time.

This is a memory/throughput stage, not an evaluation. It never loads references
or target answers. The output is accepted by matched_evaluation --reuse-expert-run.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

from merit_feddg.bard_protocol import select_bard_descriptors
from merit_feddg.capabilities import CapabilityRequest, tool_descriptors, validate_result
from merit_feddg.capability_runtime import ValueGenerationConfig
from merit_feddg.capability_study import _filter_optional_experts, _route_records
from merit_feddg.evidence_need import evidence_need
from merit_feddg.generalist_factory import (
    generalist_provenance,
    load_generalist,
    resolve_generalist_spec,
)
from merit_feddg.io import load_experiment_yaml
from merit_feddg.matched_evaluation import generation_prompt, load_cached, load_manifest
from merit_feddg.open_study import atomic_json, fingerprint, model_provenance


def release_accelerator():
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def routed_rows(original, config, root, identity, artifacts, routing_json=None):
    rows = [
        {
            "id": row["id"],
            "image": row["image"],
            "question": row["question"],
            "modality": "mixed",
            "capability": "classification",
            "task": row.get("task", "open_vqa"),
            "domain": "official-test",
            "domain_kind": "official_dataset_split",
            "role": "target",
            "group_id": row["image_sha256"],
            "image_sha256": row["image_sha256"],
        }
        for row in original
    ]
    if routing_json:
        routes = json.loads(Path(routing_json).read_text(encoding="utf-8"))
        if set(routes) != {row["id"] for row in rows}:
            raise ValueError("reused routing does not cover the exact manifest")
        routed = []
        for row in rows:
            modality = routes[row["id"]].get("modality")
            if not isinstance(modality, str) or not modality:
                raise ValueError("reused routing has an invalid modality")
            routed.append({**row, "modality": modality})
        return routed, routes, {"source": str(Path(routing_json).resolve())}

    holder = [load_generalist(config["generalist"], artifacts)]
    try:
        routed, routes = _route_records(
            rows,
            config.get("routing", {}),
            lambda: holder[0],
            {"identity": identity},
            root,
        )
    finally:
        holder[0] = None
        release_accelerator()
    return routed, routes, {"source": "fresh_image_only_generalist_routing"}


def make_request(row, descriptor, config):
    need = evidence_need(row["question"], descriptor)
    return CapabilityRequest(
        sample_id=row["id"],
        image=row["image"],
        question=row["question"],
        modality=row["modality"],
        task=row["task"],
        domain=row["domain"],
        group_id=row["group_id"],
        capability=descriptor["capability"],
        scope=descriptor["scope"],
        query=need.query if config.request_style == "need" else row["question"],
        generated_prefix="",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", default="configs/matched_bard.yaml")
    parser.add_argument("--output", default="runs/bard-expert-cache")
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--routing-json")
    parser.add_argument("--coverage-only", action="store_true")
    parser.add_argument("--skip-expert", action="append", default=[], help="Prepare disjoint native stages; incomplete caches remain marked incomplete")
    args = parser.parse_args()

    config = load_experiment_yaml(args.config)
    if config.get("prompt_contract") not in {"anchor-ce-v1", "anchor-task-v1"}:
        raise ValueError("BARD cache preparation requires the frozen ANCHOR prompt contract")
    config["generalist"] = resolve_generalist_spec(config["generalist"])
    config["generalist"]["deterministic_image_padding"] = True
    generation = ValueGenerationConfig(**config["capability_value"]["generation"])
    if (
        generation.evidence_style != "semantic"
        or not generation.token_budgeted_evidence
        or generation.request_scope_check
    ):
        raise ValueError(
            "expert-major cache preparation currently supports the frozen BARD "
            "semantic protocol with request_scope_check disabled"
        )

    original = load_manifest(args.manifest)
    # Force prompt validation now; prompts themselves are not needed by native tools.
    for row in original:
        generation_prompt(row, config)

    specs, excluded = _filter_optional_experts(config["experts"], args.artifacts)
    specs.pop("source_cases", None)
    if any(spec.get("minimum_evidence_tokens") is not None for spec in specs.values()):
        raise ValueError("cache preparation does not reproduce minimum-token descriptor filtering")

    generalist_id = generalist_provenance(config["generalist"], args.artifacts)
    expert_ids = {
        name: model_provenance(spec, args.artifacts)
        for name, spec in specs.items()
    }
    image_hashes = {
        path: file_sha256(path)
        for path in sorted({row["image"] for row in original})
    }
    identity = fingerprint(
        {
            "schema": "bard-expert-cache-v1",
            "config": config,
            "manifest": original,
            "generalist": generalist_id,
            "experts": expert_ids,
            "images": image_hashes,
        }
    )
    root = Path(args.output) / identity
    root.mkdir(parents=True, exist_ok=True)

    rows, routes, routing_audit = routed_rows(
        original,
        config,
        root,
        identity,
        args.artifacts,
        args.routing_json,
    )
    atomic_json(root / "routing.json", routes)

    # Freeze the answer-blind acquisition schedule before loading any expert.
    schedule = {}
    for row in rows:
        candidates = [
            descriptor
            for descriptor in tool_descriptors(specs, row)
            if not descriptor["requires_region"]
        ]
        schedule[row["id"]] = select_bard_descriptors(
            candidates, specs, generation.max_expert_calls
        )
    atomic_json(root / "schedule.json", schedule)

    # Coverage is answer-blind and known before any specialist model is loaded.
    coverage_cases = []
    histogram = {"0": 0, "1": 0, "2": 0, "3+": 0}
    modality_counts = {}
    for row in rows:
        descriptors = schedule[row["id"]]
        groups = []
        visual_groups = []
        retrieval_groups = []
        for descriptor in descriptors:
            expert = descriptor["expert"]
            group = str(specs[expert].get("fault_group", expert)).strip()
            if group not in groups:
                groups.append(group)
            target = retrieval_groups if descriptor["capability"] == "retrieval" else visual_groups
            if group not in target:
                target.append(group)
        key = str(len(groups)) if len(groups) < 3 else "3+"
        histogram[key] += 1
        modality = row["modality"]
        stats = modality_counts.setdefault(
            modality,
            {"n": 0, "0": 0, "1": 0, "2": 0, "3+": 0, "retrieval_cases": 0},
        )
        stats["n"] += 1
        stats[key] += 1
        stats["retrieval_cases"] += int(bool(retrieval_groups))
        coverage_cases.append(
            {
                "id": row["id"],
                "modality": modality,
                "fault_groups": groups,
                "fault_group_count": len(groups),
                "visual_fault_groups": visual_groups,
                "retrieval_fault_groups": retrieval_groups,
                "adaptive_mode": (
                    "generalist"
                    if len(groups) <= 1
                    else "pair_unanimous"
                    if len(groups) == 2
                    else "robust"
                ),
            }
        )
    coverage = {
        "schema": "bard-scheduled-coverage-v1",
        "n": len(rows),
        "fault_group_histogram": histogram,
        "by_modality": modality_counts,
        "cases": coverage_cases,
        "answers_loaded": False,
        "references_loaded": False,
        "expert_outputs_loaded": False,
    }
    atomic_json(root / "coverage.json", coverage)
    print(
        "scheduled BARD fault-group coverage: "
        + " ".join(f"{key}={value}" for key, value in histogram.items()),
        flush=True,
    )

    if args.coverage_only:
        atomic_json(
            root / "coverage-protocol.json",
            {
                "schema": "bard-coverage-only-v1",
                "identity": identity,
                "n": len(original),
                "config": config,
                "expert_provenance": expert_ids,
                "excluded": excluded,
                "routing": routing_audit,
                "coverage": {
                    "fault_group_histogram": coverage["fault_group_histogram"],
                    "by_modality": coverage["by_modality"],
                },
                "expert_inference_executed": False,
                "answers_loaded": False,
                "references_loaded": False,
            },
        )
        print(root)
        return

    from merit_feddg.capability_experts import CapabilityPool

    summary = {}
    for expert in specs:
        if expert in args.skip_expert:
            continue
        selected = [
            (row, descriptor)
            for row in rows
            for descriptor in schedule[row["id"]]
            if descriptor["expert"] == expert
        ]
        if not selected:
            continue
        pool = CapabilityPool(specs, args.artifacts, source_records=())
        started = perf_counter()
        live = cached = 0
        try:
            for row, descriptor in selected:
                request = make_request(row, descriptor, generation)
                key = fingerprint(["infer", expert, asdict(request)])
                directory = root / "expert-cache" / fingerprint(row["id"])
                path = directory / f"{key}.json"
                value = load_cached(path, identity)
                if value is not None:
                    cached += 1
                    continue
                result = validate_result(pool.infer(expert, request), expert, request)
                value = asdict(result)
                directory.mkdir(parents=True, exist_ok=True)
                atomic_json(path, {"identity": identity, "output": value})
                live += 1
        finally:
            pool.clear()
            del pool
            release_accelerator()
        summary[expert] = {
            "scheduled_requests": len(selected),
            "live_requests": live,
            "cache_hits": cached,
            "seconds": perf_counter() - started,
            "model_released_after_stage": True,
        }
        atomic_json(root / "progress.json", summary)
        print(
            f"{expert}: requests={len(selected)} live={live} cached={cached}",
            flush=True,
        )

    missing_requests = []
    for row in rows:
        for descriptor in schedule[row["id"]]:
            request = make_request(row, descriptor, generation)
            key = fingerprint(["infer", descriptor["expert"], asdict(request)])
            path = root / "expert-cache" / fingerprint(row["id"]) / f"{key}.json"
            if load_cached(path, identity) is None:
                missing_requests.append({"id": row["id"], "expert": descriptor["expert"], "key": key})
    protocol = {
        "schema": "bard-expert-cache-v1",
        "identity": identity,
        "n": len(original),
        "shards": 1,
        "shards_complete": not missing_requests,
        "missing_requests": missing_requests,
        "cache_only": True,
        "config": config,
        "expert_provenance": expert_ids,
        "excluded": excluded,
        "routing": routing_audit,
        "summary": summary,
        "coverage": {
            "fault_group_histogram": coverage["fault_group_histogram"],
            "by_modality": coverage["by_modality"],
        },
        "references_loaded": False,
        "answers_loaded": False,
        "expert_major_execution": True,
        "simultaneous_expert_models_required": False,
    }
    atomic_json(root / "protocol.json", protocol)
    print(root)


if __name__ == "__main__":
    main()
