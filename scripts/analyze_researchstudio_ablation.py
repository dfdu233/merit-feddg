#!/usr/bin/env python3
"""Fail-closed paired analysis of the frozen five-arm ResearchStudio ablation."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

METHODS = ("generalist", "joint_all", "isolated_mean", "isolated_geomedian", "bard")
CONTRASTS = (
    ("joint_all", "generalist"),
    ("isolated_mean", "joint_all"),
    ("isolated_geomedian", "isolated_mean"),
    ("bard", "isolated_geomedian"),
)


def read_gzip_json(path: Path) -> dict:
    with gzip.open(path, "rt") as handle:
        return json.load(handle)


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * p
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--anchor-root", type=Path, required=True)
    parser.add_argument("--receiver", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--executed-code-identity", required=True,
                        help="Frozen runtime code identity from the run ledger, not the analysis worktree")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260924)
    args = parser.parse_args()
    if args.bootstrap_replicates < 1:
        parser.error("bootstrap-replicates must be positive")
    sys.path.insert(0, str(args.anchor_root.resolve()))
    from anchor.corrected_sgta.evaluate_medheval_answers import evaluate_rows
    from anchor.medeval.evaluate_mixed_vqa_table import (
        VERSION as SCORE_VERSION,
        answer_token_recall,
        source_task_group,
    )

    source_text = args.source.read_text()
    source = json.loads(source_text)
    if not isinstance(source, list) or not source:
        raise ValueError("source must be a nonempty authoritative JSON list")
    ids = [str(row["id"]) for row in source]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate source IDs")
    observed = {p.parent.name for p in args.run.glob("*/provenance.json.gz")}
    if observed != set(ids):
        raise ValueError(f"full coverage required: expected={len(ids)} observed={len(observed)} missing={list(set(ids)-observed)[:8]} extra={list(observed-set(ids))[:8]}")

    decoded: dict[str, list[dict]] = {method: [] for method in METHODS}
    provenance: list[dict] = []
    input_hashes: list[str] = []
    delivered_groups: list[list[str]] = []
    frozen_identity = None
    for row in source:
        sample_id = str(row["id"])
        folder = args.run / sample_id
        prov = read_gzip_json(folder / "provenance.json.gz")
        if prov.get("row", {}).get("id") != sample_id or not prov.get("matched_joint_evidence"):
            raise ValueError(f"unmatched protocol or row identity: {sample_id}")
        if prov["row"].get("image_sha256") != row.get("image_sha256"):
            raise ValueError(f"image SHA mismatch: {sample_id}")
        if prov["row"].get("question") != row.get("question"):
            raise ValueError(f"question mismatch: {sample_id}")
        identity = (
            prov.get("manifest_sha256"), prov.get("protocol_sha256"),
            prov.get("cache_identity"),
            json.dumps(prov.get("receiver_config"), sort_keys=True),
        )
        if frozen_identity is None:
            frozen_identity = identity
        elif identity != frozen_identity:
            raise ValueError(f"receiver/manifest/protocol/cache changed mid-run: {sample_id}")
        input_hashes.append(hashlib.sha256(json.dumps({
            "id": sample_id, "question": row["question"],
            "image_sha256": row["image_sha256"], "prompt": prov["prompt"],
        }, sort_keys=True).encode()).hexdigest())
        provenance.append(prov)
        for method in METHODS:
            decoded[method].append(read_gzip_json(folder / f"{method}.json.gz"))
        evidence = [decoded[method][-1].get("evidence", []) for method in METHODS[1:]]
        if any(items != evidence[0] for items in evidence[1:]):
            raise ValueError(f"different delivered native evidence across arms: {sample_id}")
        group_by_ref = {}
        for event in prov.get("acquisition_events", []):
            for item in event.get("native_evidence", []):
                group_by_ref[json.dumps(item, sort_keys=True)] = str(event["fault_group"])
        groups = []
        for item in evidence[0]:
            key = json.dumps(item, sort_keys=True)
            if key not in group_by_ref:
                raise ValueError(f"delivered evidence lacks acquisition provenance: {sample_id}")
            group = group_by_ref[key]
            if group not in groups:
                groups.append(group)
        delivered_groups.append(groups)

    method_details = {}
    for method in METHODS:
        rows = [
            {**row, "question_id": row["id"], "gt_ans": row["answer"],
             "text": output.get("text", "")}
            for row, output in zip(source, decoded[method])
        ]
        method_details[method] = evaluate_rows(rows)["details"]
        if [str(detail["question_id"]) for detail in method_details[method]] != ids:
            raise ValueError(f"scorer changed ID order: {method}")

    per_sample = []
    scores = {method: [] for method in METHODS}
    clusters: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(source):
        group = source_task_group(row)
        cluster = str(row.get("patient_id") or row.get("image_sha256") or row["id"])
        clusters[cluster].append(index)
        record = {
            "dataset": args.dataset, "receiver": args.receiver, "split": "test",
            "sample_id": row["id"], "image_id": row["image_sha256"],
            "cluster_id": cluster, "answer_type": group,
            "input_id": input_hashes[index], "run_id": str(args.run.resolve()),
            "protocol_id": provenance[index].get("protocol_sha256"),
            "evidence_id": provenance[index].get("cache_identity"),
            "cached_native_access_seconds": sum(
                float(event.get("seconds") or 0.0)
                for event in provenance[index].get("acquisition_events", [])
            ),
            "methods": {},
        }
        for method in METHODS:
            output = decoded[method][index]
            detail = method_details[method][index]
            value = float(bool(detail["correct"])) if group == "ce" else answer_token_recall(output.get("text", ""), row["answer"])
            scores[method].append(value)
            record["methods"][method] = {
                "method_protocol_id": hashlib.sha256(json.dumps({
                    "protocol": record["protocol_id"], "method": method,
                    "receiver_config": provenance[index]["receiver_config"],
                    "executed_code_identity": args.executed_code_identity,
                }, sort_keys=True).encode()).hexdigest(),
                "delivered_source_group_ids": [] if method == "generalist" else delivered_groups[index],
                "raw_or_repair": "raw",
                "prediction": output.get("text", ""),
                "score": value, "parsed": detail["prediction"] is not None,
                "finished": bool(output.get("finished")),
                "generated_tokens": len(output.get("token_ids", [])),
                "wall_seconds": output.get("seconds"),
                "delivered_evidence_items": len(output.get("evidence", [])),
                "selected_evidence_items": output.get("selected_evidence_count"),
                "presented_evidence_items": output.get("presented_evidence_count"),
            }
        per_sample.append(record)

    n = len(source)
    def metrics(indices: list[int]) -> dict:
        base = scores["generalist"]
        result = {}
        for method in METHODS:
            values = scores[method]
            result[method] = {
                "score": sum(values[i] for i in indices) / len(indices),
                "rescue": sum(max(values[i] - base[i], 0.0) for i in indices) / len(indices),
                "harm": sum(max(base[i] - values[i], 0.0) for i in indices) / len(indices),
            }
            result[method]["net"] = result[method]["rescue"] - result[method]["harm"]
        return result

    point = metrics(list(range(n)))
    strata_indices = {
        "ce": [i for i, row in enumerate(source) if source_task_group(row) == "ce"],
        "oe": [i for i, row in enumerate(source) if source_task_group(row) == "oe"],
        **{
            f"delivered_groups_{label}": [
                i for i, groups in enumerate(delivered_groups)
                if ("3+" if len(groups) >= 3 else str(len(groups))) == label
            ]
            for label in ("0", "1", "2", "3+")
        },
        **{
            f"source_group:{group}": [
                i for i, groups in enumerate(delivered_groups) if group in groups
            ]
            for group in sorted({group for groups in delivered_groups for group in groups})
        },
    }
    cluster_keys = sorted(clusters)
    rng = random.Random(args.bootstrap_seed)
    distributions = {method: {metric: [] for metric in ("score", "rescue", "harm", "net")} for method in METHODS}
    contrast_distribution = {f"{a}-{b}": [] for a, b in CONTRASTS}
    for _ in range(args.bootstrap_replicates):
        indices = [i for _ in cluster_keys for i in clusters[rng.choice(cluster_keys)]]
        sample = metrics(indices)
        for method in METHODS:
            for metric in distributions[method]:
                distributions[method][metric].append(sample[method][metric])
        for a, b in CONTRASTS:
            contrast_distribution[f"{a}-{b}"].append(sample[a]["score"] - sample[b]["score"])

    mechanistic = CONTRASTS[1:]
    permutation_rng = random.Random(args.bootstrap_seed + 1)
    permutation_p = {}
    for a, b in mechanistic:
        cluster_deltas = [sum(scores[a][i] - scores[b][i] for i in clusters[key]) for key in cluster_keys]
        observed_delta = abs(sum(cluster_deltas))
        extreme = 0
        for _ in range(args.bootstrap_replicates):
            null_delta = abs(sum(delta if permutation_rng.getrandbits(1) else -delta for delta in cluster_deltas))
            extreme += null_delta >= observed_delta - 1e-12
        permutation_p[f"{a}-{b}"] = (extreme + 1) / (args.bootstrap_replicates + 1)
    holm_p = {}
    adjusted_floor = 0.0
    for rank, (key, p) in enumerate(sorted(permutation_p.items(), key=lambda pair: pair[1])):
        adjusted_floor = max(adjusted_floor, min(1.0, (len(permutation_p) - rank) * p))
        holm_p[key] = adjusted_floor

    report = {
        "status": "complete_full_test_descriptive",
        "dataset": args.dataset, "receiver": args.receiver, "n": n,
        "ce_n": sum(source_task_group(row) == "ce" for row in source),
        "oe_n": sum(source_task_group(row) == "oe" for row in source),
        "score_version": SCORE_VERSION,
        "executed_code_identity": args.executed_code_identity,
        "source_sha256": hashlib.sha256(source_text.encode()).hexdigest(),
        "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "evaluator_source_sha256": hashlib.sha256(
            (args.anchor_root / "anchor" / "corrected_sgta" / "evaluate_medheval_answers.py").read_bytes()
        ).hexdigest(),
        "score_definition": "CE strict correct 0/1; OE reference-token recall; sample-weighted mean",
        "raw_or_repair": "raw",
        "manifest_sha256": frozen_identity[0],
        "protocol_sha256": frozen_identity[1],
        "native_cache_identity": frozen_identity[2],
        "cached_native_access_seconds_total": sum(row["cached_native_access_seconds"] for row in per_sample),
        "native_expert_inference_seconds_total": None,
        "matched_evidence_rows": n,
        "selected_vs_presented_mismatch_n": sum(
            decoded["joint_all"][i].get("selected_evidence_count")
            != decoded["joint_all"][i].get("presented_evidence_count")
            for i in range(n)
        ),
        "delivered_group_strata_n": {
            label: sum(("3+" if len(groups) >= 3 else str(len(groups))) == label
                       for groups in delivered_groups)
            for label in ("0", "1", "2", "3+")
        },
        "descriptive_strata": {
            label: {"n": len(indices), "methods": metrics(indices)}
            for label, indices in strata_indices.items() if indices
        },
        "bootstrap": {"unit": "patient_id_else_image_sha256", "clusters": len(clusters),
                      "replicates": args.bootstrap_replicates, "seed": args.bootstrap_seed},
        "mechanistic_test": {"type": "two-sided paired cluster sign-flip Monte Carlo",
                             "replicates": args.bootstrap_replicates,
                             "seed": args.bootstrap_seed + 1,
                             "multiple_testing": "Holm correction over three prespecified mechanistic contrasts"},
        "methods": {
            method: {
                **point[method],
                "ci95": {metric: [percentile(values, .025), percentile(values, .975)]
                         for metric, values in distributions[method].items()},
                "blank_n": sum(not item.get("text", "").strip() for item in decoded[method]),
                "unfinished_n": sum(not item.get("finished") for item in decoded[method]),
                "parsed_n": sum(detail["prediction"] is not None for detail in method_details[method]),
                "receiver_decode_seconds_total": sum(float(item.get("seconds") or 0.0) for item in decoded[method]),
                "generated_tokens_total": sum(len(item.get("token_ids", [])) for item in decoded[method]),
                "exact_text_changed_n": sum(
                    decoded[method][i].get("text", "") != decoded["generalist"][i].get("text", "")
                    for i in range(n)
                ),
                "parsed_decision_changed_n": sum(
                    method_details[method][i]["prediction"]
                    != method_details["generalist"][i]["prediction"]
                    for i in range(n)
                ),
                "ce_wrong_to_right_n": sum(
                    source_task_group(source[i]) == "ce" and scores["generalist"][i] == 0.0 and scores[method][i] == 1.0
                    for i in range(n)
                ),
                "ce_right_to_wrong_n": sum(
                    source_task_group(source[i]) == "ce" and scores["generalist"][i] == 1.0 and scores[method][i] == 0.0
                    for i in range(n)
                ),
            }
            for method in METHODS
        },
        "contrasts": {
            key: {"score_delta": point[a]["score"] - point[b]["score"],
                  "ci95": [percentile(values, .025), percentile(values, .975)],
                  "cluster_permutation_p_two_sided": permutation_p.get(key),
                  "holm_adjusted_p": holm_p.get(key)}
            for (a, b), (key, values) in zip(CONTRASTS, contrast_distribution.items())
        },
        "caution": "TEST outcomes were previously inspected; descriptive paired analysis, not pristine held-out tuning evidence.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "paired_rows.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in per_sample))
    (args.output_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"n": n, "methods": {method: point[method] for method in METHODS}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
