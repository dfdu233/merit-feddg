#!/usr/bin/env python3
"""Fail-closed descriptive analysis of the frozen PathVQA failure-domain subset."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from analyze_researchstudio_ablation import percentile, repeated_long_span

METHODS = ("generalist", "joint_all", "isolated_mean", "isolated_geomedian", "bard")


def read(path: Path):
    with gzip.open(path, "rt") as handle:
        return json.load(handle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--anchor-root", type=Path, required=True)
    parser.add_argument("--executed-code-identity", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.anchor_root.resolve()))
    from anchor.corrected_sgta.evaluate_medheval_answers import evaluate_rows
    from anchor.medeval.evaluate_mixed_vqa_table import answer_token_recall, source_task_group

    source_bytes, selection_bytes, manifest_bytes = (p.read_bytes() for p in (args.source, args.selection, args.manifest))
    source = json.loads(source_bytes)
    manifest = {row["id"]: row for row in map(json.loads, manifest_bytes.splitlines())}
    selected = selection_bytes.decode().splitlines()
    source_map = {str(row["id"]): row for row in source}
    if len(source) != 6719 or len(source_map) != 6719 or len(selected) != 58 or len(set(selected)) != 58 or set(selected) - set(source_map):
        raise ValueError("frozen 58/6719 PathVQA source/selection mismatch")
    observed = {p.parent.name for p in args.run.glob("*/provenance.json.gz")}
    if observed != set(selected):
        raise ValueError(f"selected-ID coverage mismatch: missing={list(set(selected)-observed)[:8]} extra={list(observed-set(selected))[:8]}")

    ordered = [row for row in source if str(row["id"]) in observed]
    predictions = {method: [] for method in METHODS}
    cases, identity = [], None
    for row in ordered:
        sid = str(row["id"])
        folder = args.run / sid
        prov = read(folder / "provenance.json.gz")
        frozen = manifest[sid]
        if (prov.get("row", {}).get("id") != sid or prov["row"].get("question") != row["question"]
                or prov["row"].get("image_sha256") != row["image_sha256"]
                or frozen["image_sha256"] != row["image_sha256"] or frozen["question"] != row["question"]
                or prov.get("manifest_sha256") != hashlib.sha256(manifest_bytes).hexdigest()
                or not prov.get("matched_joint_evidence") or prov.get("methods") != list(METHODS)):
            raise ValueError(f"input/method/manifest mismatch: {sid}")
        key = (prov["cache_identity"], prov["protocol_sha256"], json.dumps(prov["receiver_config"], sort_keys=True))
        if identity is None:
            identity = key
        elif identity != key:
            raise ValueError(f"protocol/config changed mid-run: {sid}")
        outputs = {method: read(folder / f"{method}.json.gz") for method in METHODS}
        native = outputs["joint_all"].get("evidence", [])
        if any(outputs[method].get("evidence", []) != native for method in METHODS[2:]):
            raise ValueError(f"delivered evidence differs across arms: {sid}")
        for method in METHODS:
            predictions[method].append(outputs[method])
        groups = int(outputs["bard"].get("adopted_evidence_count", 0))
        if groups > len(native):
            raise ValueError(f"impossible delivered group count: {sid}")
        # PathVQA's normalized source labels every row Pathology; the frozen
        # image-only route is the modality used to stratify this diagnostic.
        cases.append({"sample_id": sid, "image_sha256": row["image_sha256"], "modality": prov["row"]["modality"],
                      "answer_type": source_task_group(row), "delivered_group_count": groups,
                      "selected_evidence_items": int(outputs["joint_all"]["selected_evidence_count"]),
                      "presented_evidence_items": int(outputs["joint_all"]["presented_evidence_count"]),
                      "methods": {method: {"text": outputs[method].get("text", ""),
                                           "finished": bool(outputs[method]["finished"]),
                                           "generated_tokens": len(outputs[method]["token_ids"]),
                                           "seconds": outputs[method]["seconds"]}
                                  for method in METHODS}})

    for method in METHODS:
        rows = [{**row, "question_id": row["id"], "gt_ans": row["answer"], "text": output.get("text", "")}
                for row, output in zip(ordered, predictions[method])]
        details = evaluate_rows(rows)["details"]
        if [str(item["question_id"]) for item in details] != [str(row["id"]) for row in ordered]:
            raise ValueError(f"scorer reordered IDs: {method}")
        for case, row, output, detail in zip(cases, ordered, predictions[method], details):
            arm = case["methods"][method]
            arm["score"] = (float(bool(detail["correct"])) if source_task_group(row) == "ce"
                            else answer_token_recall(output.get("text", ""), row["answer"]))
            arm["parsed"] = detail["prediction"] is not None
            arm["repeated_long_span_flag"] = repeated_long_span(arm["text"])

    n = len(cases)
    clusters = defaultdict(list)
    for case in cases:
        clusters[case["image_sha256"]].append(case)
    cluster_rows = list(clusters.values())
    methods = {}
    for method in METHODS:
        values = [case["methods"][method]["score"] for case in cases]
        base = [case["methods"]["generalist"]["score"] for case in cases]
        delta = [value - ref for value, ref in zip(values, base)]
        methods[method] = {
            "score": sum(values) / n,
            "rescue_vs_generalist": sum(max(x, 0) for x in delta) / n,
            "harm_vs_generalist": sum(max(-x, 0) for x in delta) / n,
            "blank_n": sum(not case["methods"][method]["text"].strip() for case in cases),
            "unfinished_n": sum(not case["methods"][method]["finished"] for case in cases),
            "parsed_n": sum(case["methods"][method]["parsed"] for case in cases),
            "repeated_long_span_flag_n": sum(case["methods"][method]["repeated_long_span_flag"] for case in cases),
        }
    contrasts = {}
    for arm, reference in (("joint_all", "generalist"), ("isolated_mean", "joint_all"),
                           ("isolated_geomedian", "isolated_mean"), ("bard", "isolated_geomedian")):
        effect = sum(case["methods"][arm]["score"] - case["methods"][reference]["score"] for case in cases) / n
        rng = random.Random(20260924)
        draws = []
        for _ in range(10000):
            sampled = [cluster_rows[rng.randrange(len(cluster_rows))] for _ in cluster_rows]
            draws.append(sum(case["methods"][arm]["score"] - case["methods"][reference]["score"]
                             for group in sampled for case in group) / sum(map(len, sampled)))
        contrasts[f"{arm}-{reference}"] = {"score_delta": effect, "ci95": [percentile(draws, .025), percentile(draws, .975)]}
    groups = Counter(case["delivered_group_count"] for case in cases)
    modalities = Counter(case["modality"] for case in cases)
    strata = {}
    for field in ("modality", "answer_type", "delivered_group_count"):
        buckets = defaultdict(list)
        for case in cases:
            buckets[str(case[field])].append(case)
        strata[field] = {name: {"n": len(items), "scores": {
            method: sum(item["methods"][method]["score"] for item in items) / len(items)
            for method in METHODS}} for name, items in sorted(buckets.items())}
    report = {
        "status": "complete_frozen_58_case_exploratory_diagnostic", "not_full_test": True,
        "dataset": "PathVQA", "receiver": "LLaVA-Med-7B", "n": n, "full_test_n": 6719,
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "selection_sha256": hashlib.sha256(selection_bytes).hexdigest(),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "native_cache_identity": identity[0], "protocol_sha256": identity[1],
        "executed_code_identity": args.executed_code_identity,
        "score_definition": "CE strict 0/1; OE reference-token recall; selected-case mean only",
        "sampling": "up to eight unique images per frozen modality, SHA256(modality:sample_id) rank; rare modalities have one image",
        "cluster_bootstrap": {"unit": "image_sha256", "clusters": len(clusters), "replicates": 10000, "seed": 20260924},
        "delivered_group_counts": dict(groups), "modality_counts": dict(modalities),
        "selected_evidence_items_total": sum(case["selected_evidence_items"] for case in cases),
        "presented_evidence_items_total": sum(case["presented_evidence_items"] for case in cases),
        "selected_vs_presented_mismatch_n": sum(case["selected_evidence_items"] != case["presented_evidence_items"] for case in cases),
        "descriptive_strata": strata,
        "methods": methods, "contrasts": contrasts,
        "failure_diagnostics": {
            "low_source_coverage_n": sum(case["delivered_group_count"] <= 1 for case in cases),
            "isolated_geomedian_not_better_than_generalist_n": sum(case["methods"]["isolated_geomedian"]["score"] <= case["methods"]["generalist"]["score"] for case in cases),
            "bard_worse_than_geomedian_n": sum(case["methods"]["bard"]["score"] < case["methods"]["isolated_geomedian"]["score"] for case in cases),
        },
        "caution": "Exploratory TEST subset selected without labels, intentionally modality-balanced and not population-weighted; never report as full PathVQA accuracy or tune a gate on it.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "paired_rows.jsonl").write_text("".join(json.dumps(case, ensure_ascii=False) + "\n" for case in cases))
    (args.output_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"n": n, "methods": {m: methods[m]["score"] for m in METHODS}, "diagnostics": report["failure_diagnostics"]}))


if __name__ == "__main__":
    main()
