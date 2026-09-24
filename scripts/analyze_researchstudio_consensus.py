#!/usr/bin/env python3
"""Fail-closed full-TEST paired analysis of strict consensus against Stage 1."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from analyze_researchstudio_ablation import percentile, repeated_long_span


def read(path: Path):
    with gzip.open(path, "rt") as handle:
        return json.load(handle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--stage1-run", type=Path, required=True)
    parser.add_argument("--stage1-rows", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--anchor-root", type=Path, required=True)
    parser.add_argument("--executed-code-identity", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.anchor_root.resolve()))
    from anchor.corrected_sgta.evaluate_medheval_answers import evaluate_rows
    from anchor.medeval.evaluate_mixed_vqa_table import answer_token_recall, source_task_group

    source_bytes = args.source.read_bytes()
    source = json.loads(source_bytes)
    stage1 = [json.loads(line) for line in args.stage1_rows.read_text().splitlines()]
    ids = [str(row["id"]) for row in source]
    if len(ids) != 451 or len(set(ids)) != 451 or [row["sample_id"] for row in stage1] != ids:
        raise ValueError("consensus analysis requires exact ordered full VQA-RAD TEST 451")
    observed = {p.parent.name for p in args.run.glob("*/provenance.json.gz")}
    if observed != set(ids):
        raise ValueError(f"consensus ID coverage mismatch: missing={list(set(ids)-observed)[:8]} extra={list(observed-set(ids))[:8]}")

    outputs, records, identity = [], [], None
    for row, previous in zip(source, stage1):
        sample_id = str(row["id"])
        folder, first_folder = args.run / sample_id, args.stage1_run / sample_id
        prov, old = read(folder / "provenance.json.gz"), read(first_folder / "provenance.json.gz")
        joint, old_joint = read(folder / "joint_all.json.gz"), read(first_folder / "joint_all.json.gz")
        output = read(folder / "consensus_strict.json.gz")
        if prov.get("methods") != ["joint_all", "consensus_strict"]:
            raise ValueError(f"unexpected consensus arms: {sample_id}")
        for key in ("row", "prompt", "manifest_sha256", "protocol_sha256", "cache_identity", "receiver_config", "native_cache_sha256", "matched_joint_evidence"):
            if prov.get(key) != old.get(key):
                raise ValueError(f"Stage-1 identity mismatch {sample_id}: {key}")
        if prov["row"]["id"] != sample_id or prov["row"]["question"] != row["question"] or prov["row"]["image_sha256"] != row["image_sha256"]:
            raise ValueError(f"source/row mismatch: {sample_id}")
        if joint["token_ids"] != old_joint["token_ids"] or joint["evidence"] != old_joint["evidence"] or output["evidence"] != old_joint["evidence"]:
            raise ValueError(f"matched joint/evidence mismatch: {sample_id}")
        if output.get("aggregation") != "strict_base_relative_probability_q_equals_m" or output.get("bounded_commit") is not False:
            raise ValueError(f"wrong consensus rule: {sample_id}")
        key = (prov["manifest_sha256"], prov["protocol_sha256"], prov["cache_identity"], json.dumps(prov["receiver_config"], sort_keys=True))
        if identity is None:
            identity = key
        elif identity != key:
            raise ValueError(f"consensus protocol changed mid-run: {sample_id}")
        if previous["evidence_id"] != prov["cache_identity"] or previous["protocol_id"] != prov["protocol_sha256"]:
            raise ValueError(f"paired-row identity mismatch: {sample_id}")
        outputs.append(output)
        records.append({"sample_id": sample_id, "cluster_id": previous["cluster_id"], "answer_type": source_task_group(row),
                        "generalist_score": previous["methods"]["generalist"]["score"],
                        "geomedian_score": previous["methods"]["isolated_geomedian"]["score"],
                        "bard_score": previous["methods"]["bard"]["score"],
                        "consensus_text": output.get("text", ""), "finished": bool(output["finished"]),
                        "generated_tokens": len(output["token_ids"]), "seconds": output["seconds"],
                        "expert_count": len(output["expert_branches"]),
                        "evidence_items": len(output["evidence"])})

    scorer_rows = [{**row, "question_id": row["id"], "gt_ans": row["answer"], "text": output.get("text", "")}
                   for row, output in zip(source, outputs)]
    details = evaluate_rows(scorer_rows)["details"]
    if [str(detail["question_id"]) for detail in details] != ids:
        raise ValueError("scorer changed consensus ID order")
    for row, output, detail, record in zip(source, outputs, details, records):
        record["score"] = (float(bool(detail["correct"])) if source_task_group(row) == "ce"
                           else answer_token_recall(output.get("text", ""), row["answer"]))
        record["parsed"] = detail["prediction"] is not None
        record["repeated_long_span_flag"] = repeated_long_span(output.get("text", ""))

    n = len(records)
    clusters = defaultdict(list)
    for record in records:
        clusters[record["cluster_id"]].append(record)
    cluster_rows = list(clusters.values())
    contrasts = {}
    for label, field in (("generalist", "generalist_score"), ("isolated_geomedian", "geomedian_score"), ("bard", "bard_score")):
        delta = sum(record["score"] - record[field] for record in records) / n
        draws = []
        rng = random.Random(20260924)  # identical sampled clusters across contrasts
        for _ in range(10000):
            sampled = [cluster_rows[rng.randrange(len(cluster_rows))] for _ in cluster_rows]
            draws.append(sum(record["score"] - record[field] for group in sampled for record in group) /
                         sum(len(group) for group in sampled))
        contrasts[f"consensus_strict-{label}"] = {"score_delta": delta, "ci95": [percentile(draws, .025), percentile(draws, .975)]}
    base_delta = [record["score"] - record["generalist_score"] for record in records]
    report = {
        "status": "complete_full_test_descriptive", "dataset": "VQA-RAD", "receiver": "LLaVA-Med-7B", "n": n,
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "executed_code_identity": args.executed_code_identity,
        "stage1_run": str(args.stage1_run.resolve()), "consensus_run": str(args.run.resolve()),
        "manifest_sha256": identity[0], "protocol_sha256": identity[1], "native_cache_identity": identity[2],
        "score_definition": "CE strict 0/1; OE reference-token recall; sample-weighted mean",
        "rule": "strict q=m base-relative normalized-probability consensus; evidence-conditioned adaptation",
        "score": sum(record["score"] for record in records) / n,
        "rescue_vs_generalist": sum(max(x, 0) for x in base_delta) / n,
        "harm_vs_generalist": sum(max(-x, 0) for x in base_delta) / n,
        "blank_n": sum(not record["consensus_text"].strip() for record in records),
        "unfinished_n": sum(not record["finished"] for record in records),
        "parsed_n": sum(record["parsed"] for record in records),
        "repeated_long_span_flag_n": sum(record["repeated_long_span_flag"] for record in records),
        "receiver_decode_seconds_total": sum(record["seconds"] for record in records),
        "generated_tokens_total": sum(record["generated_tokens"] for record in records),
        "cluster_bootstrap": {"unit": "patient_id_else_image_sha256", "clusters": len(clusters), "replicates": 10000, "seed": 20260924},
        "contrasts": contrasts,
        "caution": "TEST previously inspected; descriptive inference only. Native expert acquisition cost is not included.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "paired_rows.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records))
    (args.output_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"n": n, "score": report["score"], "contrasts": contrasts}, ensure_ascii=False))


if __name__ == "__main__":
    main()
