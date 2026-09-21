"""Offline candidate-oracle and trajectory-divergence audit for existing runs.

This diagnostic is post-generation only. It may read references to explain a
completed experiment, but it never feeds labels back into inference or expert
selection.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from merit_feddg.contribution import answer_metrics


def load_outputs(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("outputs"), dict):
        payload = payload["outputs"]
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"invalid output mapping: {path}")
    result = {}
    for sample_id, row in payload.items():
        if not isinstance(row, dict) or not isinstance(row.get("text"), str):
            raise TypeError(f"invalid output row for {sample_id}")
        tokens = row.get("token_ids", [])
        if not isinstance(tokens, list) or any(type(token) is not int for token in tokens):
            raise ValueError(f"invalid token_ids for {sample_id}")
        result[str(sample_id)] = {"text": row["text"], "token_ids": tokens}
    return result


def parse_named(value):
    name, separator, path = value.partition("=")
    if not separator or not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError("method must use NAME=/path/to/output.json")
    return name.strip(), path.strip()


def first_divergence(left, right):
    limit = min(len(left), len(right))
    for index in range(limit):
        if left[index] != right[index]:
            return index
    return None if len(left) == len(right) else limit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--method", action="append", type=parse_named, required=True)
    parser.add_argument("--primary", required=True)
    parser.add_argument("--references", required=True)
    parser.add_argument("--metric", choices=("token_f1", "exact_match"), default="token_f1")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    methods = dict(args.method)
    if len(methods) != len(args.method):
        parser.error("method names must be unique")
    if args.primary not in methods:
        parser.error("--primary must name one --method")

    baseline = load_outputs(args.baseline)
    outputs = {name: load_outputs(path) for name, path in methods.items()}
    references = json.loads(Path(args.references).read_text(encoding="utf-8"))
    if not isinstance(references, dict):
        raise TypeError("references must be a sample-id mapping")
    sample_ids = list(baseline)
    if set(references) != set(sample_ids):
        raise ValueError("reference IDs must exactly match baseline IDs")
    for name, rows in outputs.items():
        if set(rows) != set(sample_ids):
            raise ValueError(f"{name} IDs do not match baseline")

    metric = args.metric
    totals = Counter()
    divergence_hist = Counter()
    best_counts = Counter()
    cases = []
    for sample_id in sample_ids:
        refs = references[sample_id]
        if isinstance(refs, str):
            refs = [refs]
        base_score = answer_metrics(baseline[sample_id]["text"], refs)[metric]
        scores = {
            name: answer_metrics(row[sample_id]["text"], refs)[metric]
            for name, row in outputs.items()
        }
        primary_score = scores[args.primary]
        if primary_score > base_score:
            outcome = "rescue_or_gain"
        elif primary_score < base_score:
            outcome = "harm_or_loss"
        else:
            outcome = "unchanged"
        totals[outcome] += 1

        divergence = first_divergence(
            baseline[sample_id]["token_ids"],
            outputs[args.primary][sample_id]["token_ids"],
        )
        if divergence is None:
            bucket = "identical"
        elif divergence == 0:
            bucket = "token_1"
        elif divergence <= 2:
            bucket = "tokens_2_3"
        elif divergence <= 4:
            bucket = "tokens_4_5"
        else:
            bucket = "after_5"
        divergence_hist[bucket] += 1
        if outcome == "harm_or_loss":
            divergence_hist[f"harm:{bucket}"] += 1

        all_scores = {"generalist": base_score, **scores}
        oracle_score = max(all_scores.values())
        best = sorted(name for name, value in all_scores.items() if value == oracle_score)
        for name in best:
            best_counts[name] += 1
        cases.append(
            {
                "id": sample_id,
                "baseline_score": base_score,
                "primary_score": primary_score,
                "outcome": outcome,
                "first_divergence_zero_based": divergence,
                "scores": scores,
                "oracle_score": oracle_score,
                "oracle_best": best,
            }
        )

    n = len(sample_ids)
    base_mean = sum(row["baseline_score"] for row in cases) / n
    primary_mean = sum(row["primary_score"] for row in cases) / n
    oracle_mean = sum(row["oracle_score"] for row in cases) / n
    harm_cases = max(1, totals["harm_or_loss"])
    early_harms = sum(
        divergence_hist[f"harm:{bucket}"]
        for bucket in ("token_1", "tokens_2_3")
    )
    payload = {
        "schema": "merit-tx-failure-audit-v1",
        "n": n,
        "metric": metric,
        "primary": args.primary,
        "baseline_mean": base_mean,
        "primary_mean": primary_mean,
        "oracle_mean": oracle_mean,
        "oracle_headroom_over_baseline": oracle_mean - base_mean,
        "primary_delta": primary_mean - base_mean,
        "outcomes": dict(totals),
        "first_divergence": dict(divergence_hist),
        "early_divergence_fraction_among_harms": early_harms / harm_cases,
        "oracle_best_counts_with_ties": dict(best_counts),
        "cases": cases,
        "generation_or_selection_uses_references": False,
        "diagnostic_only": True,
    }
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
