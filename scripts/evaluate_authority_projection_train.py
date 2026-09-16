"""Offline evaluation for capability-authority free-text TRAIN pilot."""
from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
from pathlib import Path

from merit_feddg.agent_evaluate import cluster_bootstrap
from merit_feddg.open_study import atomic_json, fingerprint

ARMS = ("incumbent", "scope_text", "base_rerank", "text_rerank", "authority_projection")


def paired(candidate, incumbent, ids):
    deltas = [candidate[key] - incumbent[key] for key in ids]
    return {
        "mean_delta": statistics.mean(deltas),
        "improvements": sum(delta > 0 for delta in deltas),
        "harms": sum(delta < 0 for delta in deltas),
        "unchanged": sum(delta == 0 for delta in deltas),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--anchor-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.run
    if (root / "evaluation.json").exists():
        raise RuntimeError("refuse to overwrite prior evaluation")

    config = json.loads((root / "frozen.json").read_text())
    identity = fingerprint(config)
    completion = json.loads((root / "generation-complete.json").read_text())
    if completion["identity"] != identity:
        raise RuntimeError("run identity mismatch")
    case_files = {path.stem: path for path in (root / "cases").glob("*.json")}
    selected_ids = [item["id"] for item in config["selected"]]
    if set(case_files) - set(selected_ids):
        raise RuntimeError("unexpected case IDs")
    ids = [key for key in selected_ids if key in case_files]
    if not ids:
        raise RuntimeError("no generated cases to evaluate")

    records = {key: json.loads(case_files[key].read_text()) for key in ids}
    if any(
        record["identity"] != identity or record["id"] != key
        for key, record in records.items()
    ):
        raise RuntimeError("case identity mismatch")
    source_run = Path(config["source_run"])
    source_frozen = json.loads((source_run / "frozen.json").read_text())
    manifest = Path(source_frozen["manifest"])
    rows = {row["id"]: row for row in map(json.loads, manifest.read_text().splitlines())}
    refs = json.loads((manifest.parent / "references.json").read_text())
    if any(key not in rows or key not in refs for key in ids):
        raise RuntimeError("evaluation references do not align with generated cases")

    sys.path.insert(0, str(args.anchor_root.resolve()))
    from anchor.corrected_sgta.evaluate_medheval_answers import PROTOCOL_VERSION, evaluate_rows
    from anchor.medeval.evaluate_mixed_vqa_table import answer_token_recall

    texts = {arm: {key: records[key]["outputs"][arm] for key in ids} for arm in ARMS}
    scores = {}
    parser_counts = {}
    for arm in ARMS:
        evaluated = evaluate_rows(
            [
                {
                    "qid": key,
                    "question": rows[key]["question"],
                    "answer_type": rows[key]["answer_type"],
                    "answer": refs[key][0],
                    "text": texts[arm][key],
                }
                for key in ids
            ]
        )
        details = {item["question_id"]: item for item in evaluated["details"]}
        parser_counts[arm] = dict(collections.Counter(item["parser"] for item in details.values()))
        scores[arm] = {
            key: (
                float(details[key]["correct"])
                if rows[key]["answer_type"] == "closed"
                else answer_token_recall(texts[arm][key], refs[key][0])
            )
            for key in ids
        }

    incumbent = scores["incumbent"]
    report = {
        "identity": identity,
        "n": len(ids),
        "expected_n": len(selected_ids),
        "partial": len(ids) != len(selected_ids),
        "scorer": PROTOCOL_VERSION,
        "scope": "fixed TRAIN mechanism pilot; not clinical accuracy or a test result",
        "arms": {},
        "mechanism": {},
    }
    clusters = [rows[key]["image_sha256"] for key in ids]
    for arm in ARMS:
        values = scores[arm]
        deltas = [values[key] - incumbent[key] for key in ids]
        report["arms"][arm] = {
            "mixed_score": statistics.mean(values.values()),
            "vs_incumbent": paired(values, incumbent, ids),
            "changed_text": sum(
                texts[arm][key].strip() != texts["incumbent"][key].strip()
                for key in ids
            ),
            "image_cluster_bootstrap": cluster_bootstrap(deltas, clusters),
            "parsers": parser_counts[arm],
        }

    measured = [
        record for record in records.values() if record["transport"]["status"] == "measured"
    ]
    projected = [
        record for record in records.values() if record["projection"]["status"] == "projected"
    ]
    source_consistent = 0
    for record in projected:
        source_positive = record["source"]["native_decision_coordinate"] >= 0.5
        selected_group = record["selected"]["authority_group"]
        consistent = selected_group == "positive" if source_positive else selected_group == "negative"
        source_consistent += int(consistent)

    report["mechanism"] = {
        "projection_feasible": len(projected),
        "projection_infeasible": len(ids) - len(projected),
        "language_transport_measured": len(measured),
        "language_transport_toward_source": sum(
            record["transport"]["toward_source"] is True for record in measured
        ),
        "language_transport_away_or_flat": sum(
            record["transport"]["toward_source"] is False for record in measured
        ),
        "authority_selected_source_side": source_consistent,
        "authority_selected_source_side_denominator": len(projected),
        "candidate_pool_mean_size": statistics.mean(
            len(record["candidate_pool"]) for record in records.values()
        ),
        "candidate_pool_both_sides": sum(
            record["mapped_group_counts"]["positive"] > 0
            and record["mapped_group_counts"]["negative"] > 0
            for record in records.values()
        ),
        "free_text_output": True,
        "source_coordinate_calibrated_probability": False,
    }
    atomic_json(root / "evaluation.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
