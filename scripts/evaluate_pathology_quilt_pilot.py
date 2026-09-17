"""Offline evaluation only; reuse frozen ANCHOR, never inject labels into inference."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from merit_feddg.pathology_pilot import (  # noqa: E402
    ARMS, digest, file_sha, paired_summary, read_json, write_new,
)
from run_pathology_quilt_pilot import scorer_identity  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--references", type=Path, required=True)
    args = p.parse_args()
    frozen = read_json(args.run / "frozen.json")
    result = read_json(args.run / "result.json")
    identity = digest(frozen)
    if result.get("complete") is not True or result["identity"] != identity:
        raise ValueError("only complete frozen pilots can be evaluated")
    if scorer_identity(frozen["anchor_root"]) != frozen["scorer"]:
        raise ValueError("frozen scorer changed")
    if file_sha(frozen["manifest"]) != frozen["manifest_sha256"]:
        raise ValueError("frozen manifest changed")
    rows = {str(r["id"]): r for r in map(json.loads, Path(frozen["manifest"]).read_text().splitlines())}
    references = read_json(args.references)
    if set(references) != set(rows):
        raise ValueError("references must align to the original complete TRAIN manifest")
    ids = [r["id"] for r in frozen["rows"]]
    records = [read_json(f) for f in sorted((args.run / "cases").glob("*.json"))]
    if (len(records) != len(ids) or {r["id"] for r in records} != set(ids)
            or any(r["identity"] != identity for r in records)):
        raise ValueError("missing/duplicate/foreign case records")
    records = {r["id"]: r for r in records}
    sys.path.insert(0, frozen["anchor_root"])
    from anchor.corrected_sgta.evaluate_medheval_answers import PROTOCOL_VERSION, evaluate_rows
    from anchor.medeval.evaluate_mixed_vqa_table import answer_token_recall

    scores, details = {}, {}
    for arm in ARMS:
        evaluated = evaluate_rows([
            {"qid": k, "question": rows[k]["question"], "answer_type": rows[k]["answer_type"],
             "answer": references[k][0], "text": records[k]["arms"][arm]["text"]} for k in ids])
        detail = {str(v["question_id"]): v for v in evaluated["details"]}
        scores[arm] = {k: float(detail[k]["correct"]) if rows[k]["answer_type"] == "closed"
                       else float(answer_token_recall(records[k]["arms"][arm]["text"], references[k][0]))
                       for k in ids}
        details[arm] = detail
    summary = {arm: paired_summary(scores[arm], scores["compact"]) for arm in ARMS}
    for arm in ARMS:
        summary[arm]["closed"] = paired_summary(
            {k: scores[arm][k] for k in ids if rows[k]["answer_type"] == "closed"},
            {k: scores["compact"][k] for k in ids if rows[k]["answer_type"] == "closed"}
        ) if any(rows[k]["answer_type"] == "closed" for k in ids) else None
        summary[arm]["open"] = paired_summary(
            {k: scores[arm][k] for k in ids if rows[k]["answer_type"] == "open"},
            {k: scores["compact"][k] for k in ids if rows[k]["answer_type"] == "open"}
        ) if any(rows[k]["answer_type"] == "open" for k in ids) else None
    cache = read_json(args.run / "quilt_predictions.json")
    write_new(args.run / "evaluation.json", {
        "identity": identity, "scorer": PROTOCOL_VERSION, "scorer_hashes": frozen["scorer"],
        "references_sha256": file_sha(args.references), "arms": summary,
        "matched_vs_wrong_image": paired_summary(scores["compact_quilt"], scores["compact_wrong_image"]),
        "per_case_scores": scores, "details": details, "coverage": frozen["coverage"],
        "historical_parity": sum(r["historical_token_parity"] for r in records.values()),
        "conch_delivered_cases": sum(r["conch_was_delivered"] for r in records.values()),
        "image_clusters": len({r["image_sha256"] for r in records.values()}),
        "cost": {"actor_calls": result["actual_actor_calls"],
                 "quilt_calls": cache["actual_model_calls"], "quilt_load_seconds": cache["model_load_seconds"],
                 "quilt_inference_seconds": sum(r["seconds"] for r in cache["predictions"].values()),
                 "actor_load_seconds": result["actor_load_seconds"]},
        "scope": "small microscopy TRAIN diagnostic; not full PathVQA or clinical accuracy",
    })


if __name__ == "__main__":
    main()
