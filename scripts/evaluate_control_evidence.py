"""Complete-only offline CRES scoring with the existing pinned ANCHOR metric."""
import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

from merit_feddg.agent_evaluate import cluster_bootstrap
from merit_feddg.control_study import validate_outputs
from merit_feddg.open_study import atomic_json, fingerprint

SCORERS = ("anchor/corrected_sgta/evaluate_medheval_answers.py",
           "anchor/medeval/evaluate_mixed_vqa_table.py")


def paired(values, baseline, rows, records, arm, comparator):
    delta = [values[r["id"]] - baseline[r["id"]] for r in rows]
    changed = [records[r["id"]]["arms"][arm]["token_ids"] !=
               records[r["id"]]["arms"][comparator]["token_ids"] for r in rows]
    harms = sum(d < 0 for d in delta)
    return {"mean_delta": statistics.mean(delta), "improvements": sum(d > 0 for d in delta),
            "harms": harms, "revision_coverage": statistics.mean(changed),
            "harm_per_all_cases": harms / len(rows),
            "harm_given_revision": harms / sum(changed) if any(changed) else None,
            "image_cluster_bootstrap_95": cluster_bootstrap(delta, [r["image_sha256"] for r in rows])}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--anchor-root", type=Path, required=True)
    p.add_argument("--references", type=Path)
    p.add_argument("--freeze-scorer", action="store_true")
    args = p.parse_args()
    root = args.run
    frozen = json.loads((root / "frozen.json").read_text())

    def hashes():
        return {s: hashlib.sha256((args.anchor_root / s).read_bytes()).hexdigest() for s in SCORERS}

    current = hashes()
    pin = root / "scorer-frozen.json"
    if args.freeze_scorer:
        if pin.exists() and json.loads(pin.read_text()) != current:
            raise ValueError("cannot replace a different scorer pin")
        atomic_json(pin, current)
        print("Scorer pinned; references not read")
        return
    if not pin.exists() or json.loads(pin.read_text()) != current:
        raise ValueError("scorer missing or changed")
    identity = fingerprint(frozen)
    complete = json.loads((root / "complete.json").read_text())
    rows, arms = frozen["rows"], frozen["arms"]
    if (not complete.get("full_manifest_complete") or complete.get("identity") != identity
            or complete.get("n") != len(rows)):
        raise ValueError("complete full manifest required")
    paths = {p.stem: p for p in (root / "cases").glob("*.json")}
    if set(paths) != {fingerprint(r["id"]) for r in rows}:
        raise ValueError("missing or extra case files")
    records = {r["id"]: json.loads(paths[fingerprint(r["id"])].read_text()) for r in rows}
    if any(v.get("identity") != identity or v.get("id") != k or set(v.get("arms", {})) != set(arms)
           for k, v in records.items()):
        raise ValueError("case identity/arm mismatch")
    for record in records.values():
        validate_outputs(record, arms)
    if args.references is None:
        raise ValueError("separate offline references required")
    if (root / "evaluation.json").exists():
        raise ValueError("evaluation exists; do not overwrite")
    refs = json.loads(args.references.read_text())
    if (set(refs) != set(records) or any(not isinstance(v, list) or not v or
            any(not isinstance(s, str) or not s for s in v) for v in refs.values())):
        raise ValueError("exact complete reference ID set and nonempty string lists required")
    sys.path.insert(0, str(args.anchor_root.resolve()))
    from anchor.corrected_sgta.evaluate_medheval_answers import PROTOCOL_VERSION, evaluate_rows
    from anchor.medeval.evaluate_mixed_vqa_table import answer_token_recall

    scores = {}
    for arm in arms:
        evaluated = evaluate_rows([{"qid": r["id"], "question": r["question"],
            "answer_type": r["answer_type"], "answer": refs[r["id"]][0],
            "text": records[r["id"]]["arms"][arm]["text"]} for r in rows])
        details = {v["question_id"]: v for v in evaluated["details"]}
        scores[arm] = {r["id"]: float(details[r["id"]]["correct"]) if r["answer_type"] == "closed"
                       else answer_token_recall(records[r["id"]]["arms"][arm]["text"], refs[r["id"]][0])
                       for r in rows}
    report = {"identity": identity, "n": len(rows), "scorer": PROTOCOL_VERSION,
              "metric": "ANCHOR decoded CLOSED + OPEN token recall; first reference",
              "prompt_contract": frozen["options"]["prompt_contract"], "scorer_hashes": current,
              "reference_sha256": hashlib.sha256(args.references.read_bytes()).hexdigest(),
              "clinical_accuracy_or_causal_uplift_claim": False,
              "applicable": sum(v["applicable"] for v in records.values()),
              "controls_available": sum(v["controls_available"] for v in records.values()),
              "all_arm_wall_seconds": sum(v["wall_seconds"] for v in records.values()),
              "startup_seconds": sum(json.loads(p.read_text())["seconds"] for p in root.glob("startup-*.json")),
              "arms": {}}
    for arm in arms:
        output = [records[r["id"]]["arms"][arm] for r in rows]
        steps = [s for v in output for s in v.get("steps", [])]
        report["arms"][arm] = {
            "mixed_score": statistics.mean(scores[arm].values()),
            **{f"{kind}_score": statistics.mean(scores[arm][r["id"]] for r in rows if r["answer_type"] == kind)
               if any(r["answer_type"] == kind for r in rows) else None for kind in ("closed", "open")},
            "comparisons": {c: paired(scores[arm], scores[c], rows, records, arm, c)
                            for c in ("generalist", "compact", "deletion")},
            "mean_decode_seconds": statistics.mean(v["seconds"] for v in output),
            "score_calls": sum(v.get("score_calls", v.get("base_score_calls", 0) +
                                    v.get("conditioned_score_calls", 0)) for v in output),
            "reused_control_rows": sum(v.get("reused_control", False) for v in output),
            "guidance_active_rows": sum(v.get("guidance_applied", False) for v in output),
            "max_measured_token_kl": max((s["kl"] for s in steps if "kl" in s), default=None),
            "mean_selected_strength": statistics.mean(s["strength"] for s in steps if "strength" in s)
            if any("strength" in s for s in steps) else None}
    if hashes() != current:
        raise ValueError("scorer changed during evaluation")
    # Aggregate output excludes per-case patient text and references.
    atomic_json(root / "evaluation.json", {**report, "per_case_scores": scores})
    atomic_json(root / "evaluation-summary.json", report)
    print("Full evaluation complete:", root / "evaluation-summary.json")


if __name__ == "__main__":
    main()
