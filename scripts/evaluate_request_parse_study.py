"""Offline semantic fidelity only. Clinical answer labels are not accepted as gold."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from merit_feddg.request_parse_study import MODES, audit_parse, digest, fidelity, parse_response


def evaluate(frozen, predictions, gold_rows, packet_rows=None):
    if predictions["identity"] != frozen["identity"]:
        raise ValueError("prediction/study identity mismatch")
    rows = {r["id"]: r for r in frozen["questions"]}
    gold = {}
    for entry in gold_rows:
        key = entry["id"]
        if key in gold or key not in rows or entry["question"] != rows[key]["question"]:
            raise ValueError("gold identity mismatch/duplicate")
        if entry.get("reviewed") is not True or not str(entry.get("reviewer", "")).strip():
            raise ValueError("independent semantic review required")
        parsed = parse_response(json.dumps(entry["gold_parse"]), entry["question"])
        if not parsed["valid"]:
            raise ValueError("invalid reviewed parse: " + parsed["reason"])
        gold[key] = parsed
    if set(gold) != set(rows):
        raise ValueError("all selected questions need review; do not drop failures")
    packets = {}
    for entry in packet_rows or []:
        key = entry["id"]
        expected_hash = frozen.get("image_hashes", {}).get(key)
        if (key in packets or key not in rows or entry.get("question") != rows[key]["question"]
                or entry.get("study_identity") != frozen["identity"] or not expected_hash
                or entry.get("image_sha256") != expected_hash):
            raise ValueError("packet identity/image mismatch or missing source identity")
        from merit_feddg.capabilities import EvidenceItem
        packets[key] = tuple(EvidenceItem(**item) for item in entry["items"])
    metrics, details, seen = defaultdict(lambda: defaultdict(int)), [], set()
    for r in predictions["records"]:
        key, expert, mode = r["id"], r["expert"], r["mode"]
        uid = (key, expert, mode)
        if uid in seen or key not in gold or expert not in frozen["specs"] or mode not in MODES:
            raise ValueError("duplicate/unknown prediction")
        seen.add(uid)
        if r["question"] != rows[key]["question"]:
            raise ValueError("prediction question mismatch")
        # Revalidate raw model output rather than trusting stored 'valid' metadata.
        parsed = parse_response(r["output"]["text"], r["question"])
        f = fidelity(parsed, gold[key], frozen["specs"][expert])
        m = metrics[mode + ":" + expert]
        m["n"] += 1
        m["schema_valid"] += parsed["valid"]
        m["gold_unknown"] += gold[key]["parse"]["status"] == "unknown"
        m["parsed_not_unknown"] += parsed["valid"] and parsed["parse"]["status"] == "parsed"
        for name in ("exact", "tp", "pred_count", "gold_count"):
            m[name] += f[name]
        m["cases_with_omissions"] += bool(f["missing_atoms"] or f["missing_relations"])
        detail = {"id": key, "expert": expert, "mode": mode, **f}
        if key in packets:
            items = tuple(x for x in packets[key] if x.expert_id == expert)
            # Missing packets are unavailable measurements, not successful rejections.
            if items:
                spec = frozen["specs"][expert]
                a = audit_parse(parsed, rows[key], expert, spec, items)
                b = audit_parse(gold[key], rows[key], expert, spec, items)
                actual = a["delivery_audit"]["delivered_count"] > 0
                expected = b["delivery_audit"]["delivered_count"] > 0
                m["delivery_pairs_evaluated"] += 1
                m["unexpected_admissions"] += actual and not expected
                m["missed_admissions"] += expected and not actual
                m["expected_admissions"] += expected
                m["retained_expected_admissions"] += actual and expected
                detail["delivery_vs_reviewed_request"] = {"predicted": actual, "reviewed": expected}
        details.append(detail)
    expected = {(key, expert, mode) for key in rows for expert in frozen["specs"] for mode in MODES}
    if seen != expected:
        raise ValueError("incomplete predictions; cannot report a complete study")
    for m in metrics.values():
        m["exact_rate"] = m["exact"] / m["n"]
        m["precision"] = m["tp"] / m["pred_count"] if m["pred_count"] else None
        m["recall"] = m["tp"] / m["gold_count"] if m["gold_count"] else None
    return {"identity": frozen["identity"], "review_sha": digest(gold_rows),
            "metrics": dict(metrics), "details": details,
            "medical_accuracy_evaluated": False,
            "permission_precision_evaluated": False,
            "fixed_policy_delivery_comparison": bool(packets),
            "warning": "Span-valid is not semantically complete. Optional delivery compares to reviewed requests under the SAME policy; not clinical usefulness."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--gold", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--packets", type=Path, help="Optional verified native packets; never sent to parser")
    args = p.parse_args()
    frozen = json.loads((args.run / "frozen.json").read_text())
    predictions = json.loads((args.run / "predictions.json").read_text())
    gold = [json.loads(line) for line in args.gold.read_text().splitlines() if line.strip()]
    packet_rows = ([json.loads(line) for line in args.packets.read_text().splitlines() if line.strip()]
                   if args.packets else None)
    report = evaluate(frozen, predictions, gold, packet_rows)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    main()
