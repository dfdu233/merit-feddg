"""Read-only diagnostic of published outcomes; never imported by inference."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def analyze(root):
    root = Path(root)
    for line in (root / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        if hashlib.sha256((root / name.strip()).read_bytes()).hexdigest() != expected:
            raise ValueError(f"bundle checksum mismatch: {name}")
    rows = [json.loads(line) for line in (root / "per-case-results.jsonl").read_text().splitlines()]
    coverage, reasons, accepted = defaultdict(Counter), Counter(), []
    image_declines = 0
    for row in rows:
        modality = row["routing"]["modality"]
        coverage[modality]["cases"] += 1
        all_arm = row["arms"]["spatial_weighted"]
        events = [e for e in all_arm["trace"] if e["event"] == "tool"]
        coverage[modality]["cases_with_adopted_evidence"] += int(any(e.get("adopted") for e in events))
        coverage[modality]["answer_changed"] += int(all_arm["text"] != row["arms"]["generalist"]["text"])
        for event in row["arms"]["spatial_gate"]["trace"]:
            gate = event.get("vector_gate")
            if not gate:
                continue
            reasons[gate["reason"]] += 1
            if not gate["accepted"]:
                continue
            before, after = gate["support"]
            image_gain = after["image_mean_logp"] - before["image_mean_logp"]
            image_declines += int(image_gain < 0)
            accepted.append({"id": row["id"], "question": row["question"], "expert": event["expert"],
                             "references_for_offline_audit_only": row["references"],
                             "baseline": row["arms"]["generalist"]["text"],
                             "gated_answer": row["arms"]["spatial_gate"]["text"],
                             "image_gain": image_gain, "visual_contrast_gain": gate["gain"],
                             "control_gain": after["control_mean_logp"] - before["control_mean_logp"]})
    return {"n": len(rows), "bundle_verified": True, "coverage_by_predicted_modality": dict(coverage),
            "gate_reasons": reasons, "accepted_count": len(accepted),
            "accepted_with_original_image_score_decline": image_declines,
            "equal_vs_weighted_text_differences": sum(r["arms"]["spatial_equal"]["text"] !=
                r["arms"]["spatial_weighted"]["text"] for r in rows), "accepted_cases": accepted,
            "limitation": "Retrospective diagnostic, not a replay or new experiment; large raw masks omitted."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    Path(args.output).write_text(json.dumps(analyze(args.bundle), ensure_ascii=False, indent=2) + "\n")
