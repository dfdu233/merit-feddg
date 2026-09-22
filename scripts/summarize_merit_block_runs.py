from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean


def _read(path):
    rows = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("references_read") is not False:
            raise ValueError(
                f"line {line_number}: mechanism summary requires reference-free generation"
            )
        rows.append(row)
    if not rows:
        raise ValueError(f"empty MERIT-Block run: {path}")
    return rows


def _safe_mean(values):
    return float(mean(values)) if values else None


def summarize(rows):
    reasons = Counter()
    selected_experts = Counter()
    real_margins = []
    gammas = []
    control_counts = []
    drift_js = []
    drift_disagreement = []
    next_score_queries = 0
    sequence_score_calls = 0
    blocks = 0
    accepted = 0
    cases_with_accept = 0
    wall_seconds = 0.0

    for row in rows:
        wall_seconds += float(row.get("seconds", 0.0))
        case_accepted = False
        for block in row.get("trace", ()):
            blocks += 1
            if block.get("expert_block_committed"):
                accepted += 1
                case_accepted = True
            if block.get("selected_expert"):
                selected_experts[str(block["selected_expert"])] += 1
            drift = block.get("same_prefix_context_drift")
            if drift:
                drift_js.append(float(drift["mean_js"]))
                drift_disagreement.append(float(drift["greedy_disagreement_rate"]))
            for audit in (block.get("experts") or {}).values():
                reasons[str(audit.get("reason", "unknown"))] += 1
                if audit.get("real_margin") is not None:
                    real_margins.append(float(audit["real_margin"]))
                if audit.get("gamma") is not None:
                    gammas.append(float(audit["gamma"]))
                if audit.get("control_count") is not None:
                    control_counts.append(int(audit["control_count"]))
                next_score_queries += int(audit.get("next_score_queries", 0))
                sequence_score_calls += int(audit.get("sequence_score_calls", 0))
        if case_accepted:
            cases_with_accept += 1

    return {
        "cases": len(rows),
        "blocks": blocks,
        "accepted_blocks": accepted,
        "block_accept_rate": accepted / blocks if blocks else 0.0,
        "cases_with_accepted_block": cases_with_accept,
        "case_accept_rate": cases_with_accept / len(rows),
        "selected_experts": dict(sorted(selected_experts.items())),
        "decision_reasons": dict(sorted(reasons.items())),
        "mean_real_margin": _safe_mean(real_margins),
        "mean_gamma": _safe_mean(gammas),
        "mean_control_count": _safe_mean(control_counts),
        "drift_probes": len(drift_js),
        "mean_same_prefix_js": _safe_mean(drift_js),
        "mean_greedy_disagreement_rate": _safe_mean(drift_disagreement),
        "next_score_queries": next_score_queries,
        "sequence_score_calls": sequence_score_calls,
        "wall_seconds": wall_seconds,
        "references_read": False,
        "correctness_interpretation": (
            "none; this is a label-free mechanism/cost summary"
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Summarize reference-free MERIT-Block mechanism traces. "
            "Input syntax is NAME=PATH and may be repeated for ablations."
        )
    )
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    payload = {
        "schema": "merit-block-mechanism-summary-v1",
        "references_read": False,
        "runs": {},
    }
    for value in args.run:
        if "=" not in value:
            raise ValueError("--run must use NAME=PATH")
        name, path = value.split("=", 1)
        name = name.strip()
        if not name or name in payload["runs"]:
            raise ValueError("run names must be unique and nonempty")
        payload["runs"][name] = summarize(_read(path))

    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(target)


if __name__ == "__main__":
    main()
