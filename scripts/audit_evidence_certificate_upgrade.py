"""Replay v3 source observations under the v4 evidence-certificate rule.

This is a zero-inference diagnostic. It never changes a policy and never reads a
target/test row. Existing source observations already contain current-image
margins, matched-control margins and source-only transaction utility.

Legacy v3 action:
    support if D = real_margin - median(control_margin) > 0
    veto    if D < 0

v4 certificate:
    support iff real_margin > 0 and real_margin > max(control_margins)
    veto    iff real_margin < 0 and real_margin < min(control_margins)
    otherwise abstain

The report quantifies which legacy actions disappear and whether those removed
actions were helpful, harmful or score-neutral on frozen source data.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import median


def _read(path):
    rows = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        if str(row.get("split", "source")).casefold() in {
            "test",
            "target",
            "eval",
            "evaluation",
        }:
            raise ValueError(f"line {line_number}: target/test row is forbidden")
        required = (
            "expert_id",
            "real_effect",
            "knockoff_margins",
            "outcome_delta",
        )
        missing = [key for key in required if key not in row]
        if missing:
            raise ValueError(f"line {line_number}: missing {missing}")
        controls = tuple(float(value) for value in row["knockoff_margins"])
        if not controls:
            raise ValueError(f"line {line_number}: no matched controls")
        rows.append({**row, "knockoff_margins": controls})
    if not rows:
        raise ValueError("observation file is empty")
    return rows


def _direction_v4(real_margin, controls):
    if real_margin > 0 and real_margin > max(controls):
        return 1
    if real_margin < 0 and real_margin < min(controls):
        return -1
    return 0


def _outcome_name(value):
    if value > 0:
        return "helpful"
    if value < 0:
        return "harmful"
    return "neutral"


def _bucket():
    return defaultdict(int)


def audit(rows):
    by_expert = defaultdict(_bucket)
    total = _bucket()
    examples = []
    for row in rows:
        real = float(row["real_effect"])
        controls = tuple(row["knockoff_margins"])
        control_median = float(median(controls))
        differential = real - control_median
        legacy = 1 if differential > 0 else -1 if differential < 0 else 0
        certificate = _direction_v4(real, controls)
        outcome = float(row["outcome_delta"])
        outcome_name = _outcome_name(outcome)

        for counter in (total, by_expert[str(row["expert_id"])]):
            counter["n"] += 1
            counter[f"legacy_{'support' if legacy > 0 else 'veto' if legacy < 0 else 'abstain'}"] += 1
            counter[f"v4_{'support' if certificate > 0 else 'veto' if certificate < 0 else 'abstain'}"] += 1
            if legacy != certificate:
                counter["changed_action"] += 1
                counter[f"changed_{outcome_name}"] += 1
            if legacy > 0 and certificate == 0:
                counter["legacy_support_removed"] += 1
                counter[f"removed_support_{outcome_name}"] += 1
            if legacy < 0 and certificate == 0:
                counter["legacy_veto_removed"] += 1
                counter[f"removed_veto_{outcome_name}"] += 1
            if legacy > 0 and real <= 0:
                counter["relative_only_false_support_shape"] += 1
                counter[f"relative_only_false_support_{outcome_name}"] += 1
            if legacy < 0 and real >= 0:
                counter["relative_only_false_veto_shape"] += 1
                counter[f"relative_only_false_veto_{outcome_name}"] += 1

        if (
            len(examples) < 50
            and legacy != certificate
            and (legacy > 0 and real <= 0 or legacy < 0 and real >= 0)
        ):
            examples.append(
                {
                    "expert_id": row["expert_id"],
                    "group_id": row.get("group_id"),
                    "transaction_id": row.get("transaction_id"),
                    "real_margin": real,
                    "control_median": control_median,
                    "differential_effect": differential,
                    "legacy_direction": legacy,
                    "v4_direction": certificate,
                    "outcome_delta": outcome,
                }
            )

    return {
        "schema": "merit-tx-evidence-certificate-replay-v1",
        "source_only": True,
        "legacy_rule": "sign(real_margin - median(knockoff_margins))",
        "v4_rule": (
            "support: real_margin>0 and >all controls; "
            "veto: real_margin<0 and <all controls; else abstain"
        ),
        "overall": dict(total),
        "by_expert": {
            expert: dict(values)
            for expert, values in sorted(by_expert.items())
        },
        "relative_only_examples": examples,
        "target_test_selection": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    payload = audit(_read(args.input))
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rendered, encoding="utf-8")
        print(target)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
