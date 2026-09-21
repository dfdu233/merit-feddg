"""Fit source-only MERIT expert qualification cards.

Input JSONL rows are paired source/development observations, never target-test
records. Required fields:
  expert_id, capability, scope, modality, task, claim_type, domain, group_id,
  outcome_delta, real_effect, knockoff_effect, candidate_method,
  expert_provenance_fingerprint

outcome_delta is the bounded score change (expert transaction minus immutable
Generalist) in [-1, 1]. real_effect/knockoff_effect are label-free receiver
effects measured with current-patient evidence and a matched wrong-patient
control. The script stores conservative lower/upper bounds; it trains no gate.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

KEYS = (
    "expert_id",
    "capability",
    "scope",
    "modality",
    "task",
    "claim_type",
)


def wilson(successes, n, z, *, upper):
    if n < 1:
        raise ValueError("Wilson bound requires n >= 1")
    p = successes / n
    z2 = z * z
    center = (p + z2 / (2 * n)) / (1 + z2 / n)
    radius = (
        z
        * math.sqrt((p * (1 - p) + z2 / (4 * n)) / n)
        / (1 + z2 / n)
    )
    return min(1.0, center + radius) if upper else max(0.0, center - radius)


def mean_lcb(values, z):
    if not values:
        raise ValueError("mean bound requires observations")
    mean = sum(values) / len(values)
    if len(values) == 1:
        return -1.0
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return max(-1.0, mean - z * math.sqrt(variance / len(values)))


def read_rows(path):
    rows = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        missing = [
            key
            for key in (
                *KEYS,
                "domain",
                "group_id",
                "outcome_delta",
                "real_effect",
                "knockoff_effect",
                "candidate_method",
                "expert_provenance_fingerprint",
            )
            if key not in row
        ]
        if missing:
            raise ValueError(f"line {line_number}: missing {missing}")
        if row.get("split", "source") not in {"source", "train", "development", "dev"}:
            raise ValueError(f"line {line_number}: target/test rows are forbidden")
        for key in KEYS[:6]:
            if not isinstance(row[key], str) or not row[key].strip():
                raise ValueError(f"line {line_number}: invalid {key}")
        if not isinstance(row["domain"], str) or not row["domain"].strip():
            raise ValueError(f"line {line_number}: invalid source domain")
        if not isinstance(row["candidate_method"], str) or not row["candidate_method"].strip():
            raise ValueError(f"line {line_number}: invalid candidate_method")
        if (
            not isinstance(row["expert_provenance_fingerprint"], str)
            or not row["expert_provenance_fingerprint"].strip()
        ):
            raise ValueError(
                f"line {line_number}: invalid expert_provenance_fingerprint"
            )
        if not isinstance(row["group_id"], str) or not row["group_id"].strip():
            raise ValueError(f"line {line_number}: invalid group_id")
        for key in ("outcome_delta", "real_effect", "knockoff_effect"):
            value = row[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"line {line_number}: invalid {key}")
        if not -1 <= float(row["outcome_delta"]) <= 1:
            raise ValueError(f"line {line_number}: outcome_delta must be in [-1,1]")
        rows.append(row)
    if not rows:
        raise ValueError("qualification input is empty")
    return rows


def fit(rows, *, z=1.96):
    proposal_policies = {str(row["candidate_method"]).strip() for row in rows}
    if len(proposal_policies) != 1:
        raise ValueError(
            "qualification input must contain exactly one frozen candidate_method"
        )
    proposal_policy = next(iter(proposal_policies))
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in KEYS)].append(row)
    cards = []
    for key in sorted(grouped):
        values = grouped[key]
        # One patient/image group must not appear twice inside one card.
        groups = [row["group_id"] for row in values]
        if len(groups) != len(set(groups)):
            raise ValueError(f"duplicate group_id within qualification cell: {key}")
        provenance = {
            str(row["expert_provenance_fingerprint"]).strip()
            for row in values
        }
        if len(provenance) != 1:
            raise ValueError(
                f"mixed expert provenance within qualification cell: {key}"
            )
        deltas = [float(row["outcome_delta"]) for row in values]
        harms = sum(value < 0 for value in deltas)
        specificity = sum(
            float(row["real_effect"]) > float(row["knockoff_effect"])
            for row in values
        )
        card = dict(zip(KEYS, key, strict=True))
        card.update(
            n=len(values),
            domains=sorted({row["domain"] for row in values}),
            utility_lcb=mean_lcb(deltas, z),
            harm_ucb=wilson(harms, len(values), z, upper=True),
            specificity_lcb=wilson(specificity, len(values), z, upper=False),
            expert_provenance_fingerprint=next(iter(provenance)),
            source_only=True,
        )
        cards.append(card)
    return {
        "schema": "merit-expert-qualification-v2",
        "source_only": True,
        "proposal_policy": proposal_policy,
        "statistical_rule": {
            "utility": "normal lower confidence bound on bounded outcome_delta",
            "harm": "Wilson upper confidence bound for outcome_delta < 0",
            "specificity": "Wilson lower confidence bound for real_effect > knockoff_effect",
            "z": z,
        },
        "cards": cards,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--z", type=float, default=1.96)
    args = parser.parse_args()
    if not math.isfinite(args.z) or args.z <= 0:
        parser.error("--z must be positive and finite")
    payload = fit(read_rows(args.input), z=args.z)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
