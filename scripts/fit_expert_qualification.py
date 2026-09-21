"""Fit source-only MERIT expert qualification cards.

Input JSONL rows are paired source/development observations, never target-test
records. Required fields:
  expert_id, capability, scope, modality, task, claim_type, domain, group_id,
  outcome_delta, real_effect, knockoff_effect

outcome_delta is the bounded score change (candidate transaction minus immutable
Generalist) in [-1, 1]. real_effect/knockoff_effect are label-free expert-native
margins measured with current-patient evidence and matched wrong-patient
controls. Qualification is action-conditional: utility/harm are estimated only
on transactions the expert would support (D_e > 0), while specificity measures
whether the sign of D_e agrees with the sign of source-only transaction utility.
The script stores conservative lower/upper bounds; it trains no gate.
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
        missing = [key for key in (*KEYS, "domain", "group_id", "outcome_delta",
                                   "real_effect", "knockoff_effect") if key not in row]
        if missing:
            raise ValueError(f"line {line_number}: missing {missing}")
        if row.get("split", "source") not in {"source", "train", "development", "dev"}:
            raise ValueError(f"line {line_number}: target/test rows are forbidden")
        for key in KEYS[:6]:
            if not isinstance(row[key], str) or not row[key].strip():
                raise ValueError(f"line {line_number}: invalid {key}")
        if not isinstance(row["domain"], str) or not row["domain"].strip():
            raise ValueError(f"line {line_number}: invalid source domain")
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
        deltas = [float(row["outcome_delta"]) for row in values]
        effects = [
            float(row["real_effect"]) - float(row["knockoff_effect"])
            for row in values
        ]
        supported_rows = [
            row
            for row, effect in zip(values, effects, strict=True)
            if effect > 0
        ]
        supported = [float(row["outcome_delta"]) for row in supported_rows]
        support_domains = sorted({row["domain"] for row in supported_rows})
        if supported:
            utility_lcb = mean_lcb(supported, z)
            harm_ucb = wilson(
                sum(delta < 0 for delta in supported),
                len(supported),
                z,
                upper=True,
            )
        else:
            # No observed support action means no evidence that this expert can
            # safely authorize a commit under the frozen candidate distribution.
            utility_lcb = -1.0
            harm_ucb = 1.0

        veto_rows = [
            row
            for row, effect in zip(values, effects, strict=True)
            if effect < 0
        ]
        veto_deltas = [float(row["outcome_delta"]) for row in veto_rows]
        veto_domains = sorted({row["domain"] for row in veto_rows})
        veto_precision_lcb = (
            wilson(
                sum(delta < 0 for delta in veto_deltas),
                len(veto_deltas),
                z,
                upper=False,
            )
            if veto_deltas
            else 0.0
        )
        directional_successes = sum(
            (effect > 0 and delta > 0) or (effect < 0 and delta < 0)
            for delta, effect in zip(deltas, effects, strict=True)
        )
        card = dict(zip(KEYS, key, strict=True))
        card.update(
            n=len(values),
            domains=sorted({row["domain"] for row in values}),
            utility_lcb=utility_lcb,
            harm_ucb=harm_ucb,
            specificity_lcb=wilson(
                directional_successes,
                len(values),
                z,
                upper=False,
            ),
            support_n=len(supported),
            support_domains=support_domains,
            veto_precision_lcb=veto_precision_lcb,
            veto_n=len(veto_deltas),
            veto_domains=veto_domains,
            source_only=True,
        )
        cards.append(card)
    return {
        "schema": "merit-expert-qualification-v1",
        "source_only": True,
        "statistical_rule": {
            "utility": (
                "normal lower confidence bound on outcome_delta conditional on "
                "signed differential effect D_e > 0"
            ),
            "harm": (
                "Wilson upper confidence bound for outcome_delta < 0 conditional "
                "on D_e > 0"
            ),
            "specificity": (
                "Wilson lower confidence bound for sign(D_e) agreeing with "
                "sign(outcome_delta); zero utility is conservatively not success"
            ),
            "veto_precision": (
                "Wilson lower confidence bound for outcome_delta < 0 "
                "conditional on D_e < 0"
            ),
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
