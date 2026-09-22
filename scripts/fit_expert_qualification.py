"""Fit source-only MERIT expert qualification cards.

Input JSONL rows are paired source/development observations, never target-test
records. Required fields:
  expert_id, capability, scope, modality, task, claim_type, domain, group_id,
  outcome_delta, real_effect, knockoff_effect, support_direction

outcome_delta is the bounded score change (candidate transaction minus immutable
Generalist) in [-1, 1]. real_effect/knockoff_effect are label-free expert-native
margins measured with current-patient evidence and matched wrong-patient
controls. Qualification is action-conditional. Utility/harm are estimated only
on transactions carrying an absolute+counterfactual support certificate, while
support/veto precision are estimated
only on consequential source transactions (outcome_delta != 0). Neutral frozen
proposals therefore measure action prevalence/cost but are not mislabeled as
directional verification failures. The script stores conservative bounds; it
trains no gate.
"""

from __future__ import annotations

import argparse
import hashlib
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
                                   "real_effect", "knockoff_effect", "support_direction")
                   if key not in row]
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
        if type(row["support_direction"]) is not int or row["support_direction"] not in {-1, 0, 1}:
            raise ValueError(f"line {line_number}: support_direction must be -1/0/1")
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
        directions = [int(row["support_direction"]) for row in values]
        supported_rows = [
            row
            for row, direction in zip(values, directions, strict=True)
            if direction > 0
        ]
        supported = [float(row["outcome_delta"]) for row in supported_rows]
        support_domains = sorted({row["domain"] for row in supported_rows})
        support_help_n = sum(delta > 0 for delta in supported)
        support_harm_n = sum(delta < 0 for delta in supported)
        support_neutral_n = sum(delta == 0 for delta in supported)
        support_consequential_n = support_help_n + support_harm_n
        if supported:
            utility_lcb = mean_lcb(supported, z)
            harm_ucb = wilson(
                support_harm_n,
                len(supported),
                z,
                upper=True,
            )
        else:
            # No observed support action means no evidence that this expert can
            # safely authorize a commit under the frozen candidate distribution.
            utility_lcb = -1.0
            harm_ucb = 1.0
        support_precision_lcb = (
            wilson(
                support_help_n,
                support_consequential_n,
                z,
                upper=False,
            )
            if support_consequential_n
            else 0.0
        )

        veto_rows = [
            row
            for row, direction in zip(values, directions, strict=True)
            if direction < 0
        ]
        veto_deltas = [float(row["outcome_delta"]) for row in veto_rows]
        veto_domains = sorted({row["domain"] for row in veto_rows})
        veto_harm_n = sum(delta < 0 for delta in veto_deltas)
        veto_help_n = sum(delta > 0 for delta in veto_deltas)
        veto_neutral_n = sum(delta == 0 for delta in veto_deltas)
        veto_consequential_n = veto_harm_n + veto_help_n
        veto_precision_lcb = (
            wilson(
                veto_harm_n,
                veto_consequential_n,
                z,
                upper=False,
            )
            if veto_consequential_n
            else 0.0
        )
        consequential_pairs = [
            (delta, direction)
            for delta, direction in zip(deltas, directions, strict=True)
            if delta != 0 and direction != 0
        ]
        directional_successes = sum(
            (direction > 0 and delta > 0) or (direction < 0 and delta < 0)
            for delta, direction in consequential_pairs
        )
        directional_precision_lcb = (
            wilson(
                directional_successes,
                len(consequential_pairs),
                z,
                upper=False,
            )
            if consequential_pairs
            else 0.0
        )
        action_n = sum(direction != 0 for direction in directions)
        action_rate_lcb = (
            wilson(action_n, len(values), z, upper=False)
            if values
            else 0.0
        )
        card = dict(zip(KEYS, key, strict=True))
        card.update(
            n=len(values),
            domains=sorted({row["domain"] for row in values}),
            utility_lcb=utility_lcb,
            harm_ucb=harm_ucb,
            # Retained only as a v2-compatible audit field.  In v3 it is
            # conditional on consequential, nonzero decisions and is not an
            # authorization threshold.
            specificity_lcb=directional_precision_lcb,
            action_rate_lcb=action_rate_lcb,
            support_n=len(supported),
            support_domains=support_domains,
            support_consequential_n=support_consequential_n,
            support_help_n=support_help_n,
            support_harm_n=support_harm_n,
            support_neutral_n=support_neutral_n,
            support_precision_lcb=support_precision_lcb,
            veto_precision_lcb=veto_precision_lcb,
            veto_n=len(veto_deltas),
            veto_domains=veto_domains,
            veto_consequential_n=veto_consequential_n,
            veto_harm_n=veto_harm_n,
            veto_help_n=veto_help_n,
            veto_neutral_n=veto_neutral_n,
            source_only=True,
        )
        cards.append(card)
    source_group_ids = sorted({str(row["group_id"]) for row in rows})
    source_groups_sha256 = hashlib.sha256(
        json.dumps(source_group_ids, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()
    return {
        "schema": "merit-expert-qualification-v4",
        "source_only": True,
        "source_group_ids": source_group_ids,
        "source_groups_sha256": source_groups_sha256,
        "statistical_rule": {
            "utility": (
                "normal lower confidence bound on outcome_delta conditional on "
                "an absolute+counterfactual support certificate"
            ),
            "harm": (
                "Wilson upper confidence bound for outcome_delta < 0 conditional "
                "on an absolute+counterfactual support certificate"
            ),
            "action_rate": (
                "Wilson lower bound for a nonzero differential expert action; "
                "reported separately from action correctness"
            ),
            "support_precision": (
                "Wilson lower bound for beneficial outcome among consequential "
                "support-certificate actions; neutral outcomes are excluded"
            ),
            "certificate_rule": (
                "support requires real_margin>0 and strict dominance over every "
                "matched control margin; veto is the symmetric negative rule"
            ),
            "specificity_deprecated": (
                "directional precision on consequential certificate actions only; "
                "retained for audit and not used for authority"
            ),
            "veto_precision": (
                "Wilson lower bound for harmful outcome among consequential "
                "veto-certificate actions; neutral outcomes are excluded"
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
