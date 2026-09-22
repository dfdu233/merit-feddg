from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

FORBIDDEN = {"answer", "answers", "reference", "references", "label", "labels"}


def _digest(seed, target_id, candidate_id):
    value = f"{seed}|{target_id}|{candidate_id}".encode()
    return hashlib.sha256(value).hexdigest()


def read_rows(path):
    rows = []
    ids = set()
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        if FORBIDDEN & set(row):
            raise ValueError(f"line {line_number}: label/reference fields are forbidden")
        required = ("id", "group_id", "modality", "task", "experts")
        missing = [key for key in required if key not in row]
        if missing:
            raise ValueError(f"line {line_number}: missing {missing}")
        if row["id"] in ids:
            raise ValueError("duplicate evidence packet ID")
        ids.add(row["id"])
        if not isinstance(row["experts"], dict):
            raise TypeError("experts must be a mapping")
        rows.append(row)
    if not rows:
        raise ValueError("real-evidence packet file is empty")
    return rows


def _same_fields(left, right, fields):
    return all(left.get(field) == right.get(field) for field in fields)


def _candidate_rows(rows, row, expert_id):
    return [
        other
        for other in rows
        if other["id"] != row["id"]
        and other["group_id"] != row["group_id"]
        and expert_id in other["experts"]
    ]


def build_controls(rows, *, count=5, seed=20260923, match_fields=("modality", "task")):
    if type(count) is not int or count < 1:
        raise ValueError("control count must be positive")
    if not match_fields:
        raise ValueError("at least one answer-blind match field is required")
    for field in match_fields:
        if field in FORBIDDEN:
            raise ValueError("control matching cannot use labels or references")

    output = []
    for row in rows:
        experts = {}
        for expert_id, real in sorted(row["experts"].items()):
            candidates = _candidate_rows(rows, row, expert_id)
            matched = [
                other for other in candidates
                if _same_fields(row, other, match_fields)
            ]
            matched.sort(key=lambda other: _digest(seed, row["id"], other["id"]))
            random = sorted(
                candidates,
                key=lambda other: _digest(seed + 1, row["id"], other["id"]),
            )
            matched = matched[:count]
            random = random[:count]
            matched_packets = [
                other["experts"][expert_id] for other in matched
            ]
            random_packets = [
                other["experts"][expert_id] for other in random
            ]
            shuffled_packet = (
                random_packets[0]
                if random_packets
                else matched_packets[0]
                if matched_packets
                else None
            )
            experts[expert_id] = {
                "real": real,
                "matched_controls": matched_packets,
                "random_controls": random_packets,
                "fixtures": (
                    {"shuffled": shuffled_packet}
                    if shuffled_packet is not None
                    else {}
                ),
                "control_provenance": {
                    "matched_ids": [other["id"] for other in matched],
                    "random_ids": [other["id"] for other in random],
                    "match_fields": list(match_fields),
                    "different_group_required": True,
                    "labels_used": False,
                    "seed": seed,
                },
            }
        output.append(
            {
                "id": row["id"],
                "experts": experts,
                "controls_built_without_labels": True,
            }
        )
    return output


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build label-blind wrong-patient expert-evidence controls for MERIT-Block. "
            "Input rows contain real EvidenceItem JSON packets only; references are rejected."
        )
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument(
        "--match-fields",
        nargs="+",
        default=["modality", "task"],
        help="Answer-blind metadata fields used for matched controls.",
    )
    args = parser.parse_args()
    rows = read_rows(args.input)
    payload = build_controls(
        rows,
        count=args.count,
        seed=args.seed,
        match_fields=tuple(args.match_fields),
    )
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in payload:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(target)


if __name__ == "__main__":
    main()
