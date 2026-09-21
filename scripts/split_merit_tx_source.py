"""Create group/image-disjoint MERIT-Tx source splits without using labels.

The splitter is intended for the three statistically distinct v3 stages:
  qualification -> portfolio selection -> source canary.

Rows sharing a patient/study group OR identical image bytes are unioned before
assignment, so duplicated images under different IDs cannot leak across stages.
Assignment is deterministic from a user-declared seed and source-domain
signature. Optional frozen outputs/references are subset only *after* assignment
has been finalized; their contents never affect the split.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

_SPLITS = ("qualification", "portfolio", "canary")
_LABEL_KEYS = frozenset(
    {"answer", "answers", "label", "labels", "reference", "references", "ground_truth"}
)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path):
    rows = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        required = ("id", "image", "domain", "group_id")
        missing = [key for key in required if not str(row.get(key, "")).strip()]
        if missing:
            raise ValueError(f"manifest line {line_number}: missing {missing}")
        if _LABEL_KEYS.intersection(row):
            raise ValueError(
                f"manifest line {line_number}: labels/references are forbidden"
            )
        role = str(row.get("split", row.get("role", "source"))).casefold()
        if role not in {"source", "train", "training", "development", "dev"}:
            raise ValueError(f"manifest line {line_number}: target/test row is forbidden")
        rows.append(row)
    if not rows:
        raise ValueError("source manifest is empty")
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("source manifest IDs must be unique")
    return rows


class UnionFind:
    def __init__(self, values):
        self.parent = {value: value for value in values}

    def find(self, value):
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            nxt = self.parent[value]
            self.parent[value] = root
            value = nxt
        return root

    def union(self, left, right):
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def image_identity(path):
    value = Path(path).expanduser().resolve()
    if not value.is_file():
        raise FileNotFoundError(f"source image is missing: {value}")
    return file_sha256(value)


def build_units(rows):
    ids = [str(row["id"]) for row in rows]
    uf = UnionFind(ids)
    by_group = defaultdict(list)
    by_image = defaultdict(list)
    for row in rows:
        sample_id = str(row["id"])
        by_group[str(row["group_id"])].append(sample_id)
        by_image[image_identity(row["image"])].append(sample_id)
    for groups in (by_group, by_image):
        for members in groups.values():
            anchor = members[0]
            for member in members[1:]:
                uf.union(anchor, member)

    rows_by_root = defaultdict(list)
    for row in rows:
        rows_by_root[uf.find(str(row["id"]))].append(row)

    units = []
    for root, members in sorted(rows_by_root.items()):
        units.append(
            {
                "root": root,
                "rows": members,
                "sample_ids": sorted(str(row["id"]) for row in members),
                "group_ids": sorted({str(row["group_id"]) for row in members}),
                "image_sha256": sorted(
                    {image_identity(row["image"]) for row in members}
                ),
                "domains": sorted({str(row["domain"]) for row in members}),
            }
        )
    return units


def assign_units(units, *, seed, ratios):
    if len(ratios) != 3 or any(value <= 0 for value in ratios):
        raise ValueError("three positive split ratios are required")
    total = sum(ratios)
    normalized = [value / total for value in ratios]
    boundaries = (normalized[0], normalized[0] + normalized[1])
    assignment = {}
    for unit in units:
        identity = "|".join(
            (
                str(seed),
                ",".join(unit["domains"]),
                ",".join(unit["group_ids"]),
                ",".join(unit["image_sha256"]),
            )
        )
        digest = hashlib.sha256(identity.encode()).digest()
        value = int.from_bytes(digest[:8], "big") / 2**64
        split = (
            "qualification"
            if value < boundaries[0]
            else "portfolio"
            if value < boundaries[1]
            else "canary"
        )
        assignment[unit["root"]] = split
    return assignment


def load_mapping(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("outputs"), dict):
        payload = payload["outputs"]
    if not isinstance(payload, dict):
        raise TypeError(f"mapping must be a JSON object: {path}")
    return {str(key): value for key, value in payload.items()}


def write_json(path, payload):
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", default="merit-tx-v3-source-split")
    parser.add_argument(
        "--ratios",
        type=float,
        nargs=3,
        default=(0.5, 0.25, 0.25),
        metavar=("QUAL", "PORTFOLIO", "CANARY"),
    )
    parser.add_argument("--baseline")
    parser.add_argument("--candidate")
    parser.add_argument("--references")
    args = parser.parse_args()

    rows = load_manifest(args.manifest)
    units = build_units(rows)
    assignment = assign_units(units, seed=args.seed, ratios=args.ratios)
    split_by_id = {}
    for unit in units:
        split = assignment[unit["root"]]
        for sample_id in unit["sample_ids"]:
            split_by_id[sample_id] = split

    # Split identity is frozen before any optional mapping (including references)
    # is opened.
    root = Path(args.output)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty split directory: {root}")
    root.mkdir(parents=True, exist_ok=True)

    split_rows = {name: [] for name in _SPLITS}
    for row in rows:
        split_rows[split_by_id[str(row["id"])]].append(row)
    if any(not values for values in split_rows.values()):
        counts = {key: len(value) for key, value in split_rows.items()}
        raise ValueError(f"source split produced an empty stage: {counts}")

    for name in _SPLITS:
        target = root / name
        target.mkdir()
        with (target / "manifest.jsonl").open("w", encoding="utf-8") as handle:
            for row in split_rows[name]:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    optional = {
        "baseline": args.baseline,
        "candidate": args.candidate,
        "references": args.references,
    }
    for label, path in optional.items():
        if not path:
            continue
        mapping = load_mapping(path)
        manifest_ids = {str(row["id"]) for row in rows}
        if set(mapping) != manifest_ids:
            raise ValueError(f"{label} IDs must exactly match the unsplit source manifest")
        # This read occurs only after split_by_id is fully frozen.
        for name in _SPLITS:
            ids = [str(row["id"]) for row in split_rows[name]]
            write_json(root / name / f"{label}.json", {key: mapping[key] for key in ids})

    unit_counts = defaultdict(int)
    domain_counts = {name: defaultdict(int) for name in _SPLITS}
    split_groups = {name: set() for name in _SPLITS}
    split_images = {name: set() for name in _SPLITS}
    for unit in units:
        name = assignment[unit["root"]]
        unit_counts[name] += 1
        split_groups[name].update(unit["group_ids"])
        split_images[name].update(unit["image_sha256"])
        for domain in unit["domains"]:
            domain_counts[name][domain] += 1

    for left_index, left in enumerate(_SPLITS):
        for right in _SPLITS[left_index + 1 :]:
            if split_groups[left] & split_groups[right]:
                raise RuntimeError("group leakage across source stages")
            if split_images[left] & split_images[right]:
                raise RuntimeError("image leakage across source stages")

    audit = {
        "schema": "merit-tx-source-three-way-split-v1",
        "seed": args.seed,
        "ratios": list(args.ratios),
        "manifest": str(Path(args.manifest).resolve()),
        "manifest_sha256": file_sha256(args.manifest),
        "n_rows": len(rows),
        "n_units": len(units),
        "split_rows": {key: len(value) for key, value in split_rows.items()},
        "split_units": dict(unit_counts),
        "domains": {
            key: dict(sorted(value.items())) for key, value in domain_counts.items()
        },
        "group_overlap": 0,
        "image_sha256_overlap": 0,
        "assignment_uses_labels": False,
        "optional_mappings_loaded_after_assignment": True,
        "target_test_selection": False,
    }
    write_json(root / "audit.json", audit)
    print(root)


if __name__ == "__main__":
    main()
