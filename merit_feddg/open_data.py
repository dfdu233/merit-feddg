"""Real free-text PathVQA preparation, and strict label-free inference manifests."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image

from .io import save_json
from .prepare import _image_key, _materialize_embedded_image, _parquet_rows

INFERENCE_FIELDS = {
    "id",
    "image",
    "question",
    "modality",
    "capability",
    "task",
    "domain",
    "domain_kind",
    "role",
    "group_id",
    "image_sha256",
}


def pixel_digest(path) -> str:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        return hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()


def read_manifest(path, role) -> list[dict]:
    path = Path(path)
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not rows:
        raise ValueError(f"empty {role} manifest")
    for row in rows:
        if set(row) != INFERENCE_FIELDS:
            raise ValueError(f"inference fields must be exactly {sorted(INFERENCE_FIELDS)}")
        if row["role"] != role or any(
            not isinstance(v, str) or not v.strip() for v in row.values()
        ):
            raise ValueError("invalid role or empty inference metadata")
        image = Path(row["image"])
        if not image.is_absolute():
            image = path.parent / image
        row["image"] = str(image.resolve())
        if pixel_digest(image) != row["image_sha256"]:
            raise ValueError(f"image content changed: {row['id']}")
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("duplicate sample IDs")
    return rows


def audit_open_split(source, target):
    for field in ("id", "image_sha256", "group_id"):
        if {r[field] for r in source} & {r[field] for r in target}:
            raise ValueError(f"source/target overlap: {field}")
    if {r["domain"] for r in source} & {r["domain"] for r in target}:
        raise ValueError("source/target domain names must be distinct")
    # One image/patient unit per domain avoids treating correlated QA rows as IID.
    for rows in (source, target):
        for field in ("group_id", "image_sha256"):
            if len({r[field] for r in rows}) != len(rows):
                raise ValueError(f"pilot requires one question per independent {field}")


def prepare_open_pathvqa(artifacts, output, source_per_group=16, target_limit=16):
    if min(source_per_group, target_limit) < 1:
        raise ValueError("positive sample limits required")
    snapshot = Path(artifacts) / "datasets/flaviagiammarino--path-vqa"
    output = Path(output)
    # Bounded reservoirs: keep smallest deterministic image+question hashes per group.
    selected = defaultdict(dict)
    eligible = defaultdict(int)
    for split, index, raw in _parquet_rows(snapshot):
        if split not in {"train", "test"}:
            continue
        answer, question = str(raw["answer"]).strip(), str(raw["question"]).strip()
        if not answer or not question or answer.casefold().strip(". ") in {"yes", "no"}:
            continue
        key = _image_key(raw, "image")
        group = (
            f"pathvqa-train-proxy-{int(key[:8], 16) % 2}" if split == "train" else "pathvqa-test"
        )
        eligible[group] += 1
        priority = hashlib.sha256(f"{key}:{question}".encode()).hexdigest()
        limit = source_per_group if split == "train" else target_limit
        current = selected[group].get(key)
        if current is None or priority < current[0]:
            selected[group][key] = (priority, split, index, raw)
        if len(selected[group]) > limit:
            # Image selection independent of answer wording or model correctness.
            del selected[group][max(selected[group])]
    rows, references = [], {}
    for group, members in sorted(selected.items()):
        for key, (_, split, index, raw) in sorted(members.items()):
            image = _materialize_embedded_image(raw, "image", snapshot, output / "images", key)
            digest = pixel_digest(image)
            sample_id = f"pathvqa-{split}-{index}"
            rows.append(
                {
                    "id": sample_id,
                    "image": str(image),
                    "question": str(raw["question"]),
                    "modality": "pathology",
                    "capability": "classification",
                    "task": "open_vqa",
                    "domain": group,
                    "domain_kind": "proxy",
                    "role": "source" if split == "train" else "target",
                    "group_id": digest,
                    "image_sha256": digest,
                }
            )
            references[sample_id] = [str(raw["answer"]).strip()]
    source = [r for r in rows if r["role"] == "source"]
    target = [r for r in rows if r["role"] == "target"]
    if not source or not target:
        raise ValueError("missing official train/test open questions")
    audit_open_split(source, target)
    output.mkdir(parents=True, exist_ok=True)
    for name, partition in (("source", source), ("target", target)):
        (output / f"{name}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in partition), encoding="utf-8"
        )
    save_json(output / "references.json", references)
    report = {
        "source": len(source),
        "target": len(target),
        "eligible": dict(eligible),
        "domain_generalization_evidence": False,
        "note": "Real PathVQA free-text subset; official train/test; train hash groups are NOT hospitals. "
        "Answer type is used only to exclude yes/no; no target-derived candidate vocabulary. "
        "One question/image. Validation is unused. Pixel/group split audit passed.",
    }
    save_json(output / "prepare-report.json", report)
    return report
