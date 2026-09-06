"""Real non-binary multimodal VQA subsets, with explicit split provenance.

PathVQA keeps its official train/test roles. VQA-RAD is deliberately re-split
by RGB image identity: its QA splits must not be mistaken for image-disjoint
domains. Neither these hash groups nor image modalities represent hospitals.
"""

from __future__ import annotations

import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from .io import save_json
from .open_data import audit_open_split, pixel_digest, prepare_open_pathvqa, read_manifest
from .prepare import _image_payload, _materialize_embedded_image, _parquet_rows


def _digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _rgb_identity(raw, snapshot):
    path, payload = _image_payload(raw, "image")
    if payload is not None:
        image_source = io.BytesIO(payload)
    elif path:
        image_source = Path(path)
        if not image_source.is_absolute():
            image_source = snapshot / image_source
    else:
        raise ValueError("VQA-RAD row has no decodable image")
    with Image.open(image_source) as image:
        rgb = image.convert("RGB")
        return hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()


def _rad_group(image_sha256, seed):
    bucket = int(_digest(f"vqarad-image-split:{seed}:{image_sha256}")[:16], 16) % 5
    if bucket == 0:
        return "vqarad-image-target", "target"
    return f"vqarad-image-proxy-{0 if bucket in {1, 2} else 1}", "source"


def _prepare_vqarad(artifacts, output, source_per_group, target_limit, seed):
    snapshot = Path(artifacts) / "datasets/flaviagiammarino--vqa-rad"
    chosen, official_splits = {}, defaultdict(set)
    raw_counts, eligible_counts = Counter(), Counter()
    excluded_binary = invalid_text = 0
    for split, index, raw in _parquet_rows(snapshot):
        if split not in {"train", "test"}:
            continue
        raw_counts[split] += 1
        key = _rgb_identity(raw, snapshot)
        official_splits[key].add(split)
        question, answer = raw.get("question"), raw.get("answer")
        if not isinstance(question, str) or not isinstance(answer, str):
            invalid_text += 1
            continue
        question, answer = question.strip(), answer.strip()
        if not question or not answer:
            invalid_text += 1
            continue
        if answer.casefold().strip(" .!?,;:\t\r\n") in {"yes", "no"}:
            excluded_binary += 1
            continue
        eligible_counts[split] += 1
        # The answer is used only for the non-binary task filter. Neither image
        # assignment nor within-image question selection uses answer content.
        priority = (_digest(f"{seed}:{key}:{question}"), split, index)
        previous = chosen.get(key)
        if previous is None or priority < previous[0]:
            chosen[key] = (priority, split, index, raw, question, answer)

    groups = defaultdict(list)
    for key in chosen:
        group, role = _rad_group(key, seed)
        groups[(group, role)].append(key)
    rows, references, origins = [], {}, {}
    for (group, role), keys in sorted(groups.items()):
        limit = source_per_group if role == "source" else target_limit
        # Prefix-stable image selection when experiment limits increase.
        ordered = sorted(keys, key=lambda key: (_digest(f"{seed}:select:{key}"), key))
        for key in ordered[:limit]:
            _, split, index, raw, question, answer = chosen[key]
            image = _materialize_embedded_image(raw, "image", snapshot, output / "images", key)
            if pixel_digest(image) != key:
                raise ValueError("materialized VQA-RAD image does not match its RGB identity")
            sample_id = f"vqarad-{key[:20]}-{_digest(question)[:12]}"
            rows.append(
                {
                    "id": sample_id,
                    "image": str(image),
                    "question": question,
                    "modality": "mixed",
                    "capability": "classification",  # Legacy field, not a tool restriction.
                    "task": "open_vqa",
                    "domain": group,
                    "domain_kind": "proxy",
                    "role": role,
                    "group_id": key,
                    "image_sha256": key,
                }
            )
            references[sample_id] = [answer]
            origins[sample_id] = {
                "selected_question_original_split": split,
                "selected_question_original_index": index,
                "image_original_splits": sorted(official_splits[key]),
                "image_sha256": key,
            }
    expected = {"vqarad-image-proxy-0", "vqarad-image-proxy-1", "vqarad-image-target"}
    missing = sorted(expected - {row["domain"] for row in rows})
    if missing:
        raise ValueError(f"VQA-RAD non-binary image split has no examples in {missing}")
    report = {
        "dataset": "flaviagiammarino/vqa-rad",
        "split_protocol": "custom_rgb_image_disjoint_hash_resplit",
        "official_split_score": False,
        "seed": seed,
        "original_qa_rows_by_split": dict(raw_counts),
        "eligible_non_binary_qa_rows_by_original_split": dict(eligible_counts),
        "unique_rgb_images": len(official_splits),
        "eligible_unique_rgb_images": len(chosen),
        "images_shared_by_official_train_test": sum(
            {"train", "test"}.issubset(splits) for splits in official_splits.values()
        ),
        "excluded_yes_no_rows": excluded_binary,
        "excluded_invalid_text_rows": invalid_text,
        "eligible_images_by_group": {group: len(keys) for (group, _), keys in groups.items()},
        "selected_by_group": dict(Counter(row["domain"] for row in rows)),
        "image_assignment_uses_answers": False,
        "note": (
            "Official train/test QA rows are pooled before RGB-image grouping. One real non-yes/no "
            "question per RGB image; fixed hash allocation, not answer-balanced sampling. "
            "This is not an official VQA-RAD test score or an independent-hospital experiment."
        ),
    }
    return rows, references, origins, report


def _deduplicate_combined(rows):
    """Protect target images; drop source copies without moving any split roles."""
    retained, dropped, seen = [], [], {}
    for row in sorted(rows, key=lambda row: (row["role"] != "target", row["id"])):
        key = row["image_sha256"]
        if key in seen:
            dropped.append(
                {"id": row["id"], "role": row["role"], "kept_id": seen[key], "reason": "duplicate_rgb_image"}
            )
        else:
            seen[key] = row["id"]
            retained.append(row)
    return retained, dropped


def prepare_multimodal_vqa(
    artifacts,
    output,
    source_per_group=16,
    target_limit=16,
    datasets=("pathvqa", "vqarad"),
    seed=17,
):
    """Write strict source/target manifests and separate true reference answers.

    Limits apply per dataset (source limit per each of its two proxy groups).
    A combined run therefore requests at most 4 * source_per_group sources and
    2 * target_limit targets. Fewer eligible images are reported, never fabricated.
    """
    if min(source_per_group, target_limit) < 1:
        raise ValueError("positive sample limits required")
    if isinstance(datasets, str):
        datasets = tuple(part.strip() for part in datasets.split(",") if part.strip())
    datasets = tuple(datasets)
    if not datasets or len(set(datasets)) != len(datasets) or set(datasets) - {"pathvqa", "vqarad"}:
        raise ValueError("datasets must select pathvqa, vqarad, or both without duplicates")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows, references, reports, origins = [], {}, {}, {}
    if "pathvqa" in datasets:
        part_output = output / "pathvqa"
        reports["pathvqa"] = prepare_open_pathvqa(
            artifacts, part_output, source_per_group=source_per_group, target_limit=target_limit
        )
        part_rows = read_manifest(part_output / "source.jsonl", "source") + read_manifest(
            part_output / "target.jsonl", "target"
        )
        rows.extend({**row, "modality": "mixed"} for row in part_rows)
        references.update(json.loads((part_output / "references.json").read_text(encoding="utf-8")))
    if "vqarad" in datasets:
        part_rows, part_refs, origins, reports["vqarad"] = _prepare_vqarad(
            artifacts, output / "vqarad", source_per_group, target_limit, seed
        )
        rows.extend(part_rows)
        references.update(part_refs)
    rows, dropped = _deduplicate_combined(rows)
    source = sorted((row for row in rows if row["role"] == "source"), key=lambda row: row["id"])
    target = sorted((row for row in rows if row["role"] == "target"), key=lambda row: row["id"])
    if not source or not target:
        raise ValueError("multimodal preparation requires nonempty source and target image sets")
    audit_open_split(source, target)
    references = {row["id"]: references[row["id"]] for row in source + target}
    for name, partition in (("source", source), ("target", target)):
        (output / f"{name}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in partition), encoding="utf-8"
        )
    save_json(output / "references.json", references)
    save_json(output / "vqarad-origin.json", origins)
    report = {
        "source": len(source),
        "target": len(target),
        "datasets": reports,
        "requested_datasets": list(datasets),
        "limits_apply_per_dataset": True,
        "source_per_group": source_per_group,
        "target_limit_per_dataset": target_limit,
        "selected_by_domain": dict(Counter(row["domain"] for row in rows)),
        "cross_dataset_duplicate_images_removed": dropped,
        "domain_generalization_evidence": False,
        "real_domain_metadata": False,
        "split_audit_passed": True,
        "note": (
            "Real non-binary free-text QA only. All manifest modalities are mixed for image-only "
            "routing, never derived from reference answers. PathVQA retains official train/test roles; "
            "VQA-RAD uses a custom image-disjoint resplit. Hash groups are not hospitals. "
            "Model-inferred modality coverage must be reported after routing, not assumed from dataset names."
        ),
    }
    save_json(output / "prepare-report.json", report)
    return report
