from __future__ import annotations

import hashlib
import io
import json
from collections import Counter

import pytest
from PIL import Image

from merit_feddg import multimodal_data as data
from merit_feddg.open_data import INFERENCE_FIELDS, audit_open_split, pixel_digest, read_manifest


def raw_image(value, *, compression=6):
    image = Image.new("RGB", (8, 8), (value, value, value))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", compress_level=compression)
    return {"path": f"image-{value}.png", "bytes": buffer.getvalue()}


def rgb_key(value):
    image = Image.new("RGB", (8, 8), (value, value, value))
    return hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()


def rad_records(seed=17):
    records, counts = [], Counter()
    for value in range(250):
        group, _ = data._rad_group(rgb_key(value), seed)
        if counts[group] >= 6:
            continue
        counts[group] += 1
        # Deliberately shared train/test images encoded differently. Dedup must
        # compare RGB pixels, not payload bytes, file paths or QA row IDs.
        records.extend(
            [
                ("train", 2 * value, {
                    "image": raw_image(value, compression=0),
                    "question": f"Which organ is shown in case {value}?",
                    "answer": "lung",
                }),
                ("test", 2 * value + 1, {
                    "image": raw_image(value, compression=9),
                    "question": f"What abnormality is visible in case {value}?",
                    "answer": "effusion",
                }),
            ]
        )
        if len(counts) == 3 and min(counts.values()) == 6:
            break
    records.append(("train", 998, {
        "image": raw_image(254), "question": "Is there disease?", "answer": "Yes.",
    }))
    records.append(("test", 999, {
        "image": raw_image(254), "question": "Is there disease?", "answer": "NO!",
    }))
    return records


def test_vqarad_real_qa_is_resplit_by_rgb_image_not_original_qa_split(tmp_path, monkeypatch):
    records = rad_records()
    monkeypatch.setattr(data, "_parquet_rows", lambda _: iter(records))
    output = tmp_path / "prepared"
    report = data.prepare_multimodal_vqa(
        tmp_path, output, source_per_group=2, target_limit=3, datasets=("vqarad",)
    )
    source, target = read_manifest(output / "source.jsonl", "source"), read_manifest(
        output / "target.jsonl", "target"
    )
    audit_open_split(source, target)
    assert report["source"] == 4 and report["target"] == 3
    assert all(set(row) == INFERENCE_FIELDS and row["modality"] == "mixed" for row in source + target)
    assert all(row["domain_kind"] == "proxy" for row in source + target)
    refs = json.loads((output / "references.json").read_text())
    assert set(refs) == {row["id"] for row in source + target}
    assert all(answer[0] in {"lung", "effusion"} for answer in refs.values())
    rad = report["datasets"]["vqarad"]
    assert rad["images_shared_by_official_train_test"] == 19
    assert rad["eligible_unique_rgb_images"] == 18
    assert rad["excluded_yes_no_rows"] == 2
    assert not rad["official_split_score"] and not rad["image_assignment_uses_answers"]
    assert "custom" in rad["split_protocol"]
    origins = json.loads((output / "vqarad-origin.json").read_text())
    assert all(item["image_original_splits"] == ["test", "train"] for item in origins.values())
    assert report["split_audit_passed"] and not report["real_domain_metadata"]


def test_vqarad_assignments_ignore_answer_content_and_grow_with_limits(tmp_path, monkeypatch):
    records = rad_records()
    monkeypatch.setattr(data, "_parquet_rows", lambda _: iter(records))
    first_dir = tmp_path / "small"
    data.prepare_multimodal_vqa(tmp_path, first_dir, 1, 1, ("vqarad",))
    first = read_manifest(first_dir / "source.jsonl", "source") + read_manifest(
        first_dir / "target.jsonl", "target"
    )
    for _, _, raw in records:
        if raw["answer"] in {"lung", "effusion"}:
            raw["answer"] = "a completely different true-reference placeholder"
    second_dir = tmp_path / "larger"
    data.prepare_multimodal_vqa(tmp_path, second_dir, 3, 3, ("vqarad",))
    second = read_manifest(second_dir / "source.jsonl", "source") + read_manifest(
        second_dir / "target.jsonl", "target"
    )
    lookup = {row["id"]: row for row in second}
    assert all(row["id"] in lookup for row in first)
    assert all(row["domain"] == lookup[row["id"]]["domain"] for row in first)
    assert all(row["question"] == lookup[row["id"]]["question"] for row in first)
    assert len(second) == 9


@pytest.mark.parametrize("datasets", ["pathvqa", ("pathvqa", "vqarad")])
def test_pathvqa_wrapper_preserves_official_roles_and_marks_mixed(tmp_path, monkeypatch, datasets):
    def prepare(artifacts, output, source_per_group, target_limit):
        assert source_per_group == 2 and target_limit == 3
        output.mkdir(parents=True)
        references = {}
        for split, role, value in (("train", "source", 250), ("test", "target", 251)):
            image = output / f"{split}.png"
            Image.new("RGB", (8, 8), (value, value, value)).save(image)
            key = pixel_digest(image)
            row = {
                "id": f"pathvqa-{split}-0", "image": str(image.resolve()),
                "question": "Which organ is shown?", "modality": "pathology",
                "capability": "classification", "task": "open_vqa",
                "domain": f"pathvqa-{split}", "domain_kind": "proxy", "role": role,
                "group_id": key, "image_sha256": key,
            }
            (output / f"{role}.jsonl").write_text(json.dumps(row) + "\n")
            references[row["id"]] = ["heart"]
        (output / "references.json").write_text(json.dumps(references))
        return {"source": 1, "target": 1, "official_train_test": True}

    monkeypatch.setattr(data, "prepare_open_pathvqa", prepare)
    monkeypatch.setattr(data, "_parquet_rows", lambda _: iter(rad_records()))
    output = tmp_path / "prepared"
    report = data.prepare_multimodal_vqa(tmp_path, output, 2, 3, datasets)
    assert report["datasets"]["pathvqa"]["official_train_test"]
    source, target = read_manifest(output / "source.jsonl", "source"), read_manifest(
        output / "target.jsonl", "target"
    )
    assert source[0]["id"] == "pathvqa-train-0"
    assert target[0]["id"] == "pathvqa-test-0"
    assert source[0]["modality"] == target[0]["modality"] == "mixed"
    assert report["source"] == (1 if datasets == "pathvqa" else 5)
    assert report["target"] == (1 if datasets == "pathvqa" else 4)


def test_combined_dedup_protects_target_without_reassigning_source():
    rows = [
        {"id": "a-source", "role": "source", "image_sha256": "shared"},
        {"id": "z-target", "role": "target", "image_sha256": "shared"},
        {"id": "b-source", "role": "source", "image_sha256": "source-only"},
        {"id": "c-source", "role": "source", "image_sha256": "source-only"},
    ]
    kept, dropped = data._deduplicate_combined(rows)
    assert {row["id"] for row in kept} == {"z-target", "b-source"}
    assert {row["id"] for row in dropped} == {"a-source", "c-source"}
    assert next(row for row in dropped if row["id"] == "a-source")["kept_id"] == "z-target"


@pytest.mark.parametrize("datasets", [(), ("unknown",), ("vqarad", "vqarad")])
def test_invalid_dataset_selection_fails_before_download_or_preparation(tmp_path, datasets):
    with pytest.raises(ValueError, match="datasets"):
        data.prepare_multimodal_vqa(tmp_path, tmp_path / "out", datasets=datasets)


def test_missing_nonbinary_source_or_target_group_fails(tmp_path, monkeypatch):
    only_binary = [("train", 0, {"image": raw_image(1), "question": "Normal?", "answer": "yes"})]
    monkeypatch.setattr(data, "_parquet_rows", lambda _: iter(only_binary))
    with pytest.raises(ValueError, match="no examples"):
        data.prepare_multimodal_vqa(tmp_path, tmp_path / "out", datasets=("vqarad",))
