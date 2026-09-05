from __future__ import annotations

import io
import json

import pytest
from PIL import Image

from merit_feddg.open_data import INFERENCE_FIELDS, prepare_open_pathvqa, read_manifest


def raw_row(number, answer="connective tissue"):
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), (number, number // 2, 17)).save(buffer, format="PNG")
    return {
        "question": "What is shown?",
        "answer": answer,
        "image": {"bytes": buffer.getvalue(), "path": f"image{number}.png"},
    }


def test_real_parquet_open_preparation_and_repeatability(tmp_path):
    arrow = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    root = tmp_path / "artifacts/datasets/flaviagiammarino--path-vqa/data"
    root.mkdir(parents=True)
    train = [raw_row(i) for i in range(32)] + [raw_row(240, "yes"), raw_row(241, "no")]
    # Additional questions for an image must not duplicate the independent unit.
    train += [{**train[0], "question": "Which type of tissue?"}]
    for split, rows in (
        ("train", train),
        ("test", [raw_row(i) for i in range(40, 50)]),
        ("validation", [raw_row(60)]),
    ):
        pq.write_table(arrow.Table.from_pylist(rows), root / f"{split}-00000.parquet")
    output = tmp_path / "data"
    report = prepare_open_pathvqa(tmp_path / "artifacts", output, 2, 2)
    assert (report["source"], report["target"]) == (4, 2)
    assert report["domain_generalization_evidence"] is False
    source = read_manifest(output / "source.jsonl", "source")
    target = read_manifest(output / "target.jsonl", "target")
    assert all(set(row) == INFERENCE_FIELDS for row in source + target)
    assert len({r["image_sha256"] for r in source + target}) == 6
    refs = json.loads((output / "references.json").read_text())
    assert all(values == ["connective tissue"] for values in refs.values())
    before = (output / "source.jsonl").read_bytes()
    prepare_open_pathvqa(tmp_path / "artifacts", output, 2, 2)
    assert before == (output / "source.jsonl").read_bytes()


def test_official_split_image_overlap_is_not_silently_allowed(tmp_path):
    arrow = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    root = tmp_path / "artifacts/datasets/flaviagiammarino--path-vqa"
    root.mkdir(parents=True)
    for split in ("train", "test"):
        pq.write_table(arrow.Table.from_pylist([raw_row(1)]), root / f"{split}-00000.parquet")
    with pytest.raises(ValueError, match="overlap"):
        prepare_open_pathvqa(tmp_path / "artifacts", tmp_path / "out", 2, 2)
