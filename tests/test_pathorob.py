from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from PIL import Image

from merit_feddg.pathorob import (
    OFFICIAL_BIOLOGICAL_CLASSES,
    audit_slide_centers,
    build_loco_rows,
    discover_pathorob_axes,
    prepare_pathorob_loco,
    stable_center_sample,
)

CLASSES = ["TUMOR", "MUSC_PROP", "ADVENT", "REGR_TU", "SH_OES", "SH_MAG"]
CENTERS = ["CENTER_A", "CENTER_B", "CENTER_C"]


def _png_bytes(color: tuple[int, int, int]) -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (4, 4), color).save(stream, format="PNG")
    return stream.getvalue()


def _rows(image_as_struct: bool = True) -> list[dict]:
    rows = []
    for center_index, center in enumerate(CENTERS):
        for class_index, biological_class in enumerate(CLASSES):
            for patch_index in range(2):
                payload = _png_bytes((center_index * 40, class_index * 30, patch_index * 80))
                rows.append(
                    {
                        "image": {"bytes": payload, "path": None} if image_as_struct else payload,
                        "slide_id": f"slide-{center}-{class_index}",
                        "patch_id": f"patch-{class_index}-{patch_index}",
                        "biological_class": biological_class,
                        "medical_center": center,
                    }
                )
    return rows


def _write_parquet(snapshot: Path, rows: list[dict]) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")

    snapshot.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows), snapshot / "train-00000-of-00001.parquet")


def _read_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def test_axes_are_discovered_and_invalid_tasks_fail() -> None:
    classes, centers = discover_pathorob_axes(_rows())
    assert set(classes) == set(CLASSES)
    assert centers == CENTERS
    with pytest.raises(ValueError, match="at least 3 classes"):
        discover_pathorob_axes(_rows()[:2])
    with pytest.raises(ValueError, match="at least 2 medical centers"):
        discover_pathorob_axes([row for row in _rows() if row["medical_center"] == "CENTER_A"])


def test_sampling_is_stable_per_center_and_label_blind() -> None:
    forward = stable_center_sample(_rows(), limit_per_center=5, seed=17)
    reverse_rows = list(reversed(_rows()))
    for index, row in enumerate(reverse_rows):
        # Selection must be unchanged even if held-out labels are hidden or
        # replaced: only center and stable patch identity are admissible.
        row["biological_class"] = f"hidden-{index}"
    reverse = stable_center_sample(reverse_rows, limit_per_center=5, seed=17)
    identities = lambda rows: [
        (row["medical_center"], row["slide_id"], row["patch_id"]) for row in rows
    ]
    assert identities(forward) == identities(reverse)
    assert len(forward) == len(CENTERS) * 5
    assert all(sum(row["medical_center"] == center for row in forward) == 5 for center in CENTERS)


def test_slide_leakage_is_rejected() -> None:
    rows = _rows()
    rows[0]["slide_id"] = rows[-1]["slide_id"]
    with pytest.raises(ValueError, match="slide leakage"):
        audit_slide_centers(rows)


def test_build_loco_rows_preserves_real_domains_and_label_mapping(tmp_path: Path) -> None:
    rows = stable_center_sample(_rows(), 0, 42)
    image = tmp_path / "image.png"
    image.write_bytes(_png_bytes((1, 2, 3)))
    image_paths = {
        "|".join((row["medical_center"], row["slide_id"], row["patch_id"])): image for row in rows
    }
    manifest = build_loco_rows(rows, "CENTER_C", image_paths)
    assert {row["domain"] for row in manifest} == set(CENTERS)
    assert {row["task_type"] for row in manifest} == {"multiclass"}
    assert all(len(row["candidates"]) == 6 for row in manifest)
    assert all(row["metadata"]["domain_kind"] == "real_medical_center" for row in manifest)
    assert all(
        row["metadata"]["split_role"] == ("target" if row["domain"] == "CENTER_C" else "source")
        for row in manifest
    )
    class_to_label = {row["metadata"]["biological_class"]: row["label"] for row in manifest}
    assert len(class_to_label) == 6
    assert sorted(class_to_label.values()) == list(range(6))


def test_label_blind_pilot_may_miss_classes_without_changing_six_class_task(
    tmp_path: Path,
) -> None:
    rows = [row for row in _rows() if row["biological_class"] in {"ADVENT", "TUMOR"}]
    image = tmp_path / "image.png"
    image.write_bytes(_png_bytes((1, 2, 3)))
    image_paths = {
        "|".join((row["medical_center"], row["slide_id"], row["patch_id"])): image for row in rows
    }
    manifest = build_loco_rows(
        rows,
        "CENTER_C",
        image_paths,
        classes=OFFICIAL_BIOLOGICAL_CLASSES,
    )
    assert manifest
    assert all(len(row["candidates"]) == 6 for row in manifest)
    assert {row["metadata"]["biological_class"] for row in manifest} == {
        "ADVENT",
        "TUMOR",
    }


@pytest.mark.parametrize("image_as_struct", [True, False])
def test_prepare_writes_one_real_loco_manifest_per_center(
    tmp_path: Path, image_as_struct: bool
) -> None:
    snapshot = tmp_path / "artifacts" / "datasets" / "bifold-pathomics--PathoROB-tolkach_esca"
    _write_parquet(snapshot, _rows(image_as_struct))
    report = prepare_pathorob_loco(snapshot, tmp_path / "prepared", 12, seed=9)

    assert report["classes"] == list(OFFICIAL_BIOLOGICAL_CLASSES)
    assert report["medical_centers"] == CENTERS
    assert report["sampled_rows"] == 36
    assert report["sampling_uses_target_labels"] is False
    assert report["class_vocabulary_source"] == "frozen official PathoROB taxonomy"
    assert report["slide_leaks"] == []
    assert report["target_labels_used_for_routing_or_gating"] is False
    assert len(report["manifests"]) == 3
    canonical = _read_jsonl(report["canonical_manifest"])
    assert len(canonical) == 36
    assert {row["metadata"]["split_role"] for row in canonical} == {"assigned_by_loco_runner"}

    for entry in report["manifests"]:
        rows = _read_jsonl(entry["manifest"])
        assert len(rows) == 36
        assert entry["target_rows"] == 12
        assert entry["source_rows"] == 24
        assert entry["source_target_slide_overlap"] == []
        assert all(Path(row["image"]).is_file() for row in rows)
        target_slides = {
            row["metadata"]["slide_id"] for row in rows if row["metadata"]["split_role"] == "target"
        }
        source_slides = {
            row["metadata"]["slide_id"] for row in rows if row["metadata"]["split_role"] == "source"
        }
        assert not target_slides & source_slides
        for row in rows:
            mapping = report["label_mapping"][row["metadata"]["biological_class"]]
            assert row["label"] == mapping["index"]
            assert row["candidates"][row["label"]] == mapping["candidate"]
