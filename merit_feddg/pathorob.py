from __future__ import annotations

import hashlib
import io
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from PIL import Image

DATASET_ID = "bifold-pathomics/PathoROB-tolkach_esca"
OFFICIAL_BIOLOGICAL_CLASSES = (
    "TUMOR",
    "MUSC_PROP",
    "SH_OES",
    "SH_MAG",
    "REGR_TU",
    "ADVENT",
)
REQUIRED_COLUMNS = {
    "image",
    "slide_id",
    "patch_id",
    "biological_class",
    "medical_center",
}


def _clean_text(value: object, field: str) -> str:
    text = str(value).strip() if value is not None else ""
    if not text:
        raise ValueError(f"PathoROB row has an empty {field}")
    return text


def _row_identity(row: Mapping[str, Any]) -> str:
    return "|".join(
        (
            _clean_text(row.get("medical_center"), "medical_center"),
            _clean_text(row.get("slide_id"), "slide_id"),
            _clean_text(row.get("patch_id"), "patch_id"),
        )
    )


def readable_class_name(value: object) -> str:
    """Turn a dataset-provided class value into a prompt-safe display label.

    The experiment validates a frozen official class vocabulary against the raw
    parquet data before sampling. These expansions make those task labels
    understandable to a VLM; unseen values remain generic in this helper.
    """

    raw = _clean_text(value, "biological_class")
    normalized = re.sub(r"[^0-9A-Za-z]+", " ", raw).strip().lower()
    exact = {
        "advent": "adventitia",
        "musc prop": "muscularis propria",
        "regr tu": "regressed tumor tissue",
        "sh oes": "oesophageal mucosa",
        "sh mag": "gastric mucosa",
        "tumor": "tumor tissue",
    }
    return exact.get(normalized, normalized)


def discover_pathorob_axes(rows: Iterable[Mapping[str, Any]]) -> tuple[list[str], list[str]]:
    """Discover the biological labels and real medical centers from row metadata."""

    classes: set[str] = set()
    centers: set[str] = set()
    for row in rows:
        classes.add(_clean_text(row.get("biological_class"), "biological_class"))
        centers.add(_clean_text(row.get("medical_center"), "medical_center"))
    ordered_classes = sorted(classes, key=lambda item: (item.casefold(), item))
    ordered_centers = sorted(centers, key=lambda item: (item.casefold(), item))
    if len(ordered_classes) < 3:
        raise ValueError(
            f"PathoROB multiclass preparation requires at least 3 classes; found "
            f"{len(ordered_classes)}"
        )
    if len(ordered_centers) < 2:
        raise ValueError(
            f"PathoROB LOCO preparation requires at least 2 medical centers; found "
            f"{len(ordered_centers)}"
        )
    return ordered_classes, ordered_centers


def audit_slide_centers(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Verify that a patient/slide group cannot cross a source-target boundary."""

    slide_centers: dict[str, str] = {}
    leaks: list[dict[str, str]] = []
    for row in rows:
        slide = _clean_text(row.get("slide_id"), "slide_id")
        center = _clean_text(row.get("medical_center"), "medical_center")
        previous = slide_centers.setdefault(slide, center)
        if previous != center:
            leaks.append({"slide_id": slide, "left": previous, "right": center})
    if leaks:
        preview = ", ".join(item["slide_id"] for item in leaks[:5])
        raise ValueError(f"slide leakage across medical centers: {preview}")
    return {"slide_count": len(slide_centers), "slide_leaks": leaks}


def stable_center_sample(
    rows: Iterable[Mapping[str, Any]],
    limit_per_center: int = 0,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Deterministically sample each center without reading target class labels."""

    if limit_per_center < 0:
        raise ValueError("limit_per_center must be non-negative")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    identities: set[str] = set()
    for raw in rows:
        row = dict(raw)
        center = _clean_text(row.get("medical_center"), "medical_center")
        identity = _row_identity(row)
        if identity in identities:
            raise ValueError(f"duplicate PathoROB patch identity: {identity}")
        identities.add(identity)
        groups[center].append(row)

    selected: list[dict[str, Any]] = []
    for key in sorted(groups):
        ranked = sorted(
            groups[key],
            key=lambda row: (
                hashlib.sha256(f"{seed}|{_row_identity(row)}".encode()).hexdigest(),
                _row_identity(row),
            ),
        )
        selected.extend(ranked[:limit_per_center] if limit_per_center else ranked)
    return sorted(selected, key=_row_identity)


def build_loco_rows(
    rows: Iterable[Mapping[str, Any]],
    held_out_center: str,
    image_paths: Mapping[str, str | Path],
    classes: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """Build a LOCO manifest; target labels are copied only as evaluation references."""

    material = [dict(row) for row in rows]
    observed_classes = {
        _clean_text(row.get("biological_class"), "biological_class") for row in material
    }
    centers = sorted(
        {_clean_text(row.get("medical_center"), "medical_center") for row in material},
        key=lambda item: (item.casefold(), item),
    )
    if len(centers) < 2:
        raise ValueError("PathoROB LOCO manifest requires at least two medical centers")
    ordered_classes = list(classes) if classes is not None else list(OFFICIAL_BIOLOGICAL_CLASSES)
    if not observed_classes <= set(ordered_classes):
        raise ValueError("sampled rows contain a class outside the frozen task vocabulary")
    if held_out_center not in centers:
        raise ValueError(f"unknown held-out medical center: {held_out_center}")

    candidates = [readable_class_name(value) for value in ordered_classes]
    if len(set(candidates)) != len(candidates):
        raise ValueError("human-readable class names are not unique")
    label_by_class = {value: index for index, value in enumerate(ordered_classes)}
    prompt = (
        "Classify this oesophageal histopathology patch into exactly one tissue category. "
        f"Choose one of: {', '.join(candidates)}. Answer with only the category."
    )

    output: list[dict[str, Any]] = []
    for row in sorted(material, key=_row_identity):
        identity = _row_identity(row)
        if identity not in image_paths:
            raise KeyError(f"no materialized image for {identity}")
        center = _clean_text(row.get("medical_center"), "medical_center")
        biological_class = _clean_text(row.get("biological_class"), "biological_class")
        slide_id = _clean_text(row.get("slide_id"), "slide_id")
        patch_id = _clean_text(row.get("patch_id"), "patch_id")
        sample_digest = hashlib.sha256(identity.encode()).hexdigest()[:20]
        output.append(
            {
                "id": f"pathorob-tolkach-esca-{sample_digest}",
                "image": str(Path(image_paths[identity]).resolve()),
                "domain": center,
                "modality": "pathology",
                "prompt": prompt,
                "candidates": candidates,
                "label": label_by_class[biological_class],
                "task_type": "multiclass",
                "metadata": {
                    "dataset": DATASET_ID,
                    "biological_class": biological_class,
                    "medical_center": center,
                    "slide_id": slide_id,
                    "patch_id": patch_id,
                    "split_role": "target" if center == held_out_center else "source",
                    "domain_kind": "real_medical_center",
                },
            }
        )
    return output


def read_pathorob_rows(snapshot: str | Path) -> list[dict[str, Any]]:
    """Stream the locally downloaded Hugging Face parquet snapshot with Arrow."""

    try:
        import pyarrow.dataset as pads
    except ImportError as exc:
        raise RuntimeError("PathoROB preparation requires pyarrow") from exc

    snapshot = Path(snapshot).resolve()
    files = sorted(
        path
        for path in snapshot.rglob("*.parquet")
        if ".cache" not in path.parts and path.is_file()
    )
    if not files:
        raise FileNotFoundError(f"no parquet payload found below {snapshot}")
    dataset = pads.dataset([str(path) for path in files], format="parquet")
    missing = REQUIRED_COLUMNS - set(dataset.schema.names)
    if missing:
        raise ValueError(f"PathoROB parquet is missing columns: {sorted(missing)}")
    rows: list[dict[str, Any]] = []
    scanner = dataset.scanner(columns=sorted(REQUIRED_COLUMNS), batch_size=256)
    for batch in scanner.to_batches():
        rows.extend(batch.to_pylist())
    return rows


def _image_source(row: Mapping[str, Any]) -> tuple[bytes | None, str | None]:
    value = row.get("image")
    if isinstance(value, Mapping):
        payload = value.get("bytes")
        path = value.get("path")
        return (bytes(payload) if payload is not None else None, str(path) if path else None)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value), None
    if isinstance(value, str):
        return None, value
    return None, None


def materialize_pathorob_images(
    rows: Iterable[Mapping[str, Any]], snapshot: str | Path, image_root: str | Path
) -> dict[str, Path]:
    """Materialize embedded bytes or snapshot-relative image paths as RGB PNG files."""

    snapshot = Path(snapshot).resolve()
    image_root = Path(image_root).resolve()
    image_root.mkdir(parents=True, exist_ok=True)
    result: dict[str, Path] = {}
    for row in rows:
        identity = _row_identity(row)
        destination = image_root / f"{hashlib.sha256(identity.encode()).hexdigest()}.png"
        if not destination.is_file():
            payload, path_text = _image_source(row)
            if payload is not None:
                source: Any = io.BytesIO(payload)
            elif path_text:
                direct = Path(path_text)
                candidates = (direct, snapshot / path_text)
                source = next((path for path in candidates if path.is_file()), None)
                if source is None:
                    matches = list(snapshot.rglob(Path(path_text).name))
                    if len(matches) != 1:
                        raise FileNotFoundError(f"cannot resolve PathoROB image {path_text!r}")
                    source = matches[0]
            else:
                raise ValueError(f"PathoROB row {_row_identity(row)} has no usable image")
            with Image.open(source) as image:
                image.convert("RGB").save(destination, format="PNG")
        result[identity] = destination
    return result


def _slug(value: str) -> str:
    slug = re.sub(r"[^0-9a-z]+", "-", value.casefold()).strip("-")
    return slug or hashlib.sha256(value.encode()).hexdigest()[:12]


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file() or path.read_text(encoding="utf-8") != content:
        path.write_text(content, encoding="utf-8")


def prepare_pathorob_loco(
    snapshot: str | Path,
    output_dir: str | Path,
    limit_per_center: int = 0,
    seed: int = 42,
) -> dict[str, Any]:
    """Prepare real medical-center LOCO manifests for PathoROB Tolkach ESCA.

    Labels are copied into manifests strictly for evaluation.  This function creates
    no routing, OOD, trust, or gating features from labels in the held-out center.
    """

    snapshot = Path(snapshot).resolve()
    output_dir = Path(output_dir).resolve()
    raw_rows = read_pathorob_rows(snapshot)
    discovered_classes, centers = discover_pathorob_axes(raw_rows)
    classes = list(OFFICIAL_BIOLOGICAL_CLASSES)
    if set(discovered_classes) != set(classes):
        raise ValueError(
            "PathoROB biological classes do not match the frozen official six-class taxonomy"
        )
    slide_audit = audit_slide_centers(raw_rows)
    sampled = stable_center_sample(raw_rows, limit_per_center, seed)
    image_paths = materialize_pathorob_images(sampled, snapshot, output_dir / "images")
    candidates = [readable_class_name(value) for value in classes]
    label_mapping = {
        value: {"index": index, "candidate": candidates[index]}
        for index, value in enumerate(classes)
    }

    canonical_rows = build_loco_rows(sampled, centers[0], image_paths, classes)
    for row in canonical_rows:
        row["metadata"]["split_role"] = "assigned_by_loco_runner"
    canonical_manifest = output_dir / "manifest.jsonl"
    _write_jsonl(canonical_manifest, canonical_rows)

    manifest_reports: list[dict[str, Any]] = []
    for held_out in centers:
        rows = build_loco_rows(sampled, held_out, image_paths, classes)
        manifest_path = output_dir / f"held-out-{_slug(held_out)}" / "manifest.jsonl"
        _write_jsonl(manifest_path, rows)
        counts = Counter(
            (row["metadata"]["split_role"], row["domain"], row["metadata"]["biological_class"])
            for row in rows
        )
        target_slides = {
            row["metadata"]["slide_id"] for row in rows if row["metadata"]["split_role"] == "target"
        }
        source_slides = {
            row["metadata"]["slide_id"] for row in rows if row["metadata"]["split_role"] == "source"
        }
        overlap = sorted(target_slides & source_slides)
        if overlap:
            raise RuntimeError(f"source-target slide leakage for {held_out}: {overlap[:5]}")
        manifest_reports.append(
            {
                "held_out_center": held_out,
                "manifest": str(manifest_path),
                "rows": len(rows),
                "source_rows": sum(role == "source" for role, _, _ in counts.elements()),
                "target_rows": sum(role == "target" for role, _, _ in counts.elements()),
                "counts": [
                    {
                        "split_role": key[0],
                        "medical_center": key[1],
                        "biological_class": key[2],
                        "count": count,
                    }
                    for key, count in sorted(counts.items())
                ],
                "source_target_slide_overlap": overlap,
            }
        )

    raw_counts = Counter(
        (
            _clean_text(row.get("medical_center"), "medical_center"),
            _clean_text(row.get("biological_class"), "biological_class"),
        )
        for row in raw_rows
    )
    sampled_counts = Counter(
        (
            _clean_text(row.get("medical_center"), "medical_center"),
            _clean_text(row.get("biological_class"), "biological_class"),
        )
        for row in sampled
    )
    report: dict[str, Any] = {
        "dataset": DATASET_ID,
        "protocol": "leave-one-real-medical-center-out",
        "domain_kind": "real_medical_center",
        "medical_center_domain_generalization_claim_allowed": True,
        "patient_level_clinical_claim_allowed": False,
        "seed": seed,
        "limit_per_center": limit_per_center,
        "sampling_uses_target_labels": False,
        "class_vocabulary_source": "frozen official PathoROB taxonomy",
        "classes": classes,
        "candidates": candidates,
        "label_mapping": label_mapping,
        "medical_centers": centers,
        "raw_rows": len(raw_rows),
        "sampled_rows": len(sampled),
        "raw_counts": [
            {"medical_center": key[0], "biological_class": key[1], "count": count}
            for key, count in sorted(raw_counts.items())
        ],
        "sampled_counts": [
            {"medical_center": key[0], "biological_class": key[1], "count": count}
            for key, count in sorted(sampled_counts.items())
        ],
        **slide_audit,
        "manifests": manifest_reports,
        "canonical_manifest": str(canonical_manifest),
        "target_labels_used_for_routing_or_gating": False,
    }
    report_path = output_dir / "prepare-report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report
