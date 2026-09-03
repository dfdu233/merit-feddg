from __future__ import annotations

import hashlib
import io
import json
import zipfile
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from PIL import Image

from .io import load_yaml, save_json, save_yaml


def _slug(value: str) -> str:
    return value.lower().replace("/", "_").replace("-", "_")


def _write_text_if_changed(path: Path, content: str) -> bool:
    """Write deterministic output only when content changed, preserving cache mtimes."""

    if path.is_file() and path.read_text(encoding="utf-8") == content:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return True


def _normal_text(value: object) -> str:
    return " ".join(str(value).strip().lower().split())


def _image_payload(row: dict[str, Any], field: str) -> tuple[str | None, bytes | None]:
    value = row.get(field)
    if isinstance(value, dict):
        raw = value.get("bytes")
        return value.get("path"), bytes(raw) if raw is not None else None
    if isinstance(value, (bytes, bytearray, memoryview)):
        return None, bytes(value)
    if isinstance(value, str):
        return value, None
    return None, None


def _image_key(row: dict[str, Any], field: str) -> str:
    path, payload = _image_payload(row, field)
    if payload is not None:
        return hashlib.sha256(payload).hexdigest()
    if path:
        return hashlib.sha256(path.replace("\\", "/").lower().encode()).hexdigest()
    for key in ("img_id", "img_name", "image_id", "id"):
        if row.get(key) is not None:
            return hashlib.sha256(f"{key}:{row[key]}".encode()).hexdigest()
    raise ValueError(f"row has no usable image identity in field {field!r}")


def proxy_domain(dataset: str, image_key: str) -> str:
    """Assign images, rather than QA rows, to deterministic non-clinical proxy domains."""

    bucket = int(image_key[:8], 16) % 5
    suffix = "target" if bucket == 0 else "source_a" if bucket in {1, 2} else "source_b"
    return f"{dataset}-{suffix}"


def _infer_explicit_modality(question: object, answer: object) -> str | None:
    question_text = _normal_text(question)
    answer_text = _normal_text(answer).replace("_", "-")
    direct_question = any(
        phrase in question_text
        for phrase in (
            "what modality",
            "which modality",
            "imaging modality",
            "what type of image",
            "how was this image taken",
            "what imaging modality",
        )
    )
    if not direct_question:
        return None
    if any(token in answer_text for token in ("x-ray", "xray", "radiograph")):
        return "cxr"
    if "mri" in answer_text or answer_text.startswith("mr-"):
        return "mri"
    if answer_text == "ct" or answer_text.startswith("ct "):
        return "ct"
    return None


def _split_from_parquet(path: Path) -> str:
    return path.name.split("-", 1)[0].lower()


def _parquet_rows(snapshot: Path, batch_size: int = 16) -> Iterator[tuple[str, int, dict]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("prepare-public requires the research dependencies (pyarrow)") from exc

    files = sorted(
        path for path in snapshot.rglob("*.parquet") if ".cache" not in path.parts
    )
    if not files:
        raise FileNotFoundError(f"no parquet payload found below {snapshot}")
    counters: dict[str, int] = defaultdict(int)
    for path in files:
        split = _split_from_parquet(path)
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=batch_size):
            for row in batch.to_pylist():
                index = counters[split]
                counters[split] += 1
                yield split, index, row


def _json_rows(snapshot: Path) -> Iterator[tuple[str, int, dict]]:
    files = [snapshot / f"{split}.json" for split in ("train", "validation", "test")]
    if not all(path.is_file() for path in files):
        raise FileNotFoundError(f"expected train/validation/test JSON files below {snapshot}")
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise TypeError(f"{path} must contain a JSON list")
        for index, row in enumerate(payload):
            yield path.stem, index, row


def _materialize_embedded_image(
    row: dict,
    field: str,
    snapshot: Path,
    image_root: Path,
    image_key: str,
) -> Path:
    path_text, payload = _image_payload(row, field)
    suffix = Path(path_text or "image.png").suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
        suffix = ".png"
    destination = image_root / f"{image_key}{suffix}"
    if destination.is_file():
        return destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if payload is not None:
        destination.write_bytes(payload)
        return destination.resolve()
    if path_text:
        direct = Path(path_text)
        candidates = [direct, snapshot / path_text]
        candidates.extend(snapshot.rglob(Path(path_text).name))
        source = next((candidate for candidate in candidates if candidate.is_file()), None)
        if source is not None:
            with Image.open(source) as image:
                image.convert("RGB").save(destination)
            return destination.resolve()
    raise FileNotFoundError("selected row does not contain image bytes or a resolvable image path")


class _SlakeArchive:
    def __init__(self, snapshot: Path) -> None:
        archive = snapshot / "imgs.zip"
        if not archive.is_file():
            raise FileNotFoundError(f"missing SLAKE image archive: {archive}")
        self.archive = zipfile.ZipFile(archive)
        self.members = {
            name.replace("\\", "/").lstrip("./").lower(): name
            for name in self.archive.namelist()
            if not name.endswith("/")
        }

    def close(self) -> None:
        self.archive.close()

    def materialize(self, reference: str, image_root: Path, image_key: str) -> Path:
        normalized = reference.replace("\\", "/").lstrip("./").lower()
        member = self.members.get(normalized)
        if member is None:
            matches = [value for key, value in self.members.items() if key.endswith("/" + normalized)]
            if len(matches) != 1:
                raise FileNotFoundError(f"cannot resolve SLAKE image {reference!r} in imgs.zip")
            member = matches[0]
        suffix = Path(member).suffix.lower() or ".jpg"
        destination = image_root / f"{image_key}{suffix}"
        if not destination.is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(self.archive.read(member))
        return destination.resolve()


def _closed_candidates(answer: object) -> tuple[list[str], int] | None:
    normalized = _normal_text(answer)
    if normalized == "yes":
        return ["yes", "no"], 0
    if normalized == "no":
        return ["yes", "no"], 1
    return None


def _oct_candidates(answer: object, configured: list[str]) -> tuple[list[str], int] | None:
    aliases = {
        "normal": "normal",
        "drusen": "drusen",
        "dme": "diabetic macular edema",
        "diabetic macular edema": "diabetic macular edema",
        "cnv": "choroidal neovascularization",
        "choroidal neovascularization": "choroidal neovascularization",
    }
    normalized = aliases.get(_normal_text(answer))
    choices = [_normal_text(value) for value in configured]
    if normalized not in choices:
        return None
    return choices, choices.index(normalized)


def _enough(counts: dict[str, int], expected: set[str], limit: int) -> bool:
    return bool(limit) and all(counts.get(domain, 0) >= limit for domain in expected)


def _append_manifest_row(
    rows: list[dict],
    counts: dict[str, int],
    seen_questions: dict[str, int],
    spec: dict,
    split: str,
    index: int,
    raw: dict,
    image_key: str,
    image_path: Path,
    candidates: list[str],
    label: int,
    prompt: str,
    modality: str,
    limit_per_domain: int,
    questions_per_image: int,
) -> None:
    dataset = spec["name"]
    domain = proxy_domain(dataset, image_key)
    if limit_per_domain and counts[domain] >= limit_per_domain:
        return
    if questions_per_image and seen_questions[image_key] >= questions_per_image:
        return
    seen_questions[image_key] += 1
    counts[domain] += 1
    rows.append(
        {
            "id": f"{dataset}-{image_key[:12]}-{split}-{index}",
            "image": str(image_path),
            "domain": domain,
            "modality": modality,
            "prompt": prompt,
            "candidates": candidates,
            "label": label,
            "metadata": {
                "dataset": spec["id"],
                "original_split": split,
                "image_key": image_key,
                "domain_kind": "deterministic_proxy_partition",
            },
        }
    )


def _prepare_parquet_dataset(
    spec: dict,
    snapshot: Path,
    image_root: Path,
    limit_per_domain: int,
    questions_per_image: int,
    oct_labels: list[str],
) -> tuple[list[dict], dict[str, int]]:
    rows: list[dict] = []
    counts: dict[str, int] = defaultdict(int)
    seen_questions: dict[str, int] = defaultdict(int)
    expected = {f"{spec['name']}-{suffix}" for suffix in ("source_a", "source_b", "target")}
    modality_by_image: dict[str, str] = {}
    if spec["adapter"] == "vqa_rad":
        for _, _, raw in _parquet_rows(snapshot):
            inferred = _infer_explicit_modality(raw.get("question"), raw.get("answer"))
            if inferred:
                modality_by_image[_image_key(raw, spec["image_field"])] = inferred

    for split, index, raw in _parquet_rows(snapshot):
        image_key = _image_key(raw, spec["image_field"])
        adapter = spec["adapter"]
        if adapter == "vqa_rad" and modality_by_image.get(image_key) != "cxr":
            continue
        if adapter == "oct":
            choice = _oct_candidates(raw.get(spec["answer_field"]), oct_labels)
            prompt = spec["prompt"]
        else:
            choice = _closed_candidates(raw.get(spec["answer_field"]))
            prompt = str(raw.get(spec["question_field"], "")).strip()
        if choice is None or not prompt:
            continue
        domain = proxy_domain(spec["name"], image_key)
        if limit_per_domain and counts[domain] >= limit_per_domain:
            if _enough(counts, expected, limit_per_domain):
                break
            continue
        if questions_per_image and seen_questions[image_key] >= questions_per_image:
            continue
        image_path = _materialize_embedded_image(
            raw, spec["image_field"], snapshot, image_root, image_key
        )
        _append_manifest_row(
            rows,
            counts,
            seen_questions,
            spec,
            split,
            index,
            raw,
            image_key,
            image_path,
            choice[0],
            choice[1],
            prompt,
            spec["modality"],
            limit_per_domain,
            questions_per_image,
        )
        if _enough(counts, expected, limit_per_domain):
            break
    return rows, dict(counts)


def _prepare_slake(
    spec: dict,
    snapshot: Path,
    image_root: Path,
    limit_per_domain: int,
    questions_per_image: int,
) -> tuple[list[dict], dict[str, int]]:
    rows: list[dict] = []
    counts: dict[str, int] = defaultdict(int)
    seen_questions: dict[str, int] = defaultdict(int)
    expected = {f"{spec['name']}-{suffix}" for suffix in ("source_a", "source_b", "target")}
    archive = _SlakeArchive(snapshot)
    try:
        for split, index, raw in _json_rows(snapshot):
            language = _normal_text(raw.get("q_lang", "en"))
            modality = _normal_text(raw.get("modality", "")).replace("-", "")
            if language not in {"en", "english"} or modality not in {"xray", "radiograph"}:
                continue
            choice = _closed_candidates(raw.get("answer"))
            prompt = str(raw.get("question", "")).strip()
            if choice is None or not prompt:
                continue
            reference = str(raw.get("img_name", ""))
            image_key = _image_key({"img_name": reference}, "image")
            domain = proxy_domain(spec["name"], image_key)
            if limit_per_domain and counts[domain] >= limit_per_domain:
                if _enough(counts, expected, limit_per_domain):
                    break
                continue
            if questions_per_image and seen_questions[image_key] >= questions_per_image:
                continue
            image_path = archive.materialize(reference, image_root, image_key)
            _append_manifest_row(
                rows,
                counts,
                seen_questions,
                spec,
                split,
                index,
                raw,
                image_key,
                image_path,
                choice[0],
                choice[1],
                prompt,
                "cxr",
                limit_per_domain,
                questions_per_image,
            )
            if _enough(counts, expected, limit_per_domain):
                break
    finally:
        archive.close()
    return rows, dict(counts)


def _validate_proxy_suite(rows: list[dict]) -> dict:
    image_domains: dict[str, str] = {}
    leaks: list[dict[str, str]] = []
    for row in rows:
        image_key = row["metadata"]["image_key"]
        previous = image_domains.setdefault(image_key, row["domain"])
        if previous != row["domain"]:
            leaks.append({"image_key": image_key, "left": previous, "right": row["domain"]})
    domains = sorted({row["domain"] for row in rows})
    source = sorted(domain for domain in domains if domain.endswith(("-source_a", "-source_b")))
    target = sorted(domain for domain in domains if domain.endswith("-target"))
    if leaks:
        raise RuntimeError("image leakage detected across generated proxy domains")
    if len(source) < 2 or not target:
        raise RuntimeError("prepared suite requires at least two source domains and one target domain")
    return {"source_domains": source, "target_domains": target, "image_leaks": leaks}


def prepare_public_suite(
    config_path: str | Path,
    artifact_root: str | Path,
    output_dir: str | Path,
    limit_per_domain: int = 8,
    questions_per_image: int = 1,
) -> dict:
    """Convert downloaded public snapshots into a runnable, leakage-audited proxy suite.

    These deterministic image-level partitions are for mechanism testing. They are
    deliberately marked as non-clinical domains and cannot support a hospital-DG claim.
    """

    config = load_yaml(config_path)
    artifact_root = Path(artifact_root).resolve()
    output_dir = Path(output_dir).resolve()
    image_root = output_dir / "images"
    all_rows: list[dict] = []
    dataset_reports = []
    for spec in config["datasets"]:
        snapshot = artifact_root / "datasets" / spec["id"].replace("/", "--")
        if not snapshot.is_dir():
            raise FileNotFoundError(f"downloaded dataset snapshot not found: {snapshot}")
        dataset_image_root = image_root / _slug(spec["name"])
        if spec["adapter"] == "slake":
            rows, counts = _prepare_slake(
                spec, snapshot, dataset_image_root, limit_per_domain, questions_per_image
            )
        else:
            rows, counts = _prepare_parquet_dataset(
                spec,
                snapshot,
                dataset_image_root,
                limit_per_domain,
                questions_per_image,
                config["oct_candidates"],
            )
        if not rows:
            raise RuntimeError(f"dataset {spec['id']} produced no compatible benchmark rows")
        all_rows.extend(rows)
        dataset_reports.append({"name": spec["name"], "id": spec["id"], "rows": len(rows), "domains": counts})

    audit = _validate_proxy_suite(all_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = output_dir / "manifest.jsonl"
    manifest_content = "".join(
        json.dumps(row, ensure_ascii=False) + "\n"
        for row in sorted(all_rows, key=lambda item: item["id"])
    )
    manifest_updated = _write_text_if_changed(manifest, manifest_content)

    comparison = {
        "seed": int(config.get("seed", 42)),
        "evaluation_protocol": "public image-level proxy partitions; not hospital FedDG",
        "strict_hospital_dg_claim_allowed": False,
        "source_domains": audit["source_domains"],
        "target_domains": audit["target_domains"],
        "method": config["method"],
        "evaluation": config["evaluation"],
    }
    comparison_path = output_dir / "compare.yaml"
    save_yaml(comparison_path, comparison)
    report = {
        "manifest": str(manifest),
        "comparison_config": str(comparison_path),
        "rows": len(all_rows),
        "limit_per_domain": limit_per_domain,
        "questions_per_image": questions_per_image,
        "manifest_updated": manifest_updated,
        "protocol": comparison["evaluation_protocol"],
        "strict_hospital_dg_claim_allowed": False,
        "datasets": dataset_reports,
        **audit,
    }
    save_json(output_dir / "prepare-report.json", report)
    return report


def image_bytes_are_valid(payload: bytes) -> bool:
    """Small validation helper used by tests and preparation diagnostics."""

    try:
        with Image.open(io.BytesIO(payload)) as image:
            image.verify()
        return True
    except Exception:  # noqa: BLE001 - validation helper intentionally returns a boolean
        return False
