from __future__ import annotations

import gc
import hashlib
import importlib
import json
import time
from importlib import metadata as importlib_metadata
from pathlib import Path

import numpy as np

from .experts import (
    BiomedClipAdapter,
    BlipConceptExpert,
    CheXagentConceptExpert,
    ConchConceptExpert,
)
from .generalist import QwenLayerProbe
from .io import save_json, save_records
from .routing import MetadataRouter, normalized_entropy, route_with_medical_vlm
from .types import EvidenceRecord

EXTRACTION_CACHE_SCHEMA = 2
EXTRACTION_CONTRACT = "generalist-layer-evidence+semantic-experts+two-stage-ood-v1"


def _read_manifest(path: str | Path, limit: int = 0) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
    for row in rows:
        missing = {
            "id",
            "image",
            "domain",
            "modality",
            "prompt",
            "candidates",
            "label",
        } - row.keys()
        if missing:
            raise ValueError(f"manifest row is missing: {sorted(missing)}")
        if not row["candidates"] or row["label"] is None:
            raise ValueError("extraction requires non-empty candidates and an integer label")
    return rows


def _local_or_remote(model_id: str, artifact_root: str | Path | None) -> str:
    if artifact_root is None:
        return model_id
    local = Path(artifact_root) / "models" / model_id.replace("/", "--")
    return str(local) if local.exists() else model_id


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot_provenance(spec: dict, artifact_root: str | Path | None) -> dict:
    payload = {
        "id": str(spec["id"]),
        "revision": spec.get("revision"),
        "adapter": spec.get("adapter"),
        "factory": spec.get("factory"),
    }
    if artifact_root is None:
        payload["snapshot"] = "remote-or-untracked"
        return payload
    marker = (
        Path(artifact_root)
        / "models"
        / str(spec["id"]).replace("/", "--")
        / ".merit-download-complete.json"
    )
    try:
        recorded = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        payload["snapshot"] = "missing-completion-marker"
    else:
        payload["snapshot"] = {
            key: recorded.get(key)
            for key in (
                "schema",
                "id",
                "kind",
                "revision",
                "files",
                "payload_files",
                "bytes",
                "fingerprint",
            )
        }
    return payload


def _extraction_runtime_provenance() -> dict:
    package_root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for source in sorted(package_root.rglob("*.py")):
        relative = source.relative_to(package_root).as_posix()
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(bytes.fromhex(_file_sha256(source)))
    versions = {}
    for distribution in (
        "torch",
        "transformers",
        "open-clip-torch",
        "qwen-vl-utils",
        "conch",
    ):
        try:
            versions[distribution] = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError:
            versions[distribution] = "not-installed-as-distribution"
    return {
        "package_source_sha256": digest.hexdigest(),
        "dependency_versions": versions,
    }


def _manifest_image_provenance(manifest: str | Path) -> dict:
    manifest_path = Path(manifest).resolve()
    images: set[Path] = set()
    with manifest_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if "image" not in row:
                continue
            image = Path(str(row["image"]))
            images.add((image if image.is_absolute() else manifest_path.parent / image).resolve())
    digest = hashlib.sha256()
    total_bytes = 0
    missing: list[str] = []
    for image in sorted(images, key=lambda value: str(value)):
        digest.update(str(image).encode("utf-8") + b"\0")
        try:
            size = image.stat().st_size
            content_hash = _file_sha256(image)
        except OSError:
            missing.append(str(image))
            digest.update(b"missing\n")
            continue
        total_bytes += size
        digest.update(f"{size}\0{content_hash}\n".encode())
    return {
        "files": len(images),
        "bytes": total_bytes,
        "missing": missing,
        "fingerprint": f"sha256:{digest.hexdigest()}",
    }


def extraction_provenance(
    manifest: str | Path,
    config: dict,
    artifact_root: str | Path | None,
    *,
    limit: int = 0,
    oracle_router: bool = False,
) -> dict:
    """Fingerprint every input that can make a frozen evidence cache stale."""

    models = {
        "generalist": _snapshot_provenance(config["generalist"], artifact_root),
        "broad_specialist": _snapshot_provenance(config["broad_specialist"], artifact_root),
        "experts": {
            expert_id: _snapshot_provenance(spec, artifact_root)
            for expert_id, spec in sorted(config["experts"].items())
        },
    }
    ingredients = {
        "schema_version": EXTRACTION_CACHE_SCHEMA,
        "contract": EXTRACTION_CONTRACT,
        "manifest_sha256": _file_sha256(manifest),
        "image_payload": _manifest_image_provenance(manifest),
        "config": config,
        "models": models,
        "runtime": _extraction_runtime_provenance(),
        "limit": int(limit),
        "oracle_router": bool(oracle_router),
    }
    encoded = json.dumps(
        ingredients, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return {
        **ingredients,
        "fingerprint": f"sha256:{hashlib.sha256(encoded).hexdigest()}",
    }


def verify_extraction_cache(
    cache: str | Path,
    manifest: str | Path,
    config: dict,
    artifact_root: str | Path | None,
    *,
    limit: int = 0,
    oracle_router: bool = False,
) -> dict:
    """Reject a cache unless provenance and record count match exactly."""

    cache_path = Path(cache)
    report_path = cache_path.with_suffix(".report.json")
    reasons: list[str] = []
    try:
        observed_report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        observed_report = {}
        reasons.append("missing-or-invalid-extraction-report")
    try:
        expected = extraction_provenance(
            manifest,
            config,
            artifact_root,
            limit=limit,
            oracle_router=oracle_router,
        )
    except (OSError, TypeError, ValueError, KeyError) as error:
        return {"ready": False, "reasons": [f"cannot-fingerprint-inputs:{error}"]}
    observed = observed_report.get("extraction_provenance", {})
    if observed.get("fingerprint") != expected["fingerprint"]:
        reasons.append("extraction-provenance-mismatch")
    if expected["image_payload"]["missing"]:
        reasons.append("missing-manifest-image-payload")
    model_entries = [expected["models"]["generalist"], expected["models"]["broad_specialist"]]
    model_entries.extend(expected["models"]["experts"].values())
    if any(not isinstance(item.get("snapshot"), dict) for item in model_entries):
        reasons.append("unverified-model-snapshot")
    parsed_records: list[EvidenceRecord] = []
    try:
        with cache_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    parsed_records.append(EvidenceRecord.from_json(json.loads(line)))
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
        parsed_records = []
        reasons.append("missing-or-invalid-cache-record")
    record_count = len(parsed_records)
    if record_count != int(observed_report.get("records", -1)):
        reasons.append("cache-record-count-mismatch")
    try:
        expected_ids = [str(row["id"]) for row in _read_manifest(manifest, limit=limit)]
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
        expected_ids = []
        reasons.append("missing-or-invalid-manifest")
    observed_ids = [record.sample_id for record in parsed_records]
    if observed_ids != expected_ids or observed_ids != observed_report.get("sample_ids"):
        reasons.append("cache-sample-id-or-order-mismatch")
    try:
        cache_sha256 = _file_sha256(cache_path)
    except OSError:
        cache_sha256 = None
    if cache_sha256 != observed_report.get("cache_sha256"):
        reasons.append("cache-content-hash-mismatch")
    return {
        "ready": not reasons,
        "cache": str(cache_path.resolve()),
        "records": record_count,
        "cache_sha256": cache_sha256,
        "expected_fingerprint": expected["fingerprint"],
        "observed_fingerprint": observed.get("fingerprint"),
        "reasons": reasons,
    }


def _release(model: object | None = None) -> None:
    """Collect host and accelerator memory after the owning scope has returned.

    Passing a model is retained for callers outside this module.  Sequential
    extraction deliberately calls this with no model: its stage helper has already
    returned, so model, bound-method, and last-output references are truly gone
    before collection and CUDA cache release begin.
    """

    del model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _expert_from_spec(spec: dict, artifact_root: str | Path | None):
    model_id = _local_or_remote(spec["id"], artifact_root)
    if spec.get("factory"):
        module_name, separator, attribute = str(spec["factory"]).partition(":")
        if not separator or not module_name or not attribute:
            raise ValueError("expert factory must use the form package.module:callable")
        factory = getattr(importlib.import_module(module_name), attribute)
        return factory(model_id=model_id, **dict(spec.get("factory_kwargs", {})))
    adapter = spec["adapter"]
    if adapter == "generative_chexagent":
        return CheXagentConceptExpert(model_id)
    if adapter == "generative_blip":
        return BlipConceptExpert(model_id)
    if adapter == "contrastive_conch":
        if not Path(model_id).exists() and not model_id.startswith("hf_hub:"):
            model_id = "hf_hub:" + model_id
        return ConchConceptExpert(model_id)
    if adapter == "contrastive_biomedclip":
        return BiomedClipAdapter(model_id)
    raise ValueError(f"unsupported expert adapter: {adapter}")


def _extract_generalist_stage(
    rows: list[dict],
    state: dict[str, dict],
    general_spec: dict,
    artifact_root: str | Path | None,
) -> None:
    """Own all generalist references inside a scope that ends before cache release."""

    generalist = QwenLayerProbe(
        _local_or_remote(general_spec["id"], artifact_root),
        layers=general_spec["layers"],
        dtype=general_spec.get("dtype", "bfloat16"),
        device_map=general_spec.get("device_map", "auto"),
    )
    for row in rows:
        if general_spec.get("candidate_scores_only", False):
            candidate_scores = generalist.candidate_log_likelihoods(
                row["image"], row["prompt"], list(row["candidates"])
            )
            null_logits = np.zeros_like(candidate_scores)
            visual_layers = np.asarray(candidate_scores, dtype=float)[None, :]
        else:
            null_logits, visual_layers = generalist.probe(
                row["image"], row["prompt"], list(row["candidates"])
            )
        state[str(row["id"])]["general_null_logits"] = null_logits
        state[str(row["id"])]["general_visual_layers"] = visual_layers


def _route_from_broad(
    broad: object,
    row: dict,
    available: list[str],
    router_config: dict,
    metadata_router: MetadataRouter | None,
    oracle_router: bool,
    qwen_router: bool,
    biomedclip_router: bool,
) -> dict[str, float]:
    if len(available) == 1:
        route = {available[0]: 1.0}
    elif oracle_router:
        peak = 0.98
        tail = (1.0 - peak) / (len(available) - 1)
        route = {name: peak if name == row["modality"] else tail for name in available}
    elif qwen_router:
        route = route_with_medical_vlm(broad, row["image"], available)
    elif biomedclip_router:
        route = broad.route(row["image"], available)
    else:
        if metadata_router is None:
            raise RuntimeError("metadata router is unavailable for multiple modalities")
        route = metadata_router.route(row["image"], row.get("metadata"))
    abstain_entropy = float(router_config.get("abstain_entropy", 1.0))
    if len(available) > 1 and normalized_entropy(route) >= abstain_entropy:
        return {name: 1.0 / len(available) for name in available}
    return route


def _extract_broad_stage(
    rows: list[dict],
    state: dict[str, dict],
    config: dict,
    artifact_root: str | Path | None,
    oracle_router: bool,
) -> None:
    """Own broad-specialist/router references until this stage fully finishes."""

    broad_spec = config["broad_specialist"]
    expert_modalities = {
        expert_id: tuple(
            str(value) for value in spec.get("modalities", [spec.get("modality", expert_id)])
        )
        for expert_id, spec in config["experts"].items()
    }
    available = sorted(
        {modality for modalities in expert_modalities.values() for modality in modalities}
    )
    metadata_router = MetadataRouter(available) if len(available) > 1 else None
    contrastive = broad_spec.get("adapter") == "contrastive_biomedclip"
    if contrastive:
        broad = _expert_from_spec(broad_spec, artifact_root)
    else:
        broad = QwenLayerProbe(
            _local_or_remote(broad_spec["id"], artifact_root),
            layers=broad_spec.get("layers", [-1]),
            dtype=broad_spec.get("dtype", "bfloat16"),
            device_map=broad_spec.get("device_map", "auto"),
        )

    for row in rows:
        if contrastive:
            if hasattr(broad, "score_and_domain_embedding"):
                broad_scores, cheap_feature = broad.score_and_domain_embedding(
                    row["image"], row["prompt"], list(row["candidates"])
                )
                state[str(row["id"])]["broad_specialist_scores"] = broad_scores
                state[str(row["id"])]["cheap_domain_feature"] = np.asarray(
                    cheap_feature, dtype=float
                )
            else:
                state[str(row["id"])]["broad_specialist_scores"] = broad.image_null_scores(
                    row["image"], row["prompt"], list(row["candidates"])
                )
                if hasattr(broad, "domain_embedding"):
                    state[str(row["id"])]["cheap_domain_feature"] = np.asarray(
                        broad.domain_embedding(row["image"]), dtype=float
                    )
        else:
            _, broad_visual = broad.probe(row["image"], row["prompt"], list(row["candidates"]))
            state[str(row["id"])]["broad_specialist_scores"] = broad_visual[-1]

        modality_route = _route_from_broad(
            broad,
            row,
            available,
            config["router"],
            metadata_router,
            oracle_router,
            qwen_router=(
                not contrastive and config["router"].get("adapter") == "qwen_medical_router"
            ),
            biomedclip_router=(
                contrastive and config["router"].get("adapter") == "biomedclip_router"
            ),
        )
        # The router reasons about actual modalities, never arbitrary expert
        # IDs. Preserve that output for capability-aware controllers, while a
        # normalized expert-level view keeps the legacy cached baselines
        # backwards compatible.
        experts_per_modality = {
            modality: sum(modality in modalities for modalities in expert_modalities.values())
            for modality in available
        }
        expert_route = {
            expert_id: sum(
                float(modality_route.get(name, 0.0)) / max(experts_per_modality.get(name, 0), 1)
                for name in modalities
            )
            for expert_id, modalities in expert_modalities.items()
        }
        total = sum(expert_route.values())
        if total <= 0.0:
            expert_route = {name: 1.0 / len(expert_route) for name in expert_route}
        else:
            expert_route = {name: value / total for name, value in expert_route.items()}
        state[str(row["id"])]["modality_router_probs"] = modality_route
        state[str(row["id"])]["router_probs"] = expert_route


def _extract_one_expert_stage(
    modality: str,
    spec: dict,
    rows: list[dict],
    state: dict[str, dict],
    artifact_root: str | Path | None,
) -> None:
    """Own one specialist and its final native outputs in an isolated scope."""

    expert = _expert_from_spec(spec, artifact_root)
    for row in rows:
        started = time.perf_counter()
        if hasattr(expert, "score_and_domain_embedding"):
            scores, native_feature = expert.score_and_domain_embedding(
                row["image"], row["prompt"], list(row["candidates"])
            )
            state[str(row["id"])].setdefault("expert_native_features", {})[modality] = np.asarray(
                native_feature, dtype=float
            )
        else:
            scores = expert.image_null_scores(row["image"], row["prompt"], list(row["candidates"]))
            if hasattr(expert, "domain_embedding"):
                state[str(row["id"])].setdefault("expert_native_features", {})[modality] = (
                    np.asarray(expert.domain_embedding(row["image"]), dtype=float)
                )
        state[str(row["id"])]["expert_scores"][modality] = np.asarray(scores, dtype=float)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        state[str(row["id"])].setdefault("expert_latency_ms", {})[modality] = elapsed_ms


def extract_manifest(
    manifest: str | Path,
    config: dict,
    output: str | Path,
    artifact_root: str | Path | None = None,
    limit: int = 0,
    oracle_router: bool = False,
) -> list[EvidenceRecord]:
    """Sequential extraction keeps peak VRAM to one generalist/specialist at a time."""

    rows = _read_manifest(manifest, limit)
    provenance = extraction_provenance(
        manifest,
        config,
        artifact_root,
        limit=limit,
        oracle_router=oracle_router,
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    state: dict[str, dict] = {str(row["id"]): {"row": row, "expert_scores": {}} for row in rows}

    try:
        _extract_generalist_stage(rows, state, config["generalist"], artifact_root)
    finally:
        _release()

    try:
        _extract_broad_stage(rows, state, config, artifact_root, oracle_router)
    finally:
        _release()

    for modality, spec in sorted(config["experts"].items()):
        try:
            _extract_one_expert_stage(modality, spec, rows, state, artifact_root)
        finally:
            _release()

    records = []
    for row in rows:
        item = state[str(row["id"])]
        records.append(
            EvidenceRecord(
                sample_id=str(row["id"]),
                domain=str(row["domain"]),
                modality=str(row["modality"]),
                candidates=list(row["candidates"]),
                label=int(row["label"]),
                general_null_logits=np.asarray(item["general_null_logits"], dtype=float),
                general_visual_layers=np.asarray(item["general_visual_layers"], dtype=float),
                expert_scores=item["expert_scores"],
                broad_specialist_scores=np.asarray(item["broad_specialist_scores"], dtype=float),
                router_probs=item["router_probs"],
                metadata={
                    **dict(row.get("metadata") or {}),
                    **(
                        {"cheap_domain_feature": item["cheap_domain_feature"].tolist()}
                        if "cheap_domain_feature" in item
                        else {}
                    ),
                    "expert_native_features": {
                        name: value.tolist()
                        for name, value in item.get("expert_native_features", {}).items()
                    },
                    "expert_latency_ms": dict(item.get("expert_latency_ms", {})),
                    "modality_router_probs": dict(item.get("modality_router_probs", {})),
                    "expert_modalities": {
                        expert_id: list(spec.get("modalities", [spec.get("modality", expert_id)]))
                        for expert_id, spec in config["experts"].items()
                    },
                    "expert_capabilities": {
                        expert_id: list(spec.get("capabilities", ["classification"]))
                        for expert_id, spec in config["experts"].items()
                    },
                    "task_type": row.get("task_type", "closed_set"),
                },
            )
        )
    save_records(output, records)
    cache_sha256 = _file_sha256(output)
    save_json(
        output.with_suffix(".report.json"),
        {
            "records": len(records),
            "sample_ids": [record.sample_id for record in records],
            "cache_sha256": cache_sha256,
            "oracle_router": oracle_router,
            "manifest": str(Path(manifest).resolve()),
            "contains_images": False,
            "contains_model_weights": False,
            "extraction_provenance": provenance,
        },
    )
    return records
