from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np

from .experts import BlipConceptExpert, CheXagentConceptExpert, ConchConceptExpert
from .generalist import QwenLayerProbe
from .io import save_json, save_records
from .routing import MetadataRouter, route_with_medical_vlm
from .types import EvidenceRecord


def _read_manifest(path: str | Path, limit: int = 0) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
    for row in rows:
        missing = {"id", "image", "domain", "modality", "prompt", "candidates", "label"} - row.keys()
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


def _release(model: object) -> None:
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
    adapter = spec["adapter"]
    if adapter == "generative_chexagent":
        return CheXagentConceptExpert(model_id)
    if adapter == "generative_blip":
        return BlipConceptExpert(model_id)
    if adapter == "contrastive_conch":
        if not Path(model_id).exists() and not model_id.startswith("hf_hub:"):
            model_id = "hf_hub:" + model_id
        return ConchConceptExpert(model_id)
    raise ValueError(f"unsupported expert adapter: {adapter}")


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
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    state: dict[str, dict] = {str(row["id"]): {"row": row, "expert_scores": {}} for row in rows}

    general_spec = config["generalist"]
    generalist = QwenLayerProbe(
        _local_or_remote(general_spec["id"], artifact_root),
        layers=general_spec["layers"],
        dtype=general_spec.get("dtype", "bfloat16"),
        device_map=general_spec.get("device_map", "auto"),
    )
    for row in rows:
        null_logits, visual_layers = generalist.probe(
            row["image"], row["prompt"], list(row["candidates"])
        )
        state[str(row["id"])]["general_null_logits"] = null_logits
        state[str(row["id"])]["general_visual_layers"] = visual_layers
    _release(generalist)

    broad_spec = config["broad_specialist"]
    broad = QwenLayerProbe(
        _local_or_remote(broad_spec["id"], artifact_root),
        layers=broad_spec.get("layers", [-1]),
        dtype=broad_spec.get("dtype", "bfloat16"),
        device_map=broad_spec.get("device_map", "auto"),
    )
    available = sorted(config["experts"])
    metadata_router = MetadataRouter(available)
    for row in rows:
        _, broad_visual = broad.probe(row["image"], row["prompt"], list(row["candidates"]))
        state[str(row["id"])]["broad_specialist_scores"] = broad_visual[-1]
        if oracle_router:
            peak = 0.98
            tail = (1.0 - peak) / (len(available) - 1)
            route = {name: peak if name == row["modality"] else tail for name in available}
        elif config["router"].get("adapter") == "qwen_medical_router":
            route = route_with_medical_vlm(broad, row["image"], available)
        else:
            route = metadata_router.route(row["image"], row.get("metadata"))
        state[str(row["id"])]["router_probs"] = route
    _release(broad)

    for modality, spec in sorted(config["experts"].items()):
        expert = _expert_from_spec(spec, artifact_root)
        for row in rows:
            scores = expert.image_null_scores(
                row["image"], row["prompt"], list(row["candidates"])
            )
            state[str(row["id"])]["expert_scores"][modality] = np.asarray(scores, dtype=float)
        _release(expert)

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
                metadata=row.get("metadata"),
            )
        )
    save_records(output, records)
    save_json(
        output.with_suffix(".report.json"),
        {
            "records": len(records),
            "oracle_router": oracle_router,
            "manifest": str(Path(manifest).resolve()),
            "contains_images": False,
            "contains_model_weights": False,
        },
    )
    return records
