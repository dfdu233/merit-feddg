"""Lazy heterogeneous tools: fixed concepts, source cases, and native spatial evidence.

Nothing here sees target answers or scores the generalist's continuation candidates.
Contrastive similarities are not diagnostic probabilities. Retrieved answers remain
attached to their source questions. Custom adapters preserve their native payload.
"""

from __future__ import annotations

import gc
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from .capabilities import CapabilityRequest, CapabilityResult, EvidenceItem
from .experts.base import load_rgb
from .extract import _expert_from_spec, _local_or_remote


def _array(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=float)


def _unit_vector(value) -> np.ndarray:
    vector = _array(value).reshape(-1)
    if not vector.size or not np.isfinite(vector).all():
        raise ValueError("native image embedding must be non-empty and finite")
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("native image embedding has zero norm")
    return vector / norm


def _pixel_digest(image) -> str:
    rgb = load_rgb(image)
    return hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()


def _stable_hash(payload) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def _catalog(spec: dict) -> tuple[tuple[str, str], ...]:
    raw = spec.get("catalog", spec.get("concepts", ()))
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError("classification requires a fixed config catalog, not answer candidates")
    entries = []
    for value in raw:
        if isinstance(value, str):
            name = value.strip()
            template = spec.get("prompt_template", "A medical image showing {concept}.")
            prompt = str(template).format(concept=name)
        elif isinstance(value, Mapping):
            name = str(value.get("name", "")).strip()
            prompt = str(value.get("prompt", name)).strip()
        else:
            raise TypeError("catalog entries must be text or name/prompt mappings")
        if not name or not prompt:
            raise ValueError("catalog names and prompts must be non-empty")
        entries.append((name, prompt))
    if len({name for name, _ in entries}) != len(entries):
        raise ValueError("catalog concept names must be unique")
    return tuple(entries)


def _roi(value) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    try:
        values = tuple(float(x) for x in value)
    except (ValueError, TypeError) as exc:
        raise ValueError("segmentation region must be normalized xyxy") from exc
    if (
        len(values) != 4
        or not all(np.isfinite(x) and 0 <= x <= 1 for x in values)
        or values[0] >= values[2]
        or values[1] >= values[3]
    ):
        raise ValueError("segmentation region must be normalized xyxy with positive area")
    return values


def encode_binary_mask(mask) -> dict:
    """Lossless uncompressed, row-major RLE; explicitly not COCO column-major RLE."""
    values = np.asarray(mask)
    if values.ndim != 2 or not values.size:
        raise ValueError("mask must be a non-empty two-dimensional array")
    if not np.isfinite(values).all() or not np.isin(values, (0, 1)).all():
        raise ValueError("mask must contain only finite binary values")
    flat = values.astype(np.uint8).ravel(order="C")
    transitions = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    lengths = np.diff(np.r_[0, transitions, flat.size]).tolist()
    if int(flat[0]) == 1:
        lengths.insert(0, 0)
    return {"encoding": "rle-row-major-zero-first", "size": list(values.shape), "counts": lengths}


class MedSamCapabilityAdapter:
    """Optional box-prompted MedSAM using the official Transformers SAM interface.

    The region is supplied explicitly by the caller, never synthesized from a label.
    The output is prompt-conditioned foreground, not a disease/lesion diagnosis.
    """

    def __init__(self, model_id: str, device: str = "auto", revision: str | None = None):
        try:
            import torch
            from transformers import SamModel, SamProcessor
        except ImportError as exc:
            raise RuntimeError("MedSAM requires merit-feddg[research]") from exc
        self.torch = torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        kwargs = {"revision": revision} if revision and not Path(model_id).is_dir() else {}
        self.processor = SamProcessor.from_pretrained(model_id, **kwargs)
        self.model = SamModel.from_pretrained(model_id, **kwargs).to(self.device).eval()

    def segment(self, image, region) -> tuple[np.ndarray, float]:
        region = _roi(region)
        if region is None:
            raise ValueError("MedSAM requires an explicit normalized region")
        rgb = load_rgb(image)
        width, height = rgb.size
        box = [region[0] * width, region[1] * height, region[2] * width, region[3] * height]
        inputs = self.processor(images=rgb, input_boxes=[[box]], return_tensors="pt")
        original_sizes = inputs["original_sizes"].clone()
        reshaped_sizes = inputs["reshaped_input_sizes"].clone()
        inputs = inputs.to(self.device)
        with self.torch.inference_mode():
            outputs = self.model(**inputs, multimask_output=False)
        masks = self.processor.image_processor.post_process_masks(
            outputs.pred_masks.detach().cpu(), original_sizes, reshaped_sizes
        )
        mask = masks[0][0][0].numpy().astype(np.uint8)
        quality = float(outputs.iou_scores.detach().float().cpu().reshape(-1)[0])
        if mask.shape != (height, width) or not np.isfinite(quality):
            raise ValueError("MedSAM returned invalid original-resolution output")
        return mask, quality


class CapabilityPool:
    """A model may back multiple tools; source retrieval never indexes target records.

    ``factory: package.module:callable`` loads a native adapter via the existing
    loader. The adapter implements ``infer(request) -> CapabilityResult``. It may
    expose detection, segmentation, retrieval or generation without changing this
    registry. Its CapabilityResult and EvidenceItem identities must match the request.
    """

    def __init__(self, specs, artifacts, source_records=(), source_references=None):
        self.specs = {str(name): dict(spec) for name, spec in dict(specs).items()}
        self.artifacts = artifacts
        self.source_records = tuple(dict(row) for row in source_records)
        self.source_references = dict(source_references or {})
        ids = [str(row.get("id", "")) for row in self.source_records]
        if any(not sample_id for sample_id in ids) or len(set(ids)) != len(ids):
            raise ValueError("retrieval sources must have unique non-empty IDs")
        for row in self.source_records:
            if row.get("role", "source") != "source":
                raise ValueError("retrieval index cannot contain target records")
            if any(not str(row.get(key, "")).strip() for key in ("image", "domain", "group_id")):
                raise ValueError("retrieval sources need image, domain and group_id")
        if set(self.source_references) - set(ids):
            raise ValueError("retrieval references may contain source IDs only")
        self.models: dict[str, object] = {}
        self.features: dict[tuple[str, str], np.ndarray] = {}
        self.source_features: dict[tuple[str, str], np.ndarray] = {}
        self.text_features: dict[tuple[str, tuple], np.ndarray] = {}
        self._digest_cache: dict[tuple, str] = {}

    @staticmethod
    def _adapter(spec):
        aliases = {"conch": "contrastive_conch", "biomedclip": "contrastive_biomedclip"}
        adapter = str(spec.get("adapter", "native_factory"))
        return aliases.get(adapter, adapter)

    def _model_spec(self, spec):
        payload = dict(spec)
        payload["adapter"] = self._adapter(spec)
        if payload["adapter"] == "source_retrieval":
            payload["adapter"] = str(spec.get("encoder_adapter", "contrastive_biomedclip"))
        if spec.get("checkpoint_path"):
            payload["id"] = str(Path(spec["checkpoint_path"]).resolve())
        return payload

    def _model_key(self, spec):
        payload = self._model_spec(spec)
        return _stable_hash(
            {
                key: payload.get(key)
                for key in ("id", "revision", "adapter", "factory", "factory_kwargs", "device")
            }
        )

    def _model(self, spec):
        key = self._model_key(spec)
        if key not in self.models:
            payload = self._model_spec(spec)
            artifacts = None if spec.get("checkpoint_path") else self.artifacts
            if payload["adapter"] == "medsam" and not payload.get("factory"):
                self.models[key] = MedSamCapabilityAdapter(
                    _local_or_remote(payload["id"], artifacts),
                    device=str(payload.get("device", "auto")),
                    revision=payload.get("revision"),
                )
            else:
                self.models[key] = _expert_from_spec(payload, artifacts)
        return self.models[key]

    def _digest(self, image):
        if isinstance(image, (str, Path)):
            path = Path(image).resolve()
            info = path.stat()
            key = (str(path), info.st_size, info.st_mtime_ns)
            if key not in self._digest_cache:
                self._digest_cache[key] = _pixel_digest(path)
            return self._digest_cache[key]
        return _pixel_digest(image)

    def _image_feature(self, spec, image, *, source=False):
        key = (self._model_key(spec), self._digest(image))
        cache = self.source_features if source else self.features
        if key in self.features:
            return self.features[key]
        if key in self.source_features:
            return self.source_features[key]
        model = self._model(spec)
        if not hasattr(model, "domain_embedding"):
            raise ValueError("native contrastive/retrieval adapter needs domain_embedding(image)")
        cache[key] = _unit_vector(model.domain_embedding(image))
        return cache[key]

    def _text_vectors(self, spec, catalog):
        key = (self._model_key(spec), catalog)
        if key not in self.text_features:
            model = self._model(spec)
            prompts = [prompt for _, prompt in catalog]
            if hasattr(model, "_text_embeddings"):
                values = model._text_embeddings(prompts)
            else:
                tokens = model.tokenize(model.tokenizer, prompts).to(model.device)
                with model.torch.inference_mode():
                    values = model.model.encode_text(tokens, normalize=True)
            values = _array(values)
            if values.ndim != 2 or values.shape[0] != len(catalog):
                raise ValueError("catalog text embedding shape mismatch")
            self.text_features[key] = np.stack([_unit_vector(row) for row in values])
        return self.text_features[key]

    def source_index_identity(self):
        rows = []
        for row in sorted(self.source_records, key=lambda value: str(value["id"])):
            rows.append(
                {
                    "id": row["id"],
                    "domain": row["domain"],
                    "group_id": row["group_id"],
                    "image_sha256": self._digest(row["image"]),
                    "question": row.get("question", ""),
                    "modality": row.get("modality", ""),
                    "reference": self.source_references.get(row["id"]),
                }
            )
        return {"schema": 1, "source_count": len(rows), "fingerprint": _stable_hash(rows)}

    def reset_case(self):
        """Discard query features while keeping source index and shared model weights."""
        self.features.clear()

    def clear(self):
        self.models.clear()
        self.features.clear()
        self.source_features.clear()
        self.text_features.clear()
        self._digest_cache.clear()
        gc.collect()

    def infer(self, expert_id: str, request: CapabilityRequest) -> CapabilityResult:
        spec = self.specs[expert_id]
        adapter = self._adapter(spec)
        defaults = {
            "source_retrieval": ("retrieval",),
            "medsam": ("segmentation",),
            "contrastive_conch": ("classification",),
            "contrastive_biomedclip": ("classification",),
        }
        allowed = spec.get("capabilities", defaults.get(adapter, ()))
        if request.capability not in allowed:
            return CapabilityResult(expert_id, request.capability, (), reason="wrong_capability")
        if spec.get("modalities") and request.modality not in spec["modalities"]:
            return CapabilityResult(expert_id, request.capability, (), reason="wrong_modality")
        if spec.get("tasks") and request.task not in spec["tasks"]:
            return CapabilityResult(expert_id, request.capability, (), reason="wrong_task")
        scope = str(spec.get("scope", request.capability))
        if request.scope and request.scope != scope:
            return CapabilityResult(expert_id, request.capability, (), reason="wrong_scope")
        if request.region is not None and adapter in {
            "source_retrieval",
            "contrastive_conch",
            "contrastive_biomedclip",
        }:
            return CapabilityResult(expert_id, request.capability, (), reason="unsupported_region")
        if adapter == "source_retrieval":
            return self._retrieve(expert_id, spec, request)
        if adapter in {"contrastive_conch", "contrastive_biomedclip"}:
            return self._classify(expert_id, spec, request)
        if adapter == "medsam":
            return self._segment(expert_id, spec, request)
        model = self._model(spec)
        if not hasattr(model, "infer"):
            raise ValueError("native capability factories must implement infer(request)")
        result = model.infer(request)
        if not isinstance(result, CapabilityResult):
            raise TypeError("native adapter must return CapabilityResult, not candidate scores")
        if result.expert_id != expert_id or result.capability != request.capability:
            raise ValueError("native result identity does not match the selected capability")
        for item in result.items:
            if item.expert_id != expert_id or item.capability != request.capability:
                raise ValueError("native evidence identity does not match the selected capability")
            if item.scope != scope:
                raise ValueError("native evidence scope differs from the configured tool scope")
        return result

    def _classify(self, expert_id, spec, request):
        catalog = _catalog(spec)
        image = self._image_feature(spec, request.image)
        text = self._text_vectors(spec, catalog)
        if text.shape[1] != image.size:
            raise ValueError("native text/image embedding dimensions differ")
        scores = np.clip(text @ image, -1.0, 1.0)
        ordering = sorted(range(len(catalog)), key=lambda i: (-float(scores[i]), catalog[i][0]))
        entries = [
            {"concept": catalog[i][0], "prompt": catalog[i][1], "similarity": float(scores[i])}
            for i in ordering
        ]
        summary = "Relative visual matches within a fixed catalog (not diagnosis probabilities): "
        summary += "; ".join(f"{x['concept']} ({x['similarity']:.3f})" for x in entries[:3])
        item = EvidenceItem(
            evidence_id=f"{expert_id}:{request.sample_id}:catalog",
            expert_id=expert_id,
            capability="classification",
            scope=str(spec.get("scope", "classification")),
            payload={
                "catalog": entries,
                "score_semantics": "relative_similarity",
                "catalog_exhaustive": False,
                "unlisted_concepts": "unknown",
            },
            summary=summary,
            confidence=None,
            provenance={
                "adapter": self._adapter(spec),
                "catalog_sha256": _stable_hash(catalog),
                "target_candidates_used": False,
                "image_minus_null": False,
                "query_used": False,
                "spatial_scope": "whole_image",
            },
        )
        return CapabilityResult(expert_id, request.capability, (item,))

    def _retrieve(self, expert_id, spec, request):
        if not request.domain or not request.group_id:
            raise ValueError("retrieval needs query domain/group metadata for exclusion")
        limit = int(spec.get("top_k", 3))
        if limit < 1:
            raise ValueError("retrieval top_k must be positive")
        query_digest = self._digest(request.image)
        candidates = []
        for row in self.source_records:
            if (
                str(row["domain"]) == request.domain
                or str(row["group_id"]) == request.group_id
                or str(row["id"]) == request.sample_id
                or self._digest(row["image"]) == query_digest
                or (row.get("modality") and row["modality"] != request.modality)
            ):
                continue
            candidates.append(row)
        if not candidates:
            return CapabilityResult(expert_id, request.capability, (), reason="no_eligible_sources")
        image = self._image_feature(spec, request.image)
        ranked = []
        for row in candidates:
            feature = self._image_feature(spec, row["image"], source=True)
            if feature.shape != image.shape:
                raise ValueError("retrieval index embedding dimension mismatch")
            ranked.append((float(np.clip(image @ feature, -1, 1)), str(row["id"]), row))
        ranked.sort(key=lambda value: (-value[0], value[1]))
        references = []
        for similarity, _, row in ranked[:limit]:
            references.append(
                {
                    "source_id": row["id"],
                    "source_domain": row["domain"],
                    "source_group_id": row["group_id"],
                    "source_image": str(row["image"]),
                    "source_question": str(row.get("question", "")),
                    "source_reference": self.source_references.get(row["id"]),
                    "similarity": similarity,
                }
            )
        summary = "Similar SOURCE cases; these records are not diagnoses of the query image. "
        summary += " ".join(
            f"[{row['source_id']}] Source question: {row['source_question']} "
            f"Source reference: {json.dumps(row['source_reference'], ensure_ascii=False)}."
            for row in references
        )
        item = EvidenceItem(
            evidence_id=f"{expert_id}:{request.sample_id}:retrieval",
            expert_id=expert_id,
            capability="retrieval",
            scope=str(spec.get("scope", "retrieval")),
            payload={
                "references": references,
                "score_semantics": "relative_similarity",
                "query_diagnosis": "not_inferred",
            },
            summary=summary,
            confidence=None,
            provenance={
                "adapter": "source_retrieval",
                "excluded_domain": request.domain,
                "excluded_group": request.group_id,
                "source_only": True,
                "eligible_source_count": len(candidates),
                "target_answers_used": False,
                "query_used": False,
                "search_mode": "whole_image_similarity",
            },
        )
        return CapabilityResult(expert_id, request.capability, (item,))

    def _segment(self, expert_id, spec, request):
        region = _roi(request.region)
        if region is None:
            return CapabilityResult(
                expert_id, request.capability, (), reason="explicit_roi_required"
            )
        mask, quality = self._model(spec).segment(request.image, region)
        encoded = encode_binary_mask(mask)
        fraction = float(np.asarray(mask, dtype=float).mean())
        ys, xs = np.nonzero(mask)
        height, width = np.asarray(mask).shape
        bbox = centroid = None
        if len(xs):
            bbox = [
                float(xs.min() / width),
                float(ys.min() / height),
                float((xs.max() + 1) / width),
                float((ys.max() + 1) / height),
            ]
            centroid = [float((xs.mean() + 0.5) / width), float((ys.mean() + 0.5) / height)]
        region_id = _stable_hash(region)[:12]
        item = EvidenceItem(
            evidence_id=f"{expert_id}:{request.sample_id}:mask:{region_id}",
            expert_id=expert_id,
            capability="segmentation",
            scope=str(spec.get("scope", "segmentation")),
            payload={
                "mask": encoded,
                "prompt_box_xyxy_normalized": list(region),
                "foreground_fraction_of_image": fraction,
                "foreground_bbox_xyxy_normalized": bbox,
                "foreground_centroid_xy_normalized": centroid,
                "semantic_class": "unknown",
                "predicted_iou": quality,
                "predicted_iou_is_calibrated": False,
            },
            summary=f"Box-prompted foreground covers {fraction:.1%} of the image. "
            "The mask does not identify a disease or establish a diagnosis.",
            confidence=None,
            provenance={
                "adapter": "medsam",
                "roi_source": "explicit_request",
                "mask_resolution": "original_image",
                "target_mask_used": False,
            },
        )
        return CapabilityResult(expert_id, request.capability, (item,))
