"""Explicit native-output contract for a trainable, non-text evidence channel.

Strings resolve *registered* concepts/scopes; they are never tokenized. Unknown
concepts are rejected, not assigned a random embedding. Source identity stays in
the audit, not in the learned representation, allowing same-contract replacement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image

from .native_evidence import _bbox, _label, _mapped_mask

SEMANTICS = {"none": 0, "relative_similarity": 1,
             "uncalibrated_independent_sigmoid": 2, "probability": 3}
KINDS = {"classification": 0, "segmentation": 1, "detection": 2}


@dataclass(frozen=True)
class TensorContract:
    concepts: tuple[str, ...]
    scopes: tuple[str, ...]
    bindings: tuple[dict, ...]
    grid_size: int = 24
    max_records: int = 64

    def __post_init__(self):
        for values in (self.concepts, self.scopes):
            if not values or len(set(values)) != len(values) or any(
                not isinstance(x, str) or not x for x in values
            ):
                raise ValueError("nonempty unique concept/scope vocabularies required")
        if (type(self.grid_size) is not int or not 1 <= self.grid_size <= 64
                or type(self.max_records) is not int or not 1 <= self.max_records <= 1024):
            raise ValueError("invalid tensor evidence budget")
        seen = set()
        for b in self.bindings:
            key = (b["expert"], b["scope"], b["label"])
            if key in seen or any(not isinstance(x, str) or not x for x in key):
                raise ValueError("duplicate/invalid native concept binding")
            seen.add(key)
            if b["concept"] not in self.concepts or b["scope_id"] not in self.scopes:
                raise ValueError("binding references an unregistered concept/scope")

    @classmethod
    def from_dict(cls, value):
        return cls(**{**value, "concepts": tuple(value["concepts"]),
                      "scopes": tuple(value["scopes"]), "bindings": tuple(value["bindings"])})


@dataclass
class TensorPacket:
    kind: np.ndarray
    concept: np.ndarray
    scope: np.ndarray
    semantics: np.ndarray
    numeric: np.ndarray
    regions: np.ndarray
    sources: tuple[tuple[str, str], ...]
    rejected: tuple[dict, ...]

    def __len__(self):
        return len(self.kind)


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def compile_tensor_evidence(items, image_size, contract):
    """Map known 2-D outputs to the deterministic square-padded VLM patch grid.

    No thresholding of disease scores, softmax across unrelated classes, inferred
    negatives, or fallback from an invalid mask to a prompt box. Generation and
    retrieval require a separate interface and are explicitly unsupported here.
    """
    width, height = image_size
    if any(type(v) is not int or v <= 0 for v in image_size):
        raise ValueError("positive original image dimensions required")
    side, grid = max(width, height), contract.grid_size
    left, top = (side - width) // 2, (side - height) // 2
    bindings = {(b["expert"], b["scope"], b["label"]): b for b in contract.bindings}
    records, rejected = [], []
    for item in items:
        payload, provenance = item.payload, item.provenance
        source = (item.expert_id, item.evidence_id)

        def reject(reason, source=source):
            rejected.append({"expert_id": source[0], "evidence_id": source[1], "reason": reason})

        if item.capability not in KINDS:
            reject("unsupported_output_type")
            continue
        if any(provenance.get(k) for k in (
            "target_mask_used", "target_masks_used", "target_annotations_used"
        )):
            reject("target_annotations_forbidden")
            continue
        if item.capability == "classification":
            entries = payload.get("catalog", payload.get("findings", []))
        else:
            entries = [payload] if "mask" in payload else []
            for key in ("structures", "detections", "boxes", "objects"):
                entries = [*entries, *payload.get(key, [])]
        if not entries:
            reject("no_observations")
        for entry in entries:
            if not isinstance(entry, dict):
                reject("invalid_observation")
                continue
            binding = bindings.get((item.expert_id, item.scope, _label(entry)))
            if binding is None:
                reject("unregistered_concept_or_scope")
                continue
            score = entry.get("similarity", entry.get("score"))
            has_score = score is not None
            semantics = entry.get("score_semantics", payload.get("score_semantics", "none"))
            if (has_score and (not _finite(score) or semantics not in SEMANTICS
                              or semantics == "none")) or (
                item.capability == "classification" and not has_score
            ):
                reject("missing_or_invalid_score_semantics")
                continue
            if has_score and ((semantics in {"probability", "uncalibrated_independent_sigmoid"}
                               and not 0 <= score <= 1)
                              or (semantics == "relative_similarity" and not -1 <= score <= 1)):
                reject("score_out_of_range")
                continue
            uncertainty = entry.get("uncertainty")
            if uncertainty is not None and (not _finite(uncertainty) or not 0 <= uncertainty <= 1):
                reject("invalid_uncertainty")
                continue
            spatial = item.capability != "classification"
            region = np.zeros((grid, grid), dtype=np.float32)
            box, area = [0.0] * 4, 0.0
            if spatial:
                if "mask" in entry:
                    mask = _mapped_mask(entry, payload, provenance, image_size)
                    if mask is None:
                        reject("invalid_empty_or_unmapped_mask")
                        continue
                    raw_box = mask.getbbox()
                else:
                    bounds = _bbox(entry.get("bbox_xyxy_normalized_original_image"))
                    if bounds is None:
                        reject("explicit_original_image_box_required")
                        continue
                    raw_box = [bounds[i] * image_size[i % 2] for i in range(4)]
                    # Rasterize coverage, retaining subpixel/small boxes on the patch grid.
                    x0, y0, x1, y1 = [(v + (left if i % 2 == 0 else top)) / side
                                      for i, v in enumerate(raw_box)]
                    edges = np.arange(grid + 1) / grid
                    xs = np.maximum(0, np.minimum(edges[1:], x1) - np.maximum(edges[:-1], x0))
                    ys = np.maximum(0, np.minimum(edges[1:], y1) - np.maximum(edges[:-1], y0))
                    region = (ys[:, None] * xs[None, :] * grid**2).astype(np.float32)
                    mask = None
                box = [(v + (left if i % 2 == 0 else top)) / side
                       for i, v in enumerate(raw_box)]
                if mask is not None:
                    # Float BOX resize preserves tiny regions; uint8 rounding may erase them.
                    canvas = np.zeros((side, side), dtype=np.float32)
                    canvas[top:top + height, left:left + width] = np.asarray(mask) / 255.0
                    region = np.asarray(Image.fromarray(canvas).resize(
                        (grid, grid), Image.Resampling.BOX), dtype=np.float32)
                area = float(region.mean())
                if area <= 0:
                    reject("region_outside_grid")
                    continue
            records.append((KINDS[item.capability], contract.concepts.index(binding["concept"]),
                            contract.scopes.index(binding["scope_id"]),
                            SEMANTICS[semantics] if has_score else 0,
                            [float(score or 0), float(has_score), float(uncertainty or 0),
                             float(uncertainty is not None), *box, area, float(spatial)],
                            region.reshape(-1), source))
    # Canonical ordering makes budget truncation independent of arrival/expert order.
    records.sort(key=lambda r: (r[:4], tuple(r[4]), r[5].tobytes(), r[6]))
    for r in records[contract.max_records:]:
        rejected.append({"expert_id": r[6][0], "evidence_id": r[6][1], "reason": "token_budget"})
    records = records[:contract.max_records]
    return TensorPacket(
        *(np.asarray([r[i] for r in records], dtype=np.int64) for i in range(4)),
        np.asarray([r[4] for r in records], dtype=np.float32).reshape(-1, 10),
        np.asarray([r[5] for r in records], dtype=np.float32).reshape(-1, grid * grid),
        tuple(r[6] for r in records), tuple(rejected),
    )
