"""Frozen-model spatial evidence: no optimizer, random vectors, or fitted weights.

Content comes exclusively from the VLM's existing projected patch features.
Native geometry defines a deterministic relation operator (ProxyCLIP-inspired,
not a reproduction). Class scores without a spatial support map are rejected.
"""

import base64
import hashlib
import zlib
from collections import defaultdict
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from torch import nn

from .native_evidence import _bbox, _label, _tokens
from .tensor_evidence import TensorContract, compile_tensor_evidence


def encode_soft_mask(values):
    values = np.asarray(values, dtype="<f4")
    if (values.ndim != 2 or not values.size or values.size > 16_777_216
            or not np.isfinite(values).all() or (values < 0).any() or (values > 1).any()):
        raise ValueError("soft mask must be finite 2-D values in [0,1]")
    return {"encoding": "float32-zlib-base64", "size": list(values.shape),
            "data": base64.b64encode(zlib.compress(values.tobytes())).decode("ascii")}


def mapped_soft_mask(entry, payload, image_size):
    value = entry.get("soft_mask", {})
    if not isinstance(value, dict):
        raise TypeError("soft mask must be an encoded object")
    size = value.get("size", [])
    if (value.get("encoding") != "float32-zlib-base64" or len(size) != 2
            or any(type(v) is not int or v < 1 for v in size)
            or size[0] * size[1] > 16_777_216):
        raise ValueError("invalid soft-mask encoding/size")
    expected = size[0] * size[1] * 4
    decoder = zlib.decompressobj()
    try:
        raw = decoder.decompress(base64.b64decode(value["data"], validate=True), expected + 1)
    except zlib.error as exc:
        raise ValueError("invalid compressed soft mask") from exc
    if len(raw) != expected or not decoder.eof or decoder.unused_data:
        raise ValueError("soft-mask byte length mismatch")
    values = np.frombuffer(raw, dtype="<f4").reshape(size)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("invalid soft-mask values")
    mask = Image.fromarray(values.copy())
    if entry.get("mask_coordinate_system") == "original_image":
        if mask.size != image_size:
            raise ValueError("soft mask does not match original image")
        return mask
    transform = payload.get("image_transform", {})
    if not isinstance(transform, dict):
        raise TypeError("invalid soft-mask transform")
    width, height = image_size
    crop = _bbox(transform.get("crop_box_xyxy_normalized"))
    if (entry.get("mask_coordinate_system") != "model_grid_of_center_crop" or crop is None
            or transform.get("original_size_hw") != [height, width]
            or transform.get("model_size_hw") != size):
        raise ValueError("unmapped soft mask")
    left, top, right, bottom = [round(v * n) for v, n in zip(crop, image_size * 2)]
    if right <= left or bottom <= top:
        raise ValueError("empty crop")
    canvas = Image.new("F", image_size)
    canvas.paste(mask.resize((right - left, bottom - top), Image.Resampling.BILINEAR), (left, top))
    return canvas


def spatial_packet(items, image_size, *, grid_size=24, max_records=64, question="",
                   weighting="relevance"):
    """Names are audit identifiers, never learned embeddings or tokenized reports."""
    if weighting not in {"equal", "relevance"}:
        raise ValueError("spatial weighting must be equal or relevance")
    supported, bindings, rejected = [], [], []
    for item in items:
        payload = dict(item.payload)
        if item.capability == "classification":
            entries = payload.get("catalog", payload.get("findings", []))
            spatial = []
            for entry in entries:
                support = entry.get("spatial_support") if isinstance(entry, dict) else None
                if isinstance(support, dict) and ("soft_mask" in support or "mask" in support):
                    spatial.append({**entry, **support})
                else:
                    rejected.append({"expert_id": item.expert_id, "evidence_id": item.evidence_id,
                                     "label": _label(entry) if isinstance(entry, dict) else "",
                                     "reason": "classification_without_spatial_support"})
            payload["structures"] = spatial
            item = replace(item, capability="segmentation", payload=payload)
        entries = [payload] if "mask" in payload or "soft_mask" in payload else []
        for key in ("structures", "detections", "boxes", "objects"):
            entries.extend(payload.get(key, []))
        for entry in entries:
            if isinstance(entry, dict):
                bindings.append({"expert": item.expert_id, "scope": item.scope,
                                 "label": _label(entry), "concept": _label(entry),
                                 "scope_id": item.scope})
        supported.append(item)
    # An empty catalog is legitimate for a case with no compatible evidence.
    unique = {(b["expert"], b["scope"], b["label"]): b for b in bindings}
    contract = TensorContract(tuple(sorted({b["concept"] for b in bindings})) or ("unused",),
                              tuple(sorted({b["scope_id"] for b in bindings})) or ("unused",),
                              tuple(unique.values()), grid_size, max_records)
    packet = compile_tensor_evidence(supported, image_size, contract)
    labels = tuple(contract.concepts[int(i)] for i in packet.concept)
    scopes = tuple(contract.scopes[int(i)] for i in packet.scope)
    # Stable identifiers survive extension of the observed vocabulary at runtime.
    stable = lambda s: int.from_bytes(hashlib.sha256(s.encode()).digest()[:8], "big") >> 1
    packet.concept = np.asarray([stable(s) for s in labels], dtype=np.int64)
    packet.scope = np.asarray([stable(s) for s in scopes], dtype=np.int64)
    packet.rejected = (*rejected, *packet.rejected)
    packet.labels = labels
    query = _tokens(question)
    raw = np.asarray([1.0 + (len(_tokens(label) & query) / max(1, len(_tokens(label)))
                            if weighting == "relevance" else 0.0) for label in labels])
    # Identical source/geometry records do not multiply a model's vote.
    groups = defaultdict(list)
    for i, source in enumerate(packet.sources):
        groups[source[0]].append(i)
    weights = np.zeros(len(packet), dtype=np.float32)
    for indices in groups.values():
        shapes = defaultdict(list)
        for i in indices:
            shapes[packet.regions[i].tobytes()].append(i)
        strengths = [max(raw[i] for i in ids) for ids in shapes.values()]
        total = sum(strengths)
        for ids, strength in zip(shapes.values(), strengths):
            for i in ids:
                weights[i] = strength / total / len(ids) / len(groups)
    packet.importance = weights
    packet.weighting = weighting
    return packet


class SpatialEvidenceBridge(nn.Module):
    """Parameter-free E=R H and spatial return operator; not semantic score fusion.

    Region weights allocate content within each source; sources share total mass.
    The original token has unit mass, so empty/outside regions are exact identity.
    """
    training_free = True

    def __init__(self, visual_dim, grid_size=24, max_records=64):
        super().__init__()
        if any(type(v) is not int or v <= 0 for v in (visual_dim, grid_size, max_records)):
            raise ValueError("positive spatial dimensions required")
        self.config = {"visual_dim": visual_dim}
        self.contract = SimpleNamespace(grid_size=grid_size, max_records=max_records)
        self.last_audit = {}

    def forward(self, visual, packet):
        self.last_audit = {"training_free": True, "records": len(packet), "max_abs_token_delta": 0.0}
        if visual.shape != (1, self.contract.grid_size**2, self.config["visual_dim"]):
            raise ValueError("spatial bridge requires one matched patch grid")
        if not len(packet):
            return visual
        regions = torch.as_tensor(packet.regions, device=visual.device, dtype=torch.float32)
        weights = torch.as_tensor(packet.importance, device=visual.device, dtype=torch.float32)
        if (regions.shape != (len(packet), visual.shape[1]) or weights.shape != (len(packet),)
                or not torch.isfinite(regions).all() or not torch.isfinite(weights).all()
                or (regions < 0).any() or (regions > 1).any() or (weights < 0).any()):
            raise ValueError("invalid spatial regions/importance")
        if not torch.isfinite(visual).all():
            raise ValueError("nonfinite visual features")
        if not weights.any():
            return visual
        region_sum = regions.sum(-1, keepdim=True)
        r = regions / region_sum.clamp_min(torch.finfo(torch.float32).tiny)
        h = visual[0].float()
        evidence = r @ h
        # Return only to the observed pixels; no dense NxN affinity allocation.
        assignment = regions.T * weights
        mass = assignment.sum(-1, keepdim=True)
        result = ((h + assignment @ evidence) / (1 + mass)).to(visual.dtype).unsqueeze(0)
        result = torch.where(mass[None] > 0, result, visual)
        self.last_audit = {"training_free": True, "records": len(packet),
                           "importance": weights.tolist(), "labels": list(packet.labels),
                           "weighting": packet.weighting,
                           "max_abs_token_delta": float((result - visual).abs().max()),
                           "weights_are_correctness_probabilities": False,
                           "class_scores_injected": False}
        return result
