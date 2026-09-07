"""Frozen, output-typed sensitivity diagnostics; never correctness certificates."""

from __future__ import annotations

from time import perf_counter

import numpy as np
from PIL import Image

from .experts.base import load_rgb


def weak_views(image):
    """Photometric only: preserve geometry, not a guarantee of clinical invariance."""
    original = load_rgb(image)
    pixels = np.asarray(original, dtype=np.float32) / 255
    return [("original", original), *[
        (f"gamma_{gamma}", Image.fromarray(
            np.rint(np.clip(pixels ** gamma, 0, 1) * 255).astype(np.uint8)))
        for gamma in (0.95, 1.05)
    ]]


def summarize_outputs(outputs, capability, threshold=0.5):
    """Compare independent scores or aligned masks without softmax conversion."""
    arrays = [np.asarray(output, dtype=float) for output in outputs]
    if len(arrays) < 2 or any(a.shape != arrays[0].shape for a in arrays):
        raise ValueError("probe needs multiple identically shaped outputs")
    if any(not a.size or not np.isfinite(a).all() or ((a < 0) | (a > 1)).any()
           for a in arrays):
        raise ValueError("probe outputs must be finite independent scores in [0,1]")
    stacked = np.stack(arrays)
    if capability == "classification":
        if stacked.ndim != 2:
            raise ValueError("classification probe requires per-label vectors")
        per_label = np.abs(stacked[1:] - stacked[0]).max(axis=0)
        return {"metric": "max_per_label_absolute_score_change",
                "sensitivity": float(per_label.max()),
                "per_label_change": per_label.tolist(), "informative": True}
    if capability != "segmentation" or stacked.ndim != 4:
        raise ValueError("segmentation probe requires aligned C,H,W arrays")
    if not np.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("mask threshold must lie inside (0,1)")
    masks = stacked >= threshold
    changes, empty_pairs = [], 0
    for other in masks[1:]:
        intersection = (masks[0] & other).sum(axis=(-2, -1))
        union = (masks[0] | other).sum(axis=(-2, -1))
        empty_pairs += int((union == 0).sum())
        changes.extend((1 - intersection[union > 0] / union[union > 0]).tolist())
    return {"metric": "max_nonempty_mask_iou_distance",
            "sensitivity": max(changes) if changes else None,
            "empty_pairs": empty_pairs, "informative": bool(changes),
            "all_empty_channels": np.all(~masks, axis=(0, 2, 3)).tolist(),
            "mean_probability_change": float(np.abs(stacked[1:] - stacked[0]).mean())}


def probe_xrv(model, image, capability, structures=None, threshold=0.5):
    started = perf_counter()
    views = weak_views(image)
    outputs, labels, transform = [], None, None
    for _, view in views:
        current_labels, values, current_transform = (
            model.classify(view) if capability == "classification" else model.segment(view)
        )
        current_labels = list(current_labels)
        if labels is not None and (current_labels != labels or current_transform != transform):
            raise ValueError("probe output labels or spatial transforms changed")
        labels, transform = current_labels, current_transform
        values = np.asarray(values)
        if len(values) != len(labels):
            raise ValueError("probe labels and outputs do not align")
        if structures is not None:
            if not structures or len(set(structures)) != len(structures):
                raise ValueError("probe structures must be nonempty and unique")
            values = values[[labels.index(label) for label in structures]]
        outputs.append(values)
    return {**summarize_outputs(outputs, capability, threshold),
            "status": "measured", "labels": list(structures or labels),
            "views": [name for name, _ in views], "extra_model_calls": len(views),
            "seconds": perf_counter() - started,
            "scope": "photometric_sensitivity_not_correctness_or_domain_certificate",
            "clinical_invariance_verified": False}
