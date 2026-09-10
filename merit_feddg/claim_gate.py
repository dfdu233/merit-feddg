"""Paired local-removal diagnostic inspired by DnR and ERASER.

Measures answer sensitivity to predicted support, not medical correctness.
The frozen generalist still scores answers; no NLI/self-reported confidence is
used. A spatially shifted mask is an imperfect control, never a clinical
counterfactual. Semantic-only evidence cannot be verified by this diagnostic.
"""
from dataclasses import replace

import numpy as np
from PIL import Image, ImageFilter

from .experts.base import load_rgb
from .vector_gate import VectorGateConfig, assess_visual_contrast


def paired_local_views(image, regions):
    native = load_rgb(image)
    rows = np.asarray(regions, dtype=np.float32)
    if rows.ndim != 2 or not len(rows) or not np.isfinite(rows).all():
        raise ValueError("invalid native support")
    side = round(rows.shape[1] ** .5)
    if side * side != rows.shape[1] or rows.min() < 0 or rows.max() > 1:
        raise ValueError("support must be a bounded square grid")
    grid_mask = rows.max(axis=0).reshape(side, side)
    # TensorPacket uses the square-padded VLM grid. Undo padding before
    # perturbing original pixels; direct rectangular resize mislocalizes ROIs.
    width, height = native.size
    padded_side = max(native.size)
    left, top = (padded_side-width)//2, (padded_side-height)//2
    padded = Image.fromarray(grid_mask).resize((padded_side, padded_side), Image.Resampling.NEAREST)
    mask = np.asarray(padded, dtype=np.float32)[top:top+height, left:left+width].copy()
    if mask.max() == 0:
        raise ValueError("empty support")
    # Translation preserves the original-pixel mask's mass/distribution. Choose the least
    # overlapping of three fixed controls without consulting any model output.
    controls = [np.roll(mask, shift, axis=(0, 1)) for shift in
                ((height // 2, 0), (0, width // 2), (height // 2, width // 2))]
    control = min(controls, key=lambda value: float((mask * value).sum()))
    if np.allclose(mask, control):
        raise ValueError("support has no distinct translated control")
    blurred = native.filter(ImageFilter.GaussianBlur(radius=max(native.size) / 32))
    def apply(value):
        alpha = Image.fromarray(value.astype(np.float32))
        a = np.asarray(alpha)[..., None]
        pixels = np.asarray(native, dtype=np.float32) * (1-a) + np.asarray(blurred, dtype=np.float32) * a
        return Image.fromarray(np.rint(pixels).clip(0, 255).astype(np.uint8))
    return apply(mask), apply(control), {
        "support_mass": float(mask.sum()), "control_mass": float(control.sum()),
        "control_overlap": float((mask * control).sum()), "grid_size": side, "mass_coordinate_system": "original_image_pixels",
        "perturbation": "local_gaussian_blur", "control": "fixed_half_image_translation_min_overlap",
        "clinical_counterfactual_validated": False}


def assess_claim_support(session, state, item):
    packet = session.probe.tensor_packet((item,), session.image)
    if not len(packet):
        return {"accepted": True, "reason": "semantic_only_unverified", "visual_status": "unknown",
                "correctness_guaranteed": False, "gate_trained": False,
                "spatial_rejected": list(packet.rejected)}
    try:
        removed, control, geometry = paired_local_views(session.image, packet.regions)
    except ValueError as exc:
        return {"accepted": True, "reason": "spatial_test_unavailable", "visual_status": "unknown",
                "detail": str(exc), "correctness_guaranteed": False, "gate_trained": False,
                "spatial_transport_enabled": False}
    candidate = replace(state, items=state.items + (item,))
    # Delta = margin(control removal) - margin(support removal).
    # Reuse sequence scoring only, NOT the legacy mean-color decision protocol.
    audit = assess_visual_contrast(session._evidence_session(state), session._evidence_session(candidate),
        session.probe.new_answer_session(control, session.prompt),
        session.probe.new_answer_session(removed, session.prompt), state.prefix,
        config=VectorGateConfig(session.config.vector_gate_probe_tokens, session.config.vector_gate_min_gain),
        remaining_tokens=session.config.max_new_tokens-len(state.prefix))
    audit.update(schema="native-entry-local-removal-v1", visual_status="measured", geometry=geometry)
    audit.pop("dimensions", None)
    if "support" in audit:
        audit["paired_scores"] = [{"control_removal_logp": v["image_mean_logp"],
                                    "support_removal_logp": v["control_mean_logp"]} for v in audit.pop("support")]
        audit["control_removal_margin"] = audit.pop("image_gain")
        audit["support_removal_margin"] = audit.pop("control_gain")
        audit["local_removal_gain"] = audit.pop("gain")
        audit["reason"] = "positive_local_removal_gain" if audit["accepted"] else "no_positive_local_removal_gain"
    return audit
