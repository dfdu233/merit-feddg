"""One Bit Flip: reacquire pixels before steering, not another decoding gate.

Opt-in fixed-grid experiment. A predicted mask selects ONE region in stored
order. The same frozen encoder/projector re-encodes it. Local feature cells are
area-aligned back to the original grid; token count, original image and language
prompt stay fixed. No weights, thresholds or policies are fitted.

The 2x2 controls separate native-pixel detail from location. A displaced region
is NOT asserted to be medically irrelevant. Low-resolution reconstruction is a
controlled pixel degradation, not an exact inverse of CLIP preprocessing.
"""
from __future__ import annotations

import hashlib
import math
from contextlib import contextmanager
from time import perf_counter

import numpy as np
import torch
from PIL import Image

ARMS = ("compact", "deletion", "identity", "native_real", "low_real",
        "native_displaced", "low_displaced")
VIEW_ARMS = ARMS[3:]


def _size(size):
    if (len(size) != 2 or any(type(v) is not int or v < 2 for v in size)):
        raise ValueError("positive original width/height of at least two required")
    return tuple(size)


def pixel_hash(image):
    rgb = image.convert("RGB")
    return hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()


def region_plan(regions, image_size, vision_size):
    """First usable predicted mask, without answer, model-score or label ranking.

    Geometry threshold .5 is a fixed mask convention, not clinical confidence.
    Controls rigidly translate the same mask block; neither pixels nor mask
    orientation are reflected. Both boxes remain within fully nonpadding cells.
    """
    width, height = _size(image_size)
    if type(vision_size) is not int or vision_size < 2:
        raise ValueError("integer encoder pixel size required")
    values = np.asarray(regions, dtype=np.float64)
    if (values.ndim != 2 or values.shape[1] < 4 or not np.isfinite(values).all()
            or (values < 0).any() or (values > 1).any()):
        raise ValueError("finite K x square-grid predicted masks in [0,1] required")
    grid = math.isqrt(values.shape[1])
    if grid * grid != values.shape[1]:
        raise ValueError("square patch grid required")
    side = max(width, height)
    audit = {"selection": "first_usable_stored_mask", "rejected": [],
             "mask_cutoff": .5, "source_size_wh": [width, height],
             "vision_size": vision_size, "grid": grid,
             "controls_are_clinical_negatives": False}
    if side <= vision_size:
        return None, {**audit, "reason": "no_native_resolution_advantage"}
    left, top = (side-width)//2, (side-height)//2
    edge = np.arange(grid + 1) * side / grid
    xs = np.flatnonzero((edge[:-1] >= left) & (edge[1:] <= left+width))
    ys = np.flatnonzero((edge[:-1] >= top) & (edge[1:] <= top+height))
    if not len(xs) or not len(ys):
        return None, {**audit, "reason": "no_fully_observed_patch"}
    valid = np.zeros((grid, grid), dtype=bool)
    valid[np.ix_(ys, xs)] = True
    for index, value in enumerate(values):
        mask = value.reshape(grid, grid)
        foreground = (mask > .5) & valid
        if not foreground.any():
            audit["rejected"].append({"index": index, "reason": "no_mask_foreground"})
            continue
        y, x = np.nonzero(foreground)
        x0, x1, y0, y1 = int(x.min()), int(x.max()+1), int(y.min()), int(y.max()+1)
        # Opposite window position in valid rectangle, without reversing contents.
        cx0, cy0 = int(xs[0]+xs[-1]+1-x1), int(ys[0]+ys[-1]+1-y1)
        cx1, cy1 = cx0+x1-x0, cy0+y1-y0
        if (cx0, cy0) == (x0, y0):
            audit["rejected"].append({"index": index, "reason": "coincident_displaced_box"})
            continue
        support = np.zeros_like(mask)
        support[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
        moved = np.zeros_like(mask)
        moved[cy0:cy1, cx0:cx1] = support[y0:y1, x0:x1]
        boxes = [[round(v*side/grid) for v in box]
                 for box in ((x0, y0, x1, y1), (cx0, cy0, cx1, cy1))]
        # Rounding grid cells may differ by one pixel. Equal crop shape is an
        # experimental requirement, not a best-effort area match.
        sizes = [(b[2]-b[0], b[3]-b[1]) for b in boxes]
        if sizes[0] != sizes[1] or min(sizes[0]) <= 0 or max(sizes[0]) >= side:
            audit["rejected"].append({"index": index, "reason": "unmatched_or_full_frame_box"})
            continue
        a, b = boxes
        intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
        area = sizes[0][0]*sizes[0][1]
        plan = {"grid": grid, "side": side, "vision_size": vision_size,
                "image_size": [width, height], "region_index": index,
                "real_box": a, "displaced_box": b,
                "real_support": support.tolist(), "displaced_support": moved.tolist()}
        return plan, {**audit, "reason": "available", "region_index": index,
                      "real_box": a, "displaced_box": b,
                      "box_iou": intersection/(2*area-intersection),
                      "area_fraction_of_canvas": area/(side*side),
                      "linear_magnification": side/max(sizes[0]),
                      "input_region_sha256": hashlib.sha256(value.tobytes()).hexdigest()}
    return None, {**audit, "reason": "no_nondegenerate_matched_region"}


def region_views(image, plan, background):
    """Same boxes/framing at two source-detail levels; never changes source pixels."""
    rgb = image.convert("RGB")
    if list(rgb.size) != plan["image_size"] or max(rgb.size) != plan["side"]:
        raise ValueError("plan/source geometry mismatch")
    if (len(background) != 3 or any(type(v) is not int or not 0 <= v <= 255 for v in background)):
        raise ValueError("explicit RGB padding required")
    side, resolution = plan["side"], plan["vision_size"]
    canvas = Image.new("RGB", (side, side), tuple(background))
    canvas.paste(rgb, ((side-rgb.width)//2, (side-rgb.height)//2))
    low = canvas.resize((resolution, resolution), Image.Resampling.BICUBIC).resize(
        (side, side), Image.Resampling.BICUBIC)
    views = {}
    for location in ("real", "displaced"):
        box = plan[location+"_box"]
        views["native_"+location] = canvas.crop(box)
        views["low_"+location] = low.crop(box)
    audit = {"source_pixel_sha256": pixel_hash(rgb), "views": {}}
    for name, view in views.items():
        audit["views"][name] = {"size_wh": list(view.size), "pixel_sha256": pixel_hash(view)}
    for location in ("real", "displaced"):
        a = np.asarray(views["native_"+location], dtype=np.float32)
        b = np.asarray(views["low_"+location], dtype=np.float32)
        audit[location+"_mean_abs_pixel_difference"] = float(np.abs(a-b).mean())
    return views, audit


def _overlap_axis(grid, side, begin, end, local_side, padding):
    """Exact 1-D cell-area overlap; no guessed bilinear positional correspondence."""
    target = np.arange(grid+1, dtype=np.float64)*side/grid
    lo = np.maximum(target[:-1], begin)-begin+padding
    hi = np.minimum(target[1:], end)-begin+padding
    source = np.arange(grid+1, dtype=np.float64)*local_side/grid
    overlap = np.maximum(0, np.minimum(hi[:, None], source[1:])-np.maximum(lo[:, None], source[:-1]))
    overlap[hi <= lo] = 0
    total = overlap.sum(1, keepdims=True)
    return np.divide(overlap, total, out=np.zeros_like(overlap), where=total > 0)


def align_features(local, box, *, side, grid):
    """Map a padded crop's fixed-grid features back to original patch locations.

    Uses one model's frozen features; no cross-model latent alignment is claimed.
    Area aggregation may itself lose detail. This is part of the hypothesis test.
    """
    if (not isinstance(local, torch.Tensor) or local.ndim != 3 or local.shape[0] != 1
            or local.shape[1] != grid*grid or not local.is_floating_point()
            or not torch.isfinite(local).all()):
        raise ValueError("one finite projected patch grid required")
    if (len(box) != 4 or any(type(v) is not int for v in box)
            or not 0 <= box[0] < box[2] <= side or not 0 <= box[1] < box[3] <= side):
        raise ValueError("valid square-canvas pixel box required")
    x0, y0, x1, y1 = box
    width, height = x1-x0, y1-y0
    crop_side = max(width, height)
    ax = torch.as_tensor(_overlap_axis(grid, side, x0, x1, crop_side, (crop_side-width)//2),
                         device=local.device, dtype=torch.float32)
    ay = torch.as_tensor(_overlap_axis(grid, side, y0, y1, crop_side, (crop_side-height)//2),
                         device=local.device, dtype=torch.float32)
    image_features = local[0].float().reshape(grid, grid, -1)
    return torch.einsum("ij,jkd,lk->ild", ay, image_features, ax).reshape_as(local).to(local.dtype)


def fuse_features(original, local_aligned, support):
    """Fixed unit-mass convex fusion. Identity outside support, same token count."""
    if (original.shape != local_aligned.shape or original.ndim != 3 or original.shape[0] != 1
            or original.dtype != local_aligned.dtype or original.device != local_aligned.device
            or not original.is_floating_point() or not torch.isfinite(original).all()
            or not torch.isfinite(local_aligned).all()):
        raise ValueError("same finite original/local tensor contract required")
    mass = torch.as_tensor(support, device=original.device, dtype=torch.float32)
    if (mass.numel() != original.shape[1] or not torch.isfinite(mass).all()
            or (mass < 0).any() or (mass > 1).any()):
        raise ValueError("one finite mask weight per patch required")
    mass = mass.reshape(1, -1, 1)
    if not mass.any():
        return original
    mixed = ((original.float()+mass*local_aligned.float())/(1+mass)).to(original.dtype)
    return torch.where(mass > 0, mixed, original)


@contextmanager
def projected_view(projector, local_aligned, support):
    """Single scoped projector hook, restored on all exits; no per-token branches."""
    marker = "_merit_reencoding_active"
    if getattr(projector, marker, False):
        raise RuntimeError("nested/shared reencoding hook is unsupported")
    audit = {"projector_calls": 0, "max_abs_feature_delta": 0.0}
    def apply(_module, _inputs, output):
        result = fuse_features(output, local_aligned, support)
        audit["projector_calls"] += 1
        audit["max_abs_feature_delta"] = max(audit["max_abs_feature_delta"],
            float((result-output).abs().max()))
        return result
    setattr(projector, marker, True)
    handle = None
    try:
        handle = projector.register_forward_hook(apply)
        yield audit
    finally:
        if handle is not None:
            handle.remove()
        delattr(projector, marker)


def factorial_contrasts(scores):
    """Paired score contrasts, not automatic method selection or causal proof."""
    if any(k not in scores or not math.isfinite(scores[k]) for k in VIEW_ARMS):
        raise ValueError("all four finite paired scores required")
    detail_real = scores["native_real"]-scores["low_real"]
    detail_displaced = scores["native_displaced"]-scores["low_displaced"]
    return {"detail_at_real": detail_real, "detail_at_displaced": detail_displaced,
            "location_at_native": scores["native_real"]-scores["native_displaced"],
            "detail_location_interaction": detail_real-detail_displaced}


def reencoding_case(probe, row, historical, protocol):
    """Live LLaVA-Med adapter. Source-binding audit reuses the CRES protocol.

    Historical methods/files are untouched. All arms receive the same original
    image and uniform 64-token answer contract; reencoding arms use deletion's
    exact other evidence. No target answer or target mask is inspected.
    """
    from dataclasses import replace
    from .capabilities import EvidenceItem
    from .capability_runtime import NativeSession, NativeState, ValueGenerationConfig
    from .control_study import prompt_config, reuse, validate_outputs
    from .matched_evaluation import generation_prompt

    started = perf_counter()
    if any(k in row for k in ("answer", "answers", "reference", "references", "label", "mask")):
        raise ValueError("reference-bearing inference row")
    model, tower = probe.model, probe.model.get_vision_tower()
    if (model.config.image_aspect_ratio != "pad" or not probe.deterministic_image_padding
            or tower.select_feature != "patch" or model.training):
        raise ValueError("frozen fixed-grid LLaVA with deterministic square padding required")
    grid = math.isqrt(int(tower.num_patches))
    resolution = int(tower.config.image_size)
    if grid*grid != tower.num_patches:
        raise ValueError("unsupported dynamic visual grid")
    projector = model.get_model().mm_projector
    source_cfg = ValueGenerationConfig(**historical["generation_config"])
    if (source_cfg.semantic_spatial or source_cfg.vector_gate != "off"
            or source_cfg.evidence_style != "semantic" or not source_cfg.compact_native):
        raise ValueError("ungated compact native source cache required")
    native = tuple(EvidenceItem(**x) for x in historical["evidence"])
    if any(e.provenance.get(k) for e in native for k in ("target_mask_used", "target_masks_used", "target_answers_used")):
        raise ValueError("target annotations forbidden")
    old = NativeSession(probe, row["image"], generation_prompt(row, protocol["config"]), row["question"], source_cfg)
    old.context(NativeState(items=native))
    recorded = next((x["evidence_transport"] for x in reversed(historical.get("trace", []))
                     if x.get("event") == "decode" and x.get("evidence_transport")), None)
    if not recorded or any(k not in recorded or recorded[k] != old.last_transport.get(k) for k in
            ("prompt_sha256", "evidence_sha256", "presented", "omitted", "context")):
        raise ValueError("source cache delivery binding mismatch")
    cfg = replace(source_cfg, max_new_tokens=64, block_tokens=64, behavior_probe="off", uncertainty_from_probe=False)
    prompt = generation_prompt(row, prompt_config(protocol["config"], "uniform"))
    text = NativeSession(probe, row["image"], prompt, row["question"], cfg)
    before = perf_counter()
    block = text.propose(NativeState(items=native), 64)
    arms = {"compact": {"text": text.decode(block.tokens).strip(), "token_ids": list(block.tokens),
                        "seconds": perf_counter()-before}}
    visible = {(v["expert_id"], v["evidence_id"]) for v in text.last_transport["presented"]}
    retained = tuple(e for e in native if (e.expert_id, e.evidence_id) in visible)
    masks = tuple(e for e in retained if e.capability == "segmentation")
    other = tuple(e for e in retained if e.capability != "segmentation")
    context = NativeSession(probe, row["image"], prompt, row["question"], cfg)
    image, base_prompt = context.context(NativeState(items=other))
    if {(v["expert_id"], v["evidence_id"]) for v in context.last_transport["presented"]} != {
            (e.expert_id, e.evidence_id) for e in other}:
        raise ValueError("deletion changed nonintervened evidence delivery")
    def answer():
        start = perf_counter()
        result = probe.new_answer_session(image, base_prompt).propose((), count=1, length=64)[0]
        return {"text": result.text.strip(), "token_ids": list(result.tokens),
                "seconds": perf_counter()-start}
    arms["deletion"] = answer()
    packet = probe.tensor_packet(masks, row["image"], question=row["question"], weighting="equal")
    with Image.open(row["image"]) as handle:
        if handle.getexif().get(274, 1) != 1:
            raise ValueError("nontrivial EXIF orientation is not supported")
        original = handle.convert("RGB").copy()
    if packet.regions.shape[1] != grid*grid:
        raise ValueError("packet grid differs from frozen vision grid")
    plan, audit = region_plan(packet.regions, original.size, resolution)
    result = {"id": row["id"], "arms": arms, "applicable": plan is not None,
              "region_audit": audit, "source_delivery_verified": True,
              "compact_transport": text.last_transport, "deletion_transport": context.last_transport,
              "new_expert_calls": 0, "selection_uses_labels": False, "max_new_tokens": 64,
              "visual_token_count": grid*grid, "clinical_gate": False}
    if plan is None:
        for name in ARMS[2:]:
            arms[name] = reuse(arms["deletion"], audit["reason"])
    else:
        index = plan["region_index"]
        result["region_source"] = {"source": list(packet.sources[index]), "label": packet.labels[index]}
        background = tuple(int(v*255) for v in probe.image_processor.image_mean)
        views, result["view_audit"] = region_views(original, plan, background)
        hidden = int(model.config.hidden_size)
        device = model.get_input_embeddings().weight.device
        dtype = next(projector.parameters()).dtype
        dummy = torch.zeros((1, grid*grid, hidden), device=device, dtype=dtype)
        with projected_view(projector, dummy, np.zeros((grid, grid))) as identity_audit:
            arms["identity"] = answer()
        result["identity_audit"] = dict(identity_audit)
        if identity_audit["projector_calls"] != 1 or arms["identity"]["token_ids"] != arms["deletion"]["token_ids"]:
            raise RuntimeError("zero-support production token parity failed")
        for name in VIEW_ARMS:
            start = perf_counter()
            inputs = probe._inputs(views[name], "")
            pixels = inputs["images"]
            if not isinstance(pixels, torch.Tensor) or tuple(pixels.shape) != (1, 3, resolution, resolution):
                raise ValueError("unexpected local vision preprocessing shape")
            encoded = model.encode_images(pixels)
            location = "displaced" if name.endswith("displaced") else "real"
            aligned = align_features(encoded, plan[location+"_box"], side=plan["side"], grid=grid)
            vision_seconds = perf_counter()-start
            with projected_view(projector, aligned, plan[location+"_support"]) as hook_audit:
                arms[name] = answer()
            if hook_audit["projector_calls"] != 1:
                raise RuntimeError("expected exactly one decoder visual prefill")
            arms[name].update(vision_seconds=vision_seconds, decode_seconds=arms[name]["seconds"],
                extra_vision_encodes=1, per_token_auxiliary_branches=0, hook_audit=dict(hook_audit))
            arms[name]["seconds"] += vision_seconds
            del encoded, aligned, inputs
    result["wall_seconds"] = perf_counter()-started
    validate_outputs(result, ARMS)
    return result
