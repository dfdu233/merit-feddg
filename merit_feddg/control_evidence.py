"""Control-referenced evidence steering (CRES), an uncalibrated research policy.

All distributions are from one frozen VLM at the SAME committed prefix. Native
expert scores are never treated as token logits or correctness probabilities.
The control envelope is deterministic sensitivity analysis, not a confidence
interval, causal uplift estimate, or conformal guarantee.
"""
from __future__ import annotations

import copy
import hashlib
import math
from time import perf_counter

import numpy as np


def _vector(value):
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 1 or not result.size or not np.isfinite(result).all():
        raise ValueError("finite nonempty vocabulary vector required")
    return result


def _logp(scores):
    shifted = scores - scores.max()
    return shifted - np.log(np.exp(shifted).sum())


def residual_direction(base, evidence, controls):
    """Keep only direction shared against deletion AND every supplied control.

    Center each log ratio under p0 to remove arbitrary branch logit offsets.
    A control-only effect cannot steer when evidence == base. Adding controls
    can only shrink each coordinate's magnitude, never turn uncertainty into a
    stronger update. Correlated controls do not accumulate confidence.
    """
    base, evidence = _vector(base), _vector(evidence)
    controls = tuple(_vector(c) for c in controls)
    if evidence.shape != base.shape or any(c.shape != base.shape for c in controls):
        raise ValueError("all branches must share the same vocabulary")
    if not controls:
        return np.zeros_like(base)
    probability = np.exp(_logp(base))
    differences = np.stack([evidence - comparator for comparator in (base, *controls)])
    differences -= (differences @ probability)[:, None]
    if not np.isfinite(differences).all():
        raise ValueError("nonfinite centered log ratio")
    lower, upper = differences.min(axis=0), differences.max(axis=0)
    return np.where(lower > 0, lower, np.where(upper < 0, upper, 0.0))


def constrained_scores(base, direction, *, max_strength=1.0, kl_budget=0.05):
    """Largest alpha in [0,max_strength] on p_alpha ∝ p0 exp(alpha*r).

    D_KL(p_alpha || p0) is monotone because its derivative is
    alpha * Var_{p_alpha}(r). Bisection returns the feasible endpoint. This
    bounds a soft distribution, NOT the correctness of greedy argmax outputs.
    """
    if (not math.isfinite(max_strength) or not 0 <= max_strength <= 1
            or not math.isfinite(kl_budget) or kl_budget < 0):
        raise ValueError("finite strength in [0,1] and nonnegative KL budget required")
    base, direction = _vector(base), _vector(direction)
    if base.shape != direction.shape:
        raise ValueError("direction vocabulary mismatch")
    logp0 = _logp(base)

    def at(alpha):
        scores = base + alpha * direction
        logpa = _logp(scores)
        kl = max(0.0, float(np.exp(logpa) @ (logpa - logp0)))
        if not math.isfinite(kl):
            raise ValueError("nonfinite KL")
        return scores, kl

    if kl_budget == 0 or max_strength == 0 or not np.any(direction):
        return base.copy(), {"strength": 0.0, "kl": 0.0, "budget": kl_budget}
    scores, kl = at(max_strength)
    alpha = max_strength
    if kl > kl_budget:
        low, high = 0.0, max_strength
        for _ in range(48):
            middle = (low + high) / 2
            if at(middle)[1] <= kl_budget:
                low = middle
            else:
                high = middle
        alpha = low
        scores, kl = at(alpha)
    return scores, {"strength": alpha, "kl": kl, "budget": kl_budget}


def spatial_controls(packet, image_size):
    """Translate all regions together within fully non-padding patch cells.

    Two fixed cyclic half-shifts preserve each region's value multiset, source
    weights, and pairwise overlaps. Boundary/padding cells stay fixed. These are
    deliberately broken geometry correspondences, not realistic anatomy or
    randomization-test samples. Degenerate/duplicate controls are unavailable.
    Original packet and native numeric metadata are never modified. The cloned
    packet is consumed only by SpatialEvidenceBridge (regions + importance).
    """
    regions = np.asarray(packet.regions)
    if (regions.ndim != 2 or not regions.shape[1] or not np.isfinite(regions).all()
            or (regions < 0).any() or (regions > 1).any()):
        raise ValueError("finite patch regions in [0,1] required")
    grid = math.isqrt(regions.shape[1])
    if grid * grid != regions.shape[1]:
        raise ValueError("square patch grid required")
    if len(image_size) != 2 or any(type(v) is not int or v <= 0 for v in image_size):
        raise ValueError("positive original image dimensions required")
    width, height = image_size
    side = max(width, height)
    left, top = (side - width) // 2, (side - height) // 2
    edges = np.arange(grid + 1) * side / grid
    xs = np.flatnonzero((edges[:-1] >= left) & (edges[1:] <= left + width))
    ys = np.flatnonzero((edges[:-1] >= top) & (edges[1:] <= top + height))
    if len(xs) < 2 or len(ys) < 2 or not len(regions):
        return (), {"reason": "insufficient_nonpadding_geometry", "controls": []}
    original = regions.reshape(-1, grid, grid)
    controls, audit, seen = [], [], {regions.tobytes()}
    for name, axis, shift in (("horizontal_half_shift", 2, len(xs) // 2),
                              ("vertical_half_shift", 1, len(ys) // 2)):
        values = original.copy()
        interior = original[:, ys[:, None], xs]
        values[:, ys[:, None], xs] = np.roll(interior, shift, axis=axis)
        moved = values.reshape(regions.shape)
        identity = moved.tobytes()
        if identity in seen:
            continue
        seen.add(identity)
        synthetic = copy.deepcopy(packet)
        synthetic.regions = moved
        synthetic.control_kind = name
        controls.append(synthetic)
        audit.append({"kind": name, "shift": shift,
                      "region_sha256": hashlib.sha256(identity).hexdigest(),
                      "changed_cells": int(np.count_nonzero(moved != regions))})
    # Require both fixed controls; do not silently make the test easier on a case.
    available = len(controls) == 2
    return (tuple(controls) if available else ()), {
        "reason": "available" if available else "degenerate_control_family",
        "controls": audit, "padding_and_boundary_cells_fixed": True,
        "original_region_sha256": hashlib.sha256(regions.tobytes()).hexdigest(),
        "controls_are_independent_samples": False}


def decode_controlled(base, evidence, controls, *, max_tokens, eos_ids,
                      max_strength=1.0, kl_budget=0.05, mode="robust"):
    """Production score replay; no gradients, sampling, or fitted parameters.

    mode=raw is the explicit no-control, KL-only ablation. Runtime failures
    propagate and are not reclassified as epistemic abstention.
    """
    if type(max_tokens) is not int or max_tokens < 1 or mode not in {"robust", "raw"}:
        raise ValueError("invalid decoder options")
    constrained_scores([0.0], [0.0], max_strength=max_strength, kl_budget=kl_budget)
    controls = tuple(controls)
    start = perf_counter()
    if max_strength == 0 or kl_budget == 0 or (mode == "robust" and not controls):
        block = base.propose((), count=1, length=max_tokens)[0]
        return {"text": block.text.strip(), "token_ids": list(block.tokens),
                "seconds": perf_counter() - start, "steps": [], "score_calls": 0,
                "guidance_applied": False, "reason": "zero_budget_or_missing_controls"}
    prefix, steps = [], []
    for _ in range(max_tokens):
        committed = tuple(prefix)
        a, b = _vector(base.next_scores(committed)), _vector(evidence.next_scores(committed))
        if a.shape != b.shape:
            raise ValueError("evidence vocabulary mismatch")
        null = [session.next_scores(committed) for session in controls] if mode == "robust" else []
        r = residual_direction(a, b, null) if mode == "robust" else b - a
        scores, audit = constrained_scores(a, r, max_strength=max_strength, kl_budget=kl_budget)
        token = int(np.argmax(scores))
        steps.append({**audit, "base_top": int(np.argmax(a)), "evidence_top": int(np.argmax(b)),
                      "selected": token, "residual_nonzero": int(np.count_nonzero(r)),
                      "residual_linf": float(np.abs(r).max())})
        prefix.append(token)
        if token in eos_ids:
            break
    return {"text": base.decode(prefix).strip(), "token_ids": prefix,
            "seconds": perf_counter() - start, "steps": steps,
            "score_calls": len(steps) * (2 + (len(controls) if mode == "robust" else 0)),
            "guidance_applied": any(s["strength"] > 0 for s in steps),
            "cache_policy": "same-prefix production replay", "mode": mode,
            "strength_is_correctness_probability": False}
