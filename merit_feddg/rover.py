"""Source-only region intervention pilot. Stability is not a truth/DG certificate."""
from __future__ import annotations

import numpy as np

from .evidence_decode import _log_probs


def box_pixels(box, size):
    values = np.asarray(box, dtype=float)
    if values.shape != (4,) or not np.isfinite(values).all():
        raise ValueError("box must contain four finite normalized coordinates")
    x1, y1, x2, y2 = values
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise ValueError("invalid normalized xyxy box")
    w, h = size
    result = (int(x1*w), int(y1*h), int(np.ceil(x2*w)), int(np.ceil(y2*h)))
    if min(result[2]-result[0], result[3]-result[1]) < 8:
        raise ValueError("region smaller than 8 pixels")
    return result


def region_boxes(box, size):
    """Equal-size corner control, selected geometrically, never declared healthy."""
    x1, y1, x2, y2 = box_pixels(box, size)
    w, h = size
    bw, bh = x2-x1, y2-y1
    def intersection(candidate):
        a, b, c, d = candidate
        return max(0, min(c, x2)-max(a, x1))*max(0, min(d, y2)-max(b, y1))
    candidates = [(x, y, x+bw, y+bh) for x in (0, w-bw) for y in (0, h-bh)]
    control = min(candidates, key=intersection)
    if intersection(control) / (bw*bh) > 0.1:
        raise ValueError("no low-overlap same-size control; use a smaller predicted region")
    dx, dy = max(1, round(bw*.15)), max(1, round(bh*.15))
    expanded = (max(0, x1-dx), max(0, y1-dy), min(w, x2+dx), min(h, y2+dy))
    if expanded == (x1, y1, x2, y2):
        raise ValueError("expanded crop identical to region")
    return {"region": (x1, y1, x2, y2), "control": control, "expanded": expanded}


def js_divergence(p, q):
    mid = (p+q)/2
    def kl(a):
        keep = a > 0
        return float(np.sum(a[keep]*np.log(a[keep]/mid[keep])))
    return max(0., (kl(p)+kl(q))/2)


def region_distribution(base, region, control, expanded, weight=.5, checked=True):
    """Use positional specificity minus crop instability, in nats, to set weight.

    All four branches share one tokenizer and exact committed prefix. This is
    a fixed inference heuristic; thresholds are NOT fitted on target answers.
    """
    if not np.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError("weight must be in [0,1]")
    logs, masks = zip(*[_log_probs(x) for x in (base, region, control, expanded)])
    if any(x.shape != logs[0].shape for x in logs):
        raise ValueError("different vocabulary dimensions")
    if any(not np.array_equal(masks[0], m) for m in masks):
        raise ValueError("different vocabulary supports")
    b, r, c, e = [np.exp(x) for x in logs]
    specificity = js_divergence(r, c)
    instability = js_divergence(r, e)
    gate = max(0., (specificity-instability)/(specificity+instability+1e-12))
    effective = weight * (gate if checked else 1.)
    mixed = (1-effective)*b + effective*r
    scores = np.full_like(mixed, -np.inf)
    positive = mixed > 0
    scores[positive] = np.log(mixed[positive])
    return scores, {"specificity_js": specificity, "instability_js": instability,
                    "gate": gate, "weight": effective}


def decode_regions(sessions, mode, *, max_tokens=24, weight=.5):
    """Auditable replay decoder; intentionally prioritizes parity over speed.

    LLaVA's next_scores uses production forced-prefix replay. This does NOT
    claim incremental KV/vision caching or clinical claim-boundary detection.
    """
    if mode not in {"base", "crop", "control", "fusion", "rover"}:
        raise ValueError("unknown region mode")
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    prefix, trace = [], []
    for step in range(max_tokens):
        if mode in {"base", "crop", "control"}:
            branch = {"base": "base", "crop": "region", "control": "control"}[mode]
            scores = sessions[branch].next_scores(prefix)
            info = {"branch": branch, "weight": 0.}
        else:
            raw = {name: session.next_scores(prefix) for name, session in sessions.items()}
            scores, info = region_distribution(raw["base"], raw["region"], raw["control"],
                                               raw["expanded"], weight, mode == "rover")
        token = int(np.argmax(scores))
        trace.append({"step": step, "token": token, **info})
        prefix.append(token)
        if token in sessions["base"].eos_ids:
            break
    return {"text": sessions["base"].decode(prefix).strip(), "token_ids": prefix, "trace": trace}
