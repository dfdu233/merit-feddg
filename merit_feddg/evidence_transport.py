"""Whole-record packing with actual full-prompt token accounting.

Packing is deterministic in caller-provided order. It is not a learned utility
policy. A presented record is an input intervention, not proof of model usage.
"""

import hashlib
import json


def record_source(record):
    return {key: record[key] for key in ("expert_id", "evidence_id")}


def pack_records(records, render, measure, *, max_chars, reserve_tokens, companion_render=None):
    selected, rejected = [], []
    base = measure(render([]), reserve_tokens)
    if not base["fits"]:
        raise ValueError("original prompt and reserved answer exceed model context")
    if companion_render is not None and not measure(companion_render([]), reserve_tokens)["fits"]:
        raise ValueError("companion prompt and reserved answer exceed model context")
    count = base
    companion_count = measure(companion_render([]), reserve_tokens) if companion_render else None
    for record in records:
        candidate = [*selected, record]
        if len(json.dumps(candidate, ensure_ascii=False)) > max_chars:
            rejected.append({**record_source(record), "reason": "character_budget"})
            continue
        usage = measure(render(candidate), reserve_tokens)
        companion = measure(companion_render(candidate), reserve_tokens) if companion_render else None
        if not usage["fits"] or (companion is not None and not companion["fits"]):
            rejected.append({**record_source(record), "reason": "token_budget"})
            continue
        selected, count = candidate, usage
        companion_count = companion
    prompt = render(selected)
    return selected, {
        "schema": "evidence-transport-v1", "presented": [record_source(v) for v in selected],
        "omitted": rejected, "context": count,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "evidence_sha256": hashlib.sha256(json.dumps(
            selected, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
        "presentation_is_not_causal_usage": True,
        "matched_layout_budget": companion_render is not None,
        "shared_remaining_tokens": min(count["remaining_tokens"], companion_count["remaining_tokens"])
            if companion_count is not None and "remaining_tokens" in count and "remaining_tokens" in companion_count else None,
    }
