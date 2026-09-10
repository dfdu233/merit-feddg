"""Whole-record packing with actual full-prompt token accounting.

Packing is deterministic in caller-provided order. It is not a learned utility
policy. A presented record is an input intervention, not proof of model usage.
"""

import hashlib
import json


def record_source(record):
    return {key: record[key] for key in ("expert_id", "evidence_id")}


def pack_records(records, render, measure, *, max_chars, reserve_tokens):
    selected, rejected = [], []
    base = measure(render([]), reserve_tokens)
    if not base["fits"]:
        raise ValueError("original prompt and reserved answer exceed model context")
    count = base
    for record in records:
        candidate = [*selected, record]
        if len(json.dumps(candidate, ensure_ascii=False)) > max_chars:
            rejected.append({**record_source(record), "reason": "character_budget"})
            continue
        usage = measure(render(candidate), reserve_tokens)
        if not usage["fits"]:
            rejected.append({**record_source(record), "reason": "token_budget"})
            continue
        selected, count = candidate, usage
    prompt = render(selected)
    return selected, {
        "schema": "evidence-transport-v1", "presented": [record_source(v) for v in selected],
        "omitted": rejected, "context": count,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "evidence_sha256": hashlib.sha256(json.dumps(
            selected, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
        "presentation_is_not_causal_usage": True,
    }
