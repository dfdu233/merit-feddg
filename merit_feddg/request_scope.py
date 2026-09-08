"""Question-only task coverage, independent of domain-risk estimation.

Explicit finite catalogs must not silently answer unrestricted clinical questions.
Lexical aliases declare a request, not a diagnosis, a truth label or a negative finding.
"""
from __future__ import annotations

import copy
import re
from dataclasses import replace


def normalized(text):
    return " ".join(re.findall(r"[a-z0-9]+", str(text).lower()))


def contains_phrase(question, phrase):
    phrase = normalized(phrase)
    return bool(phrase) and f" {phrase} " in f" {normalized(question)} "


def assess_request(question, spec, capability):
    contract = spec.get("request_contract")
    result = {"allowed": False, "matched_concepts": [], "kind": "task_coverage_not_domain_risk"}
    if not contract:
        return {**result, "reason": "missing_request_contract"}
    mode = contract.get("mode")
    if mode == "free_query":
        if capability not in {"generation", "retrieval"}:
            raise ValueError("free_query requires a generative or retrieval capability")
        return {**result, "allowed": True, "reason": "query_conditioned_tool"}
    if mode != "named_concepts":
        raise ValueError("unknown request contract mode")
    aliases = contract.get("concept_aliases", {})
    if not isinstance(aliases, dict) or not aliases:
        raise ValueError("named_concepts needs explicit concept_aliases")
    matched = []
    for concept, names in aliases.items():
        if (not isinstance(concept, str) or not concept.strip()
                or not isinstance(names, list) or any(not isinstance(n, str) or not n.strip() for n in names)):
            raise ValueError("invalid request concept aliases")
        if any(contains_phrase(question, name) for name in [concept, *names]):
            matched.append(concept)
    return {**result, "allowed": bool(matched), "matched_concepts": matched,
            "reason": "explicit_supported_concept" if matched else "outside_explicit_request_coverage"}


def bind_request(items, audit):
    return tuple(replace(item, provenance={**item.provenance,
                 "request_scope": copy.deepcopy(audit)}) for item in items)


def focused_items(items):
    """No top-score fallback. Unsupported native labels cannot take over the answer."""
    output = []
    fields = {"catalog": "concept", "findings": "finding", "structures": "anatomical_structure",
              "detections": "label", "boxes": "label", "objects": "label"}
    for item in items:
        audit = item.provenance.get("request_scope", {})
        if not audit.get("allowed"):
            continue
        if item.capability in {"generation", "retrieval"}:
            output.append(item)
            continue
        names = {normalized(name) for name in audit.get("matched_concepts", [])}
        payload = copy.deepcopy(item.payload)
        usable = False
        for field, label in fields.items():
            if isinstance(payload.get(field), list):
                payload[field] = [entry for entry in payload[field] if isinstance(entry, dict)
                                  and normalized(entry.get(label, "")) in names]
                usable |= bool(payload[field])
        # Single-mask plugins use explicit semantic_class. No implicit whole-image ROI.
        if "mask" in payload and normalized(payload.get("semantic_class", "")) in names:
            usable = True
        if usable:
            output.append(replace(item, payload=payload))
    return tuple(output)
