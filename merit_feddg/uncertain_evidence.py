"""Finite native-observation envelopes, NOT clinical confidence sets.

No label matching, answer rules, score thresholds, or fitted components. The
original observation always participates. Unknown semantic correspondences
are dropped rather than resolved by a guessed diagnosis or object association.
"""
from __future__ import annotations

import copy
import json
from dataclasses import replace

from .structured_evidence import _typed_record

_KEY = "native_uncertainty"
_NUMERIC = {"value", "similarity", "foreground_fraction", "bbox_xyxy_normalized"}


def attach_alternatives(item, alternatives, *, origin="observed_variation"):
    """Adapter hook. Alternatives are same-expert, same-request native payloads.

    The caller must align coordinates before attaching spatial observations.
    This records empirical variation only; it cannot certify domain coverage.
    """
    envelope = {"origin": origin, "alternatives": copy.deepcopy(list(alternatives))}
    _validate(envelope)
    return replace(item, payload={**copy.deepcopy(item.payload), _KEY: envelope})


def _validate(envelope):
    if not isinstance(envelope, dict) or set(envelope) != {"origin", "alternatives"}:
        raise ValueError("invalid native uncertainty envelope")
    if envelope["origin"] not in {"observed_variation", "adapter_hypotheses"}:
        raise ValueError("uncertainty origin is not a calibration certificate")
    alternatives = envelope["alternatives"]
    if not isinstance(alternatives, list) or not alternatives:
        raise ValueError("at least one explicit alternative required")
    if any(not isinstance(p, dict) or _KEY in p for p in alternatives):
        raise ValueError("alternatives must be nonrecursive native payload dictionaries")
    json.dumps(envelope, allow_nan=False)


def _identity(node):
    # All non-numeric semantics must agree. Do not identify two objects merely
    # because they share a class; duplicate identities invalidate correspondence.
    return json.dumps({k: v for k, v in node.items() if k not in _NUMERIC},
                      sort_keys=True, ensure_ascii=False)


def _unique_nodes(nodes):
    groups = {}
    for node in nodes:
        groups.setdefault(_identity(node), []).append(node)
    return {key: values[0] for key, values in groups.items() if len(values) == 1}


def shared_observations(node_sets):
    """Intersect semantics, widen native numeric ranges. Empty means unknown.

    Adding alternatives cannot add semantic nodes or narrow their numeric range.
    Bounds describe these observed values, never disease probability or mask CI.
    """
    if not node_sets:
        return []
    mappings = [_unique_nodes(nodes) for nodes in node_sets]
    output = []
    for key, first in mappings[0].items():
        if not all(key in mapping for mapping in mappings):
            continue
        nodes = [mapping[key] for mapping in mappings]
        node = {k: copy.deepcopy(v) for k, v in first.items() if k not in _NUMERIC}
        ranges = {}
        for field in sorted(_NUMERIC):
            if not all(field in n for n in nodes):
                continue
            values = [n[field] for n in nodes]
            if field == "bbox_xyxy_normalized":
                ranges[field] = {"lower": [min(v[j] for v in values) for j in range(4)],
                                 "upper": [max(v[j] for v in values) for j in range(4)]}
            else:
                ranges[field] = {"lower": min(values), "upper": max(values)}
        if ranges:
            node["observed_value_ranges"] = ranges
        output.append(node)
    return output


def compile_uncertain_evidence(items, question, max_chars):
    """Compile whole packets; never silently truncate an uncertainty envelope.

    Unsupported/no alternatives remain explicitly uncalibrated observations.
    Retrieval stays source-only; generated statements stay unverified. No
    alternate summaries, raw payloads or masks bypass this text-only bridge.
    """
    if type(max_chars) is not int or max_chars < 2:
        raise ValueError("max_chars must be an integer >= 2")
    output = []
    for item in items:
        envelope = item.payload.get(_KEY)
        payloads = [item.payload]
        if envelope is not None:
            _validate(envelope)
            payloads += envelope["alternatives"]
        records = [_typed_record(replace(item, payload=p), question) for p in payloads]
        node_sets = [r["payload"]["observations"] if r else [] for r in records]
        nodes = shared_observations(node_sets)
        if not nodes:
            continue
        record = copy.deepcopy(records[0])
        record["payload"].update({
            "schema": "uncertainty-preserving-observation-v1",
            "observations": nodes,
            "uncertainty": {
                "status": "empirical_only" if envelope else "unknown",
                "origin": envelope["origin"] if envelope else "single_native_output",
                "observation_count": len(payloads),
                "truth_coverage_guaranteed": False,
                "domain_applicability_certified": False,
            },
        })
        # One scalar confidence must not override a set-valued observation.
        record["payload"].pop("native_confidence", None)
        record["payload"].pop("confidence_is_calibrated", None)
        record["payload"]["limitations"] = sorted({
            limitation for r in records if r for limitation in r["payload"]["limitations"]
        }) + [("Shared observations and numeric ranges do not establish clinical truth. "
               "Do not infer a new diagnosis, laterality or relation from missing attributes.")]
        if len(json.dumps(output + [record], ensure_ascii=False)) <= max_chars:
            output.append(record)
    return output
