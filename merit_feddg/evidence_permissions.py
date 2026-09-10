"""Finite, object-scoped predicates supported by every supplied observation.

These permissions describe *model measurements*, not clinical truth. No score
threshold becomes a diagnosis. No organ mask becomes a lesion. Predicates use
same-expert, same-request objects only; unrelated experts are never joined.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import replace

from .structured_evidence import _typed_record
from .uncertain_evidence import (
    _KEY,
    _classification_table,
    _factor_shared_fields,
    _unique_nodes,
    _validate,
    shared_observations,
)


def _coordinate_frame(payload, label):
    # A generic bbox is not enough to establish its frame. In particular screen
    # x ordering is not patient left/right and no relation is inferred between
    # a whole-image class and a spatial object.
    for name in ("structures", "detections", "boxes", "objects"):
        matches = [v for v in payload.get(name, []) if isinstance(v, dict) and label in (
            v.get("anatomical_structure"), v.get("semantic_class"), v.get("label"),
            v.get("finding"), v.get("concept"), v.get("object"))]
        if len(matches) == 1 and "bbox_xyxy_normalized_original_image" in matches[0]:
            return "original_image_xy"
    return None


def supported_relations(node_sets, payloads):
    """Intersection of strict score orders and separated image-x intervals.

    Numeric interval comparisons are over the Cartesian envelope (conservative
    when measured values are correlated). Enlarging an envelope cannot add an
    order. Duplicate object identities invalidate correspondence.
    """
    shared = shared_observations(node_sets)
    relations = []
    for left in shared:
        for right in shared:
            if left is right or left.get("observation") == right.get("observation"):
                continue
            a, b = left.get("observed_value_ranges", {}), right.get("observed_value_ranges", {})
            if (left.get("kind") == right.get("kind")
                    and left.get("kind") in {"finding_score", "visual_match"}
                    and left.get("score_semantics") == right.get("score_semantics")
                    and left.get("score_semantics") is not None
                    and "value" in a and "value" in b and a["value"]["lower"] > b["value"]["upper"]):
                relations.append({"predicate": "native_score_greater", "subject": left["observation"],
                                  "object": right["observation"], "clinical_ordering": False})
            if (left.get("kind") == right.get("kind") == "anatomy_region"
                    and "bbox_xyxy_normalized" in a and "bbox_xyxy_normalized" in b
                    and all(_coordinate_frame(p, node["observation"]) == "original_image_xy"
                            for p in payloads for node in (left, right))
                    and a["bbox_xyxy_normalized"]["upper"][2] < b["bbox_xyxy_normalized"]["lower"][0]):
                relations.append({"predicate": "image_x_before", "subject": left["observation"],
                                  "object": right["observation"], "patient_laterality_inferred": False})
    return relations


def _widen_nodes(nodes, radius):
    """Apply one simultaneous scalar error radius to every scalar score.

    Spatial uncertainty needs a spatial loss/card and is deliberately unsupported
    by this scalar-score calibration path.
    """
    endpoints = []
    for sign in (-1, 1):
        bound = copy.deepcopy(nodes)
        for node in bound:
            if node.get("kind") in {"finding_score", "visual_match"} and "value" in node:
                node["value"] += sign * radius
        endpoints.append(bound)
    return endpoints


def permission_record(item, question):
    envelope = item.payload.get(_KEY)
    payloads = [item.payload]
    if envelope is not None:
        _validate(envelope)
        payloads.extend(envelope["alternatives"])
    records = [_typed_record(replace(item, payload=p), question) for p in payloads]
    node_sets = [r["payload"]["observations"] if r else [] for r in records]
    if not all(node_sets):
        return None
    precision = item.provenance.get("native_precision", {})
    radius = precision.get("radius")
    if radius is not None:
        if (type(radius) not in (int, float) or not math.isfinite(radius) or radius < 0
                or precision.get("loss") != "native_simultaneous_score_error"):
            raise ValueError("invalid native precision radius/units")
        original_sets = node_sets
        node_sets = [bound for nodes in original_sets for bound in _widen_nodes(nodes, radius)]
    if precision.get("status") == "unsupported_domain":
        # Domain support cannot be restored by photometric stability.
        return None
    nodes = shared_observations(node_sets)
    if not nodes:
        return None
    # No pairwise diagnostic ranking or geometry claim from a single unverified
    # observation. Its native measurements may still be presented as observations.
    relational = envelope is not None or radius is not None
    relations = supported_relations(node_sets, payloads) if relational else []
    source_record = records[0]
    shared, compact = _factor_shared_fields(nodes)
    table_key, table = _classification_table(shared, compact)
    source_record["payload"] = {
        "schema": "native-permissions-v1", "claim_query": question,
        table_key or "observations": table if table_key else compact,
        "shared_observation_fields": shared, "observation_count": len(nodes),
        "supported_measurement_relations": relations[:8],
        "supported_relation_count_before_budget": len(relations),
        "basis": "declared_native_score_radius" if radius is not None else (
            "observed_variation_only" if envelope else "single_unverified_observation"),
        "native_precision": precision or {"status": "not_calibrated"},
        "truth_coverage_guaranteed": False, "free_text_compliance_guaranteed": False,
        "unsupported_attributes": ["unmeasured_diagnosis", "unmeasured_lesion_location",
                                   "patient_laterality_from_image_x", "absence_from_missing_output"],
    }
    return source_record


def compile_permission_evidence(items, question, max_chars):
    output = []
    for item in items:
        record = permission_record(item, question)
        if record and len(json.dumps([*output, record], ensure_ascii=False)) <= max_chars:
            output.append(record)
    return output


def permission_audit(item, question):
    """Mechanism audit, separate from answer correctness or clinical adjudication."""
    point = replace(item, payload={k: v for k, v in item.payload.items() if k != _KEY})
    point_record = _typed_record(point, question)
    observed = permission_record(item, question)
    return {"expert_id": item.expert_id, "evidence_id": item.evidence_id,
            "point_unique_nodes": len(_unique_nodes(point_record["payload"]["observations"]))
            if point_record else 0,
            "shared_nodes": observed["payload"]["observation_count"] if observed else 0,
            "relations": observed["payload"]["supported_measurement_relations"] if observed else [],
            "not_a_medical_accuracy_metric": True}
