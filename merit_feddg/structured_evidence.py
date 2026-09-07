"""Typed, claim-scoped presentation of heterogeneous native tool observations.

The graph is deliberately weaker than a diagnosis graph: it preserves what a
tool actually returned, its score semantics and its spatial scope.  It never
turns a similarity, missing detection or predicted anatomy mask into a clinical
truth.  This module is used by the real generation path, not only by tests.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence

from .capabilities import EvidenceItem


def _text(value) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _number(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _box(value) -> list[float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 4:
        return None
    values = [_number(item) for item in value]
    if any(item is None for item in values):
        return None
    x0, y0, x1, y1 = values
    if not 0 <= x0 < x1 <= 1 or not 0 <= y0 < y1 <= 1:
        return None
    return [float(item) for item in values]


def _classification_nodes(payload: Mapping) -> tuple[list[dict], list[str]]:
    nodes: list[dict] = []
    limitations = ["Scores are model observations, not confirmed diagnoses."]
    for key, label_key, score_key in (
        ("catalog", "concept", "similarity"),
        ("findings", "finding", "score"),
    ):
        entries = payload.get(key, ())
        if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
            continue
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            label, score = _text(entry.get(label_key)), _number(entry.get(score_key))
            if label is None or score is None:
                continue
            nodes.append(
                {
                    "kind": "visual_match" if key == "catalog" else "finding_score",
                    "observation": label,
                    "value": score,
                    "polarity": "unknown",
                    "spatial_scope": "whole_image",
                }
            )
    semantics = _text(payload.get("score_semantics"))
    if semantics:
        for node in nodes:
            node["score_semantics"] = semantics
    if payload.get("catalog_exhaustive") is False:
        limitations.append("The configured catalog is non-exhaustive.")
    limitations.append("A low or missing score does not establish absence.")
    return nodes, limitations


def _spatial_nodes(payload: Mapping) -> tuple[list[dict], list[str]]:
    nodes: list[dict] = []
    entries = payload.get("structures", ())
    if isinstance(entries, Sequence) and not isinstance(entries, (str, bytes)):
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            anatomy = _text(entry.get("anatomical_structure"))
            if anatomy is None:
                continue
            node = {
                "kind": "anatomy_region",
                "observation": anatomy,
                "polarity": "predicted_region",
                "spatial_scope": "predicted_mask",
            }
            bbox = _box(
                entry.get("bbox_xyxy_normalized_original_image")
                or entry.get("bbox_xyxy_normalized")
            )
            if bbox is not None:
                node["bbox_xyxy_normalized"] = bbox
            fraction = _number(
                entry.get("foreground_fraction_of_crop")
                or entry.get("foreground_fraction_of_image")
            )
            if fraction is not None and 0 <= fraction <= 1:
                node["foreground_fraction"] = fraction
            nodes.append(node)
    for key in ("detections", "boxes", "objects"):
        entries = payload.get(key, ())
        if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
            continue
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            bbox = _box(
                entry.get("bbox_xyxy_normalized_original_image")
                or entry.get("bbox_xyxy_normalized")
                or entry.get("bbox")
            )
            if bbox is None:
                continue
            label = next(
                (
                    text
                    for name in ("semantic_class", "label", "object", "finding", "concept")
                    if (text := _text(entry.get(name))) is not None
                ),
                "untyped detected region",
            )
            node = {
                "kind": "detected_region",
                "observation": label,
                "polarity": "predicted_region",
                "spatial_scope": "predicted_box",
                "bbox_xyxy_normalized": bbox,
            }
            score = _number(entry.get("score"))
            if score is not None:
                node["value"] = score
                node["score_semantics"] = _text(payload.get("score_semantics")) or "native"
            nodes.append(node)
    direct_box = _box(
        payload.get("foreground_bbox_xyxy_normalized")
        or payload.get("bbox_xyxy_normalized_original_image")
        or payload.get("bbox_xyxy_normalized")
    )
    if payload.get("mask") and not nodes:
        node = {
            "kind": "prompted_foreground",
            "observation": _text(payload.get("semantic_class")) or "untyped foreground",
            "polarity": "predicted_region",
            "spatial_scope": "predicted_mask",
        }
        if direct_box is not None:
            node["bbox_xyxy_normalized"] = direct_box
        fraction = _number(payload.get("foreground_fraction_of_image"))
        if fraction is not None and 0 <= fraction <= 1:
            node["foreground_fraction"] = fraction
        nodes.append(node)
    limitations = [
        "Predicted regions localize anatomy or foreground; they do not establish a diagnosis.",
        "A missing or empty region does not establish clinical absence.",
    ]
    return nodes, limitations


def _retrieval_nodes(payload: Mapping) -> tuple[list[dict], list[str]]:
    nodes: list[dict] = []
    references = payload.get("references", ())
    if isinstance(references, Sequence) and not isinstance(references, (str, bytes)):
        for entry in references:
            if not isinstance(entry, Mapping):
                continue
            source_id = _text(entry.get("source_id"))
            question = _text(entry.get("source_question"))
            if source_id is None and question is None:
                continue
            node = {
                "kind": "source_analogy",
                "source_id": source_id or "unknown",
                "source_question": question or "withheld",
                "applies_to": "source_image_only",
                "polarity": "not_query_evidence",
            }
            similarity = _number(entry.get("similarity"))
            if similarity is not None:
                node["similarity"] = similarity
            reference = _text(entry.get("source_reference"))
            if reference is not None:
                node["source_reference"] = reference
            nodes.append(node)
    return nodes, [
        "A retrieved case belongs to another image and patient.",
        "Similarity is analogy, not evidence that the query image has the same diagnosis.",
    ]


def _generation_nodes(payload: Mapping) -> tuple[list[dict], list[str]]:
    statement = _text(payload.get("generated_text"))
    nodes = [] if statement is None else [{
        "kind": "specialist_statement",
        "observation": statement,
        "polarity": "unverified",
        "spatial_scope": "unspecified",
    }]
    return nodes, [
        "This is a fallible specialist-model statement, not a reference report or ground truth."
    ]


def _typed_record(item: EvidenceItem, question: str) -> dict | None:
    payload = item.payload
    if item.capability == "classification":
        observations, limitations = _classification_nodes(payload)
    elif item.capability in {"segmentation", "detection"}:
        observations, limitations = _spatial_nodes(payload)
    elif item.capability == "retrieval":
        observations, limitations = _retrieval_nodes(payload)
    elif item.capability == "generation":
        observations, limitations = _generation_nodes(payload)
    else:
        return None
    if not observations:
        return None
    record = {
        "expert_id": item.expert_id,
        "evidence_id": item.evidence_id,
        "capability": item.capability,
        "scope": item.scope,
        "payload": {
            "schema": "typed-clinical-evidence-v1",
            "claim_query": question,
            "observations": observations,
            "limitations": limitations,
            "provenance": "model_generated",
        },
    }
    confidence = _number(item.confidence)
    if confidence is not None:
        record["payload"]["native_confidence"] = confidence
        record["payload"]["confidence_is_calibrated"] = False
    return record


def compile_typed_evidence(items, question: str, max_chars: int) -> list[dict]:
    """Compile whole typed nodes within a JSON budget; never truncate a statement."""

    if type(max_chars) is not int or max_chars < 2:
        raise ValueError("max_chars must be an integer of at least two")
    records = [record for item in items if (record := _typed_record(item, question))]
    output: list[dict] = []
    for record in records:
        observations = record["payload"]["observations"]
        while observations and len(json.dumps(output + [record], ensure_ascii=False)) > max_chars:
            observations.pop()
        if observations and len(json.dumps(output + [record], ensure_ascii=False)) <= max_chars:
            output.append(record)
    return output
