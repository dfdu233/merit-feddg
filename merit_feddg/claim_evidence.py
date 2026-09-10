"""Training-free native-entry transport; no fitted medical truth model.

Structured lists are split at adapter-defined boundaries. Free text is retained
as one observation, not misrepresented as extracted atomic medical claims.
"""
import copy
import re
from dataclasses import replace

LIST_FIELDS = ("findings", "catalog", "structures", "detections", "boxes", "objects")
PLACEHOLDERS = {"cxr_description", "generated_text", "null"}


def native_entries(items):
    result = []
    for item in items:
        fields = [k for k in LIST_FIELDS if isinstance(item.payload.get(k), list)]
        if len(fields) != 1:
            result.append(copy.deepcopy(item))
            continue
        field = fields[0]
        for index, entry in enumerate(item.payload[field]):
            payload = copy.deepcopy(item.payload)
            payload[field] = [copy.deepcopy(entry)]
            result.append(replace(item, evidence_id=f"{item.evidence_id}/native/{field}/{index}",
                payload=payload, summary="", confidence=None, provenance={**copy.deepcopy(item.provenance),
                "native_entry": {"parent_id": item.evidence_id, "field": field, "index": index,
                                 "parent_summary_in_audit": True, "parent_confidence": item.confidence,
                                 "confidence_scope": "parent_packet_not_entry"}}))
    return tuple(result)


def attribute_check(item, question):
    """Conservative interface check, NOT relevance/correctness classification.

    Only reject demonstrable schema/short-answer mismatches. Unknown language or
    ambiguous content passes with unknown status. No answer-type metadata used.
    """
    q = question.lower()
    spatial_question = bool(re.search(r"\b(where|which side|what side|left or right|laterality)\b", q))
    text = item.payload.get("generated_text")
    audit = {"passed": True, "status": "unknown", "reason": "no_proven_interface_mismatch",
             "correctness_established": False, "question_attribute": "spatial" if spatial_question else "unknown"}
    if text is not None:
        normalized = text.strip().lower().strip(" .!?") if isinstance(text, str) else ""
        if not normalized or normalized in PLACEHOLDERS:
            audit.update(passed=False, status="invalid", reason="empty_or_schema_placeholder")
        elif spatial_question and normalized in {"yes", "no", "positive", "negative"}:
            audit.update(passed=False, status="mismatch", reason="bare_polarity_cannot_supply_spatial_attribute")
        else:
            audit.update(status="unchecked_text", reason="retained_fallible_observation")
    return audit
