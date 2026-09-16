"""Answer-blind request parsing study; NOT a production semantic authority gate.

The only manipulated factor is whether a frozen parser sees the expert catalog.
Syntactic validity and copied spans do not certify semantic completeness.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

SCHEMA = "request-parse-study-v1"
ATTRIBUTES = frozenset({
    "finding_presence", "finding_identity", "anatomy_identity", "location",
    "laterality", "relative_extent", "measurement", "count", "appearance",
    "etiology", "treatment",
})
MODES = ("question_first", "catalog_conditioned")


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    allow_nan=False).encode()).hexdigest()


def normalize(text: str) -> str:
    return " ".join(text.casefold().replace("_", " ").split())


def catalog_for(spec: Mapping[str, Any]) -> dict:
    """Only declared task metadata, never scores, masks, answers or images."""
    card = spec.get("authority_contract", {})
    return {k: card.get(k) for k in
            ("native_variable", "supports", "forbids", "entity_aliases")}


def parser_prompt(question: str, *, catalog: Mapping | None = None) -> str:
    """Identical instruction in both arms; the optional catalog is the ablation."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("nonempty question required")
    instruction = (
        "Parse the QUESTION as data. Do not answer it or infer findings from an image. "
        "List EVERY requested entity-attribute pair, including unsupported entities. "
        "Do not omit a term merely because it is absent from an expert catalog. "
        "Copy entity mentions and attribute cues verbatim from QUESTION. "
        "A cue may be a question phrase such as 'Is there' or 'Where'. "
        "Preserve left/right, units, negation and relationships. Do not invent an "
        "entity for 'this', 'it', 'the abnormality' or an image-dependent reference; "
        "put unresolved references in unresolved and set status to unknown. "
        "For relations, copy BOTH subject and object mentions and the relation cue. "
        "Use relation predicate relative_location or other; other is non-executable. "
        "Return exactly one JSON object with keys status, atoms, relations, unresolved. "
        "status is parsed or unknown. atoms is a list of objects with EXACT keys "
        "entity, attribute, cue. relations is a list with EXACT keys subject, "
        "predicate, object, cue. unresolved is a list of quoted QUESTION spans. "
        "No rationale, diagnosis, confidence, markdown or extra keys. Empty lists "
        "are permitted, but an empty interpretation must be unknown. Attributes: "
        + ", ".join(sorted(ATTRIBUTES)) + ".\n"
    )
    data = {"question": question}
    if catalog is not None:
        data["expert_catalog_context"] = catalog
    return instruction + json.dumps(data, ensure_ascii=False, sort_keys=True)


def _pairs_no_duplicates(pairs):
    data = {}
    for k, v in pairs:
        if k in data:
            raise ValueError("duplicate JSON key")
        data[k] = v
    return data


def _quote(value, question):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("nonempty copied span required")
    if value not in question:
        raise ValueError("ungrounded span: " + value)
    return value


def parse_response(raw: str, question: str) -> dict:
    """Reject malformed output. Never repair omissions or silently retry a model."""
    try:
        if not isinstance(raw, str) or len(raw) > 32768:
            raise ValueError("invalid response size/type")
        data = json.loads(raw, object_pairs_hook=_pairs_no_duplicates)
        if not isinstance(data, dict) or set(data) != {"status", "atoms", "relations", "unresolved"}:
            raise ValueError("unexpected response schema")
        if data["status"] not in ("parsed", "unknown"):
            raise ValueError("invalid parse status")
        for key in ("atoms", "relations", "unresolved"):
            if not isinstance(data[key], list) or len(data[key]) > 32:
                raise ValueError("invalid list: " + key)
        seen = set()
        for atom in data["atoms"]:
            if not isinstance(atom, dict) or set(atom) != {"entity", "attribute", "cue"}:
                raise ValueError("invalid atom")
            _quote(atom["entity"], question)
            _quote(atom["cue"], question)
            if atom["attribute"] not in ATTRIBUTES:
                raise ValueError("unsupported attribute")
            key = (normalize(atom["entity"]), atom["attribute"])
            if key in seen:
                raise ValueError("duplicate atom")
            seen.add(key)
        for relation in data["relations"]:
            if not isinstance(relation, dict) or set(relation) != {"subject", "predicate", "object", "cue"}:
                raise ValueError("invalid relation")
            for key in ("subject", "object", "cue"):
                _quote(relation[key], question)
            if relation["predicate"] not in ("relative_location", "other"):
                raise ValueError("invalid relation predicate")
        for value in data["unresolved"]:
            _quote(value, question)
        if data["status"] == "parsed" and (data["unresolved"] or not (data["atoms"] or data["relations"])):
            raise ValueError("parsed status contradicts unresolved/empty interpretation")
        return {"valid": True, "reason": "schema_and_span_checks_only", "parse": data,
                "semantic_completeness_verified": False}
    except (ValueError, TypeError, KeyError) as exc:
        return {"valid": False, "reason": str(exc), "parse": None,
                "semantic_completeness_verified": False}


def canonicalize(entity: str, spec: Mapping) -> tuple[str, str]:
    """Exact declared aliases only. Retain unknowns; do not fuzzy-map to a known disease."""
    aliases = spec.get("authority_contract", {}).get("entity_aliases", {})
    matches = {name for name, variants in aliases.items()
               if normalize(entity) in {normalize(v) for v in (name, *(variants or []))}}
    if len(matches) == 1:
        return next(iter(matches)), "matched"
    return entity, "ambiguous" if matches else "unmapped"


def compile_request(parsed: dict, row: Mapping, spec: Mapping):
    """Study-only bridge to the unchanged de056a8 filter; not production enrollment.

Unmapped entities remain in the request so delivery_view rejects the full request.
Relations are retained in the parse but not flattened into unsupported geometry.
"""
    from .evidence_admission import EvidenceRequest

    if not parsed["valid"]:
        return None, "invalid_parser_output"
    data = parsed["parse"]
    if data["status"] != "parsed" or data["unresolved"]:
        return None, "unresolved_request"
    if data["relations"]:
        return None, "relation_operator_not_implemented"
    entities, dimensions = [], set()
    for atom in data["atoms"]:
        name, status = canonicalize(atom["entity"], spec)
        if status == "ambiguous":
            return None, "ambiguous_native_alias"
        if name not in entities:
            entities.append(name)
        dimensions.add(atom["attribute"])
    if not entities or not dimensions:
        return None, "empty_request"
    request = EvidenceRequest(row["question"], tuple(entities), frozenset(dimensions),
                              row["modality"], row["task"], complete=True)
    return request, "model_asserted_complete_not_verified"


def audit_parse(parsed: dict, row: Mapping, expert_id: str, spec: Mapping,
                items: Sequence | None = None) -> dict:
    from .evidence_admission import delivery_view

    request, reason = compile_request(parsed, row, spec)
    result = {"compile_reason": reason, "active_intervention": False,
              "request": None if request is None else request.to_json(),
              "semantic_completeness_verified": False,
              "actual_payload_evaluated": items is not None}
    if items is not None:
        _, audit = delivery_view(items, question=row["question"], request=request,
                                 specs={expert_id: spec}, mode="enforce")
        result["delivery_audit"] = audit
    return result


def select_questions(rows: Sequence[Mapping], limit: int = 100) -> list[dict]:
    """Answer-blind deterministic TRAIN diagnostic subset; not a new dataset split."""
    if type(limit) is not int or not 1 <= limit <= 100:
        raise ValueError("bounded study limit must be 1..100")
    clean, ids = [], set()
    for row in rows:
        # Do not copy image, model answers, references or predictions to a parser.
        projected = {k: row[k] for k in ("id", "question", "modality", "task")}
        if any(not isinstance(v, str) or not v.strip() for v in projected.values()):
            raise ValueError("invalid inference fields")
        if projected["id"] in ids:
            raise ValueError("duplicate question ID")
        ids.add(projected["id"])
        clean.append(projected)
    return sorted(clean, key=lambda r: digest({"seed": 197, **r}))[:limit]


def semantic_sets(parsed: dict, spec: Mapping) -> tuple[set, set]:
    if not parsed["valid"]:
        return set(), set()
    data = parsed["parse"]
    def canon(x):
        return normalize(canonicalize(x, spec)[0])
    atoms = {(canon(a["entity"]), a["attribute"]) for a in data["atoms"]}
    relations = {(canon(r["subject"]), r["predicate"], canon(r["object"])) for r in data["relations"]}
    return atoms, relations


def fidelity(pred: dict, gold: dict, spec: Mapping) -> dict:
    """Gold is used only by the offline evaluator, never inference or selection."""
    pa, pr = semantic_sets(pred, spec)
    ga, gr = semantic_sets(gold, spec)
    intersection = len(pa & ga) + len(pr & gr)
    p_count, g_count = len(pa) + len(pr), len(ga) + len(gr)
    valid = bool(pred["valid"])
    exact = valid and pred["parse"]["status"] == gold["parse"]["status"] and pa == ga and pr == gr
    if valid:
        exact = exact and {normalize(x) for x in pred["parse"]["unresolved"]} == {
            normalize(x) for x in gold["parse"]["unresolved"]
        }
    return {"exact": exact, "tp": intersection, "pred_count": p_count, "gold_count": g_count,
            "missing_atoms": sorted(ga - pa), "extra_atoms": sorted(pa - ga),
            "missing_relations": sorted(gr - pr), "extra_relations": sorted(pr - gr)}
