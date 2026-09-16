"""Machine-readable semantic authority contracts for frozen specialists.

This module separates capability, applicability and reliability. It handles only
what an expert is natively authorized to change; it deliberately does not
estimate whether one prediction is correct.
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

SEMANTIC_DIMENSIONS = frozenset({
    "finding_presence", "finding_identity", "anatomy_identity", "location",
    "laterality", "relative_extent", "measurement", "count", "appearance",
    "etiology", "treatment",
})


@dataclass(frozen=True)
class NativeVariable:
    entity_type: str
    attribute: str
    values: tuple[str, ...]
    output_semantics: str

    def __post_init__(self):
        if any(not isinstance(v, str) or not v.strip() for v in
               (self.entity_type, self.attribute, self.output_semantics)):
            raise ValueError("native variable fields must be nonempty strings")
        if not self.values or any(not isinstance(v, str) or not v.strip() for v in self.values):
            raise ValueError("native variable values must be nonempty strings")


@dataclass(frozen=True)
class CapabilityAuthorityContract:
    expert_id: str
    capability: str
    scope: str
    native_variable: NativeVariable
    supports: frozenset[str]
    forbids: frozenset[str]
    entity_aliases: tuple[tuple[str, tuple[str, ...]], ...] = ()
    requires_entity_match_for: frozenset[str] = frozenset()

    def __post_init__(self):
        if not self.expert_id or not self.capability or not self.scope:
            raise ValueError("authority identity, capability and scope must be nonempty")
        unknown = (self.supports | self.forbids | self.requires_entity_match_for) - SEMANTIC_DIMENSIONS
        if unknown:
            raise ValueError(f"unknown semantic dimensions: {sorted(unknown)}")
        overlap = self.supports & self.forbids
        if overlap:
            raise ValueError(f"supported and forbidden dimensions overlap: {sorted(overlap)}")
        if not self.supports:
            raise ValueError("authority contract must support at least one dimension")
        if not self.requires_entity_match_for <= self.supports:
            raise ValueError("entity-match requirements must be a subset of supported dimensions")
        names = [name for name, _ in self.entity_aliases]
        if len(names) != len(set(names)):
            raise ValueError("authority entity names must be unique")
        for name, aliases in self.entity_aliases:
            if not name.strip() or any(not alias.strip() for alias in aliases):
                raise ValueError("authority entity names and aliases must be nonempty")
        if self.requires_entity_match_for and not self.entity_aliases:
            raise ValueError("entity-match requirements need an explicit native entity catalog")

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["schema"] = "native-authority-v1"
        data["supports"] = sorted(self.supports)
        data["forbids"] = sorted(self.forbids)
        data["requires_entity_match_for"] = sorted(self.requires_entity_match_for)
        data["entity_aliases"] = {name: list(aliases) for name, aliases in self.entity_aliases}
        data["native_variable"]["values"] = list(self.native_variable.values)
        return data


def _strings(values: Any, name: str) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f"{name} must be a nonempty list")
    result = tuple(str(v).strip() for v in values)
    if any(not v for v in result):
        raise ValueError(f"{name} contains an empty value")
    return result


def _entity_aliases(raw: Any) -> tuple[tuple[str, tuple[str, ...]], ...]:
    if raw is None:
        return ()
    if not isinstance(raw, Mapping):
        raise TypeError("authority_contract.entity_aliases must be a mapping")
    result = []
    for entity, aliases in raw.items():
        name = str(entity).strip()
        if not name:
            raise ValueError("authority entity name must be nonempty")
        if aliases is None:
            values = ()
        elif isinstance(aliases, (list, tuple)):
            values = tuple(str(alias).strip() for alias in aliases)
        else:
            raise TypeError("authority entity aliases must be lists")
        if any(not alias for alias in values):
            raise ValueError("authority entity alias must be nonempty")
        result.append((name, values))
    return tuple(result)


def authority_contract(expert_id: str, spec: Mapping[str, Any], capability: str):
    """Parse an explicit contract. Missing contracts remain legacy/undeclared."""
    raw = spec.get("authority_contract")
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise TypeError("authority_contract must be a mapping")
    if raw.get("schema", "native-authority-v1") != "native-authority-v1":
        raise ValueError("unsupported authority contract schema")
    declared = str(raw.get("capability", capability)).strip()
    if declared != capability:
        raise ValueError("authority contract capability mismatch")
    scope = str(raw.get("scope", spec.get("scope", capability))).strip()
    if scope != str(spec.get("scope", capability)).strip():
        raise ValueError("authority contract scope mismatch")
    native = raw.get("native_variable")
    if not isinstance(native, Mapping):
        raise TypeError("authority contract requires native_variable")
    variable = NativeVariable(
        entity_type=str(native.get("entity_type", "")).strip(),
        attribute=str(native.get("attribute", "")).strip(),
        values=_strings(native.get("values"), "native_variable.values"),
        output_semantics=str(native.get("output_semantics", "")).strip(),
    )
    supports = frozenset(_strings(raw.get("supports"), "authority_contract.supports"))
    forbids = frozenset(str(v).strip() for v in raw.get("forbids", ()))
    if any(not v for v in forbids):
        raise ValueError("authority_contract.forbids contains an empty value")
    requires = frozenset(str(v).strip() for v in raw.get("requires_entity_match_for", ()))
    if any(not v for v in requires):
        raise ValueError("requires_entity_match_for contains an empty value")
    return CapabilityAuthorityContract(
        expert_id, capability, scope, variable, supports, forbids,
        _entity_aliases(raw.get("entity_aliases")), requires,
    )


def _normalized(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(text).lower()))


def _contains_phrase(text: str, phrase: str) -> bool:
    needle = _normalized(phrase)
    return bool(needle) and f" {needle} " in f" {_normalized(text)} "


def matched_entities(question: str, contract: CapabilityAuthorityContract) -> tuple[str, ...]:
    """Match only entities explicitly declared by the native model contract."""
    matched = []
    for entity, aliases in contract.entity_aliases:
        if any(_contains_phrase(question, name) for name in (entity, *aliases)):
            matched.append(entity)
    return tuple(matched)


def question_semantics(question: str) -> frozenset[str]:
    """Transparent English pilot rules; empty means authority is not established."""
    text = _normalized(question)
    if not text:
        return frozenset()
    requested: set[str] = set()
    if re.search(r"\b(which side|left|right|bilateral|laterality|side)\b", text):
        requested.add("laterality")
    if re.search(r"\b(where|location|located|locali[sz](?:e|ed|ation)|region)\b", text):
        requested.add("location")
    if re.search(r"\b(how many|number of|count)\b", text):
        requested.add("count")
    if re.search(r"\b(diameter|volume|measure|measurement|how large|how big|centimeter|millimeter|cm|mm)\b", text):
        requested.add("measurement")
    if re.search(r"\b(relative area|proportion|fraction|percent|percentage|extent)\b", text):
        requested.add("relative_extent")
    if re.search(r"\b(color|colour|appearance|morphology|shape|pattern|texture)\b", text):
        requested.add("appearance")
    if re.search(r"\b(cause|caused by|etiology|aetiology|why did)\b", text):
        requested.add("etiology")
    if re.search(r"\b(treatment|therapy|manage|management|medication|surgery)\b", text):
        requested.add("treatment")
    if re.search(r"\b(what|which) (organ|body part|anatomical structure|structure|system)\b", text):
        requested.add("anatomy_identity")
    if re.search(r"\b(what|which) (diagnosis|disease|disorder|finding|abnormality|pathology)\b", text):
        requested.add("finding_identity")
    presence = (
        re.search(r"^(is|are) there\b", text)
        or re.search(r"^(do|does|can) (you |this |the )?(see|show|have|demonstrate|reveal)\b", text)
        or re.search(r"\b(evidence of|present|seen|visualized|demonstrated)\b", text)
    )
    if presence:
        requested.add("finding_presence")
    return frozenset(requested)


def assess_authority(question: str, spec: Mapping[str, Any], capability: str,
                     *, expert_id: str = "") -> dict[str, Any]:
    """Audit semantic authority without estimating correctness or reliability."""
    contract = authority_contract(expert_id or str(spec.get("id", "expert")), spec, capability)
    requested = question_semantics(question)
    if contract is None:
        return {
            "schema": "native-authority-audit-v1", "declared": False,
            "status": "undeclared", "requested_dimensions": sorted(requested),
            "authorized_dimensions": [], "unsupported_dimensions": [],
            "matched_entities": [], "entity_unmatched_dimensions": [],
            "global_transport_allowed": True, "local_intervention_allowed": False,
            "reason": "legacy_expert_without_authority_contract",
            "reliability_estimated": False,
        }
    entities = matched_entities(question, contract)
    if not requested:
        status, authorized, unsupported = "unknown", frozenset(), frozenset()
        entity_unmatched = frozenset()
        reason = "question_semantics_not_established"
    else:
        authorized = requested & contract.supports
        unsupported = requested - contract.supports
        entity_unmatched = authorized & contract.requires_entity_match_for if not entities else frozenset()
        authorized = authorized - entity_unmatched
        unsupported = unsupported | entity_unmatched
        if not authorized:
            status = "denied"
            reason = "native_entity_not_declared" if entity_unmatched else "no_requested_dimension_is_authorized"
        elif unsupported:
            status, reason = "partial", "only_subset_of_requested_dimensions_authorized"
        else:
            status, reason = "exact", "all_requested_dimensions_authorized"
    return {
        "schema": "native-authority-audit-v1", "declared": True, "status": status,
        "requested_dimensions": sorted(requested),
        "authorized_dimensions": sorted(authorized),
        "unsupported_dimensions": sorted(unsupported),
        "matched_entities": list(entities),
        "entity_unmatched_dimensions": sorted(entity_unmatched),
        # Legacy text transport can perturb the whole answer; only exact matches
        # may use it. Partial matches are reserved for a future local intervention.
        "global_transport_allowed": status == "exact",
        "local_intervention_allowed": bool(authorized), "reason": reason,
        "contract": contract.to_json(), "reliability_estimated": False,
    }
