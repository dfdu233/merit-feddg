"""Capability-bounded projection over a finite pool of free-text answers.

This module does not generate text, infer medical truth, or calibrate a specialist.
It only projects a finite free-text candidate distribution so that one declared
specialist-native binary attribute can change while all other candidate
preferences are minimally disturbed.

The projection is a finite-pool I-projection. Unknown/unmapped candidates keep
exactly their original total probability mass. Within the positive and negative
groups, relative candidate odds are preserved exactly. Only the group marginal
changes.

``base_scores`` may be any finite candidate score vector. In the MERIT pilot the
scores are length-normalized sequence log-likelihoods from the frozen generalist,
so the normalized pool distribution is a candidate preference distribution, NOT
the model's exact probability over all possible text.

The XRV helper reproduces TorchXRayVision's published operating-point
normalization. The resulting coordinate is NOT a calibrated clinical probability.
It is used only as the source model's own decision coordinate in this pilot.
"""
from __future__ import annotations

import math
import re
from typing import Any, Sequence

import numpy as np

POSITIVE = "positive"
NEGATIVE = "negative"
UNKNOWN = "unknown"
VALID_GROUPS = {POSITIVE, NEGATIVE, UNKNOWN}


def _unit_interval(value: float, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be numeric, not boolean")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite in [0, 1]")
    return result


def normalized_pool_distribution(scores: Sequence[float]) -> np.ndarray:
    """Stable softmax over a finite candidate pool."""
    z = np.asarray(scores, dtype=np.float64)
    if z.ndim != 1 or len(z) < 1 or not np.isfinite(z).all():
        raise ValueError("scores must be a nonempty finite vector")
    shifted = z - z.max()
    weights = np.exp(shifted)
    total = float(weights.sum())
    if not math.isfinite(total) or total <= 0:
        raise FloatingPointError("invalid candidate normalization")
    return weights / total


def xrv_operating_coordinate(raw_score: float, operating_point: float) -> float:
    """TorchXRayVision operating-point normalization for one native label.

    The published operating point maps to 0.5, raw 0 maps to 0 and raw 1 maps
    to 1. This is a source-native decision coordinate, not a calibrated
    correctness probability and not a target-domain decision threshold.
    """
    p = _unit_interval(raw_score, "raw_score")
    threshold = float(operating_point)
    if not math.isfinite(threshold) or not 0.0 < threshold < 1.0:
        raise ValueError("operating_point must be finite and strictly inside (0, 1)")
    if p < threshold:
        return p / (2.0 * threshold)
    return 1.0 - ((1.0 - p) / (2.0 * (1.0 - threshold)))


def _validate_groups(groups: Sequence[str], n: int) -> np.ndarray:
    if len(groups) != n:
        raise ValueError("groups must align one-to-one with candidates")
    values = np.asarray(list(groups), dtype=object)
    if any(group not in VALID_GROUPS for group in values):
        raise ValueError(f"groups must be one of {sorted(VALID_GROUPS)}")
    return values


def mapped_positive_mass(probabilities: Sequence[float], groups: Sequence[str]) -> float | None:
    """Positive mass conditional on the mapped positive/negative subset."""
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 1 or not np.isfinite(p).all() or (p < 0).any():
        raise ValueError("probabilities must be a finite nonnegative vector")
    g = _validate_groups(groups, len(p))
    pos = float(p[g == POSITIVE].sum())
    neg = float(p[g == NEGATIVE].sum())
    mapped = pos + neg
    return None if mapped <= np.finfo(np.float64).eps else pos / mapped


def project_binary_authority(
    base_scores: Sequence[float],
    groups: Sequence[str],
    source_positive_mass: float,
) -> dict[str, Any]:
    """Minimum-change projection on the specialist-authorized binary attribute.

    Let ``p`` be the normalized finite-pool base distribution, with candidates
    partitioned into positive, negative and unknown groups. The feasible set fixes:

    * total unknown mass exactly to its base value;
    * positive mass inside the mapped subset to ``source_positive_mass``;
    * negative mass to the complement.

    The unique KL-minimizing solution preserves all within-group relative odds.
    If either positive or negative text is absent, the requested binary marginal
    is not identifiable from this finite pool; the function explicitly reports
    infeasibility and leaves the base distribution unchanged.
    """
    target = _unit_interval(source_positive_mass, "source_positive_mass")
    p = normalized_pool_distribution(base_scores)
    g = _validate_groups(groups, len(p))
    pos_mask = g == POSITIVE
    neg_mask = g == NEGATIVE
    unk_mask = g == UNKNOWN
    pos_mass = float(p[pos_mask].sum())
    neg_mass = float(p[neg_mask].sum())
    unknown_mass = float(p[unk_mask].sum())
    mapped_mass = pos_mass + neg_mass

    if not pos_mask.any() or not neg_mask.any() or mapped_mass <= np.finfo(np.float64).eps:
        mapped_positive = (
            None if mapped_mass <= np.finfo(np.float64).eps else pos_mass / mapped_mass
        )
        return {
            "status": "infeasible_missing_semantic_side",
            "base_probabilities": p,
            "projected_probabilities": p.copy(),
            "source_positive_mass": target,
            "base_mapped_positive_mass": mapped_positive,
            "projected_mapped_positive_mass": mapped_positive,
            "unknown_mass": unknown_mass,
            "kl_q_to_p": 0.0,
        }

    q = np.zeros_like(p)
    target_pos = mapped_mass * target
    target_neg = mapped_mass * (1.0 - target)
    scores = np.asarray(base_scores, dtype=np.float64)
    q[pos_mask] = normalized_pool_distribution(scores[pos_mask]) * target_pos
    q[neg_mask] = normalized_pool_distribution(scores[neg_mask]) * target_neg
    q[unk_mask] = p[unk_mask]

    if not np.isclose(q.sum(), 1.0, atol=1e-12, rtol=1e-10):
        raise FloatingPointError("projection lost probability mass")
    positive = q > 0
    shifted = scores - scores.max()
    log_p = shifted - np.log(np.exp(shifted).sum())
    kl = float(np.sum(q[positive] * (np.log(q[positive]) - log_p[positive])))
    if not math.isfinite(kl):
        raise FloatingPointError("nonfinite projection divergence")
    return {
        "status": "projected",
        "base_probabilities": p,
        "projected_probabilities": q,
        "source_positive_mass": target,
        "base_mapped_positive_mass": pos_mass / mapped_mass,
        "projected_mapped_positive_mass": target,
        "unknown_mass": unknown_mass,
        "kl_q_to_p": kl,
    }


def stable_select(probabilities: Sequence[float], base_probabilities: Sequence[float]) -> int:
    """Argmax projected mass, breaking numerical ties by the base distribution."""
    q = np.asarray(probabilities, dtype=np.float64)
    p = np.asarray(base_probabilities, dtype=np.float64)
    if q.ndim != 1 or p.shape != q.shape or not np.isfinite(q).all() or not np.isfinite(p).all():
        raise ValueError("aligned finite probability vectors required")
    maximum = float(q.max())
    tied = np.flatnonzero(np.isclose(q, maximum, rtol=1e-12, atol=1e-15))
    if len(tied) == 1:
        return int(tied[0])
    return int(tied[int(np.argmax(p[tied]))])


def project_free_text_candidates(
    candidates: Sequence[str],
    base_scores: Sequence[float],
    groups: Sequence[str],
    source_positive_mass: float,
) -> dict[str, Any]:
    """Project and select one complete free-text answer without rewriting it."""
    if len(candidates) != len(base_scores) or not candidates:
        raise ValueError("candidates and scores must be aligned and nonempty")
    if any(not isinstance(text, str) or not text.strip() for text in candidates):
        raise ValueError("every candidate must be nonempty text")
    result = project_binary_authority(base_scores, groups, source_positive_mass)
    base_idx = stable_select(result["base_probabilities"], result["base_probabilities"])
    if result["status"] == "projected":
        selected = stable_select(result["projected_probabilities"], result["base_probabilities"])
    else:
        selected = base_idx
    return {
        **result,
        "base_index": base_idx,
        "selected_index": selected,
        "base_text": candidates[base_idx],
        "selected_text": candidates[selected],
        "free_text_output_preserved": True,
    }


def _normalize_text(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


_NEGATIVE_PATTERNS = (
    r"\bno evidence of\b",
    r"\bno sign of\b",
    r"\bnegative for\b",
    r"\bwithout\b",
    r"\babsent\b",
    r"\bnot present\b",
    r"\bnot seen\b",
    r"\bnot evident\b",
    r"\bnot demonstrated\b",
    r"\bdoes not (?:show|demonstrate|reveal)\b",
    r"\bthere (?:is|are) no\b",
)


def binary_finding_group(answer: str, aliases: Sequence[str]) -> str:
    """Map a complete free-text answer to the one binary attribute under study.

    This conservative mapper only supports the frozen whole-image presence pilot.
    It is not a general medical NLI system. Leading yes/no must agree with any
    explicit finding clauses. Uncertain, historical or ambiguous clauses remain
    unknown; negation is restricted to the clause containing the finding.
    """
    text = _normalize_text(answer)
    if not text:
        return UNKNOWN
    cleaned_aliases = [_normalize_text(alias) for alias in aliases if str(alias).strip()]
    labels = set()
    for clause in re.split(r"[.;!?]|\bbut\b|\bhowever\b", text):
        matches = [m for alias in cleaned_aliases
                   for m in re.finditer(r"\b" + re.escape(alias) + r"\b", clause)]
        if not matches:
            continue
        if re.search(r"\b(possible|possibly|may|might|could|uncertain|suspected|history|historical|previous|resolved)\b|cannot exclude|can't exclude|rule out", clause):
            return UNKNOWN
        # Multiple assertions in one clause cannot be reliably scoped by this mapper.
        if re.search(r"\b(and|or|while|although)\b|,", clause):
            return UNKNOWN
        negative = any(re.search(pattern, clause) for pattern in _NEGATIVE_PATTERNS)
        negative |= any(re.search(r"\b(?:no|not|without)\b", clause[:m.start()]) is not None for m in matches)
        labels.add(NEGATIVE if negative else POSITIVE)
    leading = re.match(r"^(yes|no)\b", text)
    if leading:
        labels.add(POSITIVE if leading.group(1) == "yes" else NEGATIVE)
    return next(iter(labels)) if len(labels) == 1 else UNKNOWN


def transport_diagnostic(
    base_scores: Sequence[float],
    conditioned_scores: Sequence[float],
    groups: Sequence[str],
    source_positive_mass: float,
) -> dict[str, Any]:
    """Measure whether language transport moves the mapped marginal toward source semantics."""
    source = _unit_interval(source_positive_mass, "source_positive_mass")
    p0 = normalized_pool_distribution(base_scores)
    pe = normalized_pool_distribution(conditioned_scores)
    base_mass = mapped_positive_mass(p0, groups)
    evidence_mass = mapped_positive_mass(pe, groups)
    if base_mass is None or evidence_mass is None:
        return {
            "status": "unmapped",
            "base_positive_mass": base_mass,
            "conditioned_positive_mass": evidence_mass,
            "source_positive_mass": source,
            "movement": None,
            "toward_source": None,
        }
    desired_delta = source - base_mass
    observed_delta = evidence_mass - base_mass
    if abs(desired_delta) <= 1e-12:
        toward = abs(observed_delta) <= 1e-12
    else:
        toward = observed_delta * desired_delta > 0
    return {
        "status": "measured",
        "base_positive_mass": base_mass,
        "conditioned_positive_mass": evidence_mass,
        "source_positive_mass": source,
        "movement": observed_delta,
        "toward_source": bool(toward),
        "distance_before": abs(source - base_mass),
        "distance_after": abs(source - evidence_mass),
    }
