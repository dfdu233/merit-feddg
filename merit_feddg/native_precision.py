"""Optional source-only calibration of a simultaneous native score error radius.

Independent implementation of the bounded monotone-loss CRC correction. This
module never partitions datasets. It consumes an existing calibration manifest;
the official-test runner does not fit or enable these cards by default.
"""

from __future__ import annotations

import math

from .open_study import fingerprint

LOSS = "native_simultaneous_score_error"


def fit_native_precision(rows, radii, *, alpha, expert, capability, scope):
    """Rows contain measured max absolute native-score error, NOT answer F1.

    Every patient/image group contributes one worst native error. For each
    represented real domain, select the smallest predeclared radius whose
    (sum(group losses)+1)/(n_groups+1) <= alpha. The maximum domain radius
    retains each domain's monotone loss bound under its exchangeability
    assumptions. It certifies neither unseen domains nor free-text answers.
    """
    if (not rows or type(alpha) not in (float, int) or not 0 < alpha < 1
            or not radii or any(type(r) not in (float, int) or not math.isfinite(r) or r < 0 for r in radii)
            or list(radii) != sorted(set(radii)) or capability != "classification"):
        raise ValueError("classification requires source errors, alpha in (0,1), and increasing radii")
    domains, owners = {}, {}
    for row in rows:
        if (row.get("split") not in {"source", "train"} or row.get("loss") != LOSS
                or row.get("domain_kind") not in {"hospital", "center", "site", "scanner"}
                or (row.get("expert"), row.get("capability"), row.get("scope")) != (expert, capability, scope)):
            raise ValueError("real source domains and exact native-loss/expert/scope identity required")
        error, domain, group = row.get("error"), row.get("domain"), row.get("group_id")
        if (type(error) not in (float, int) or not math.isfinite(error) or error < 0
                or not isinstance(domain, str) or not domain or not isinstance(group, str) or not group):
            raise ValueError("finite native error and explicit independent group/domain required")
        if group in owners and owners[group] != domain:
            raise ValueError("one calibration group cannot count as independent in multiple domains")
        owners[group] = domain
        groups = domains.setdefault(domain, {})
        groups[group] = max(groups.get(group, 0), error)
    summaries = {}
    for domain, groups in domains.items():
        n = len(groups)
        risk = [(sum(error > r for error in groups.values()) + 1) / (n + 1) for r in radii]
        choice = next((r for r, bound in zip(radii, risk, strict=True) if bound <= alpha), None)
        summaries[domain] = {"groups": n, "radius": choice, "corrected_risks": risk}
    supported = all(v["radius"] is not None for v in summaries.values())
    return {"schema": "native-precision-crc-v1", "expert": expert, "capability": capability,
            "scope": scope, "loss": LOSS, "alpha": alpha, "radii": list(radii),
            "radius": max(v["radius"] for v in summaries.values()) if supported else None,
            "status": "supported_on_recorded_domains" if supported else "insufficient_calibration",
            "domains": summaries, "calibration_sha256": fingerprint(rows),
            "assumptions": ["independent_exchangeable_groups_within_recorded_domain",
                            "fixed_predictor_and_predeclared_radius_grid",
                            "native_errors_measured_against_native_task_reference"],
            "unseen_domain_guarantee": False, "free_text_guarantee": False}


def precision_for(card, item, domain):
    if (card.get("schema") != "native-precision-crc-v1" or card.get("loss") != LOSS
            or (card.get("expert"), card.get("capability"), card.get("scope")) != (
                item.expert_id, item.capability, item.scope)):
        raise ValueError("native precision card identity mismatch")
    if domain not in card.get("domains", {}) or card.get("radius") is None:
        return {"status": "unsupported_domain", "loss": LOSS, "card_sha256": fingerprint(card)}
    radius = card["radius"]
    if type(radius) not in (int, float) or not math.isfinite(radius) or radius < 0:
        raise ValueError("invalid native radius")
    return {"status": "recorded_domain_assumptions_required", "radius": radius,
            "loss": LOSS, "card_sha256": fingerprint(card), "unseen_domain_guarantee": False}
