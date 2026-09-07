"""Small source-only guidance selection with independent group confirmation.

No target labels, regression controller, statistical coverage guarantee or
synthetic domain labels. Proxy groups are reported but cannot qualify a policy.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .open_study import fingerprint


def calibration_partition(group_id):
    # Global group ID: the same patient/image can never appear in both halves.
    return "select" if int(fingerprint(str(group_id))[:8], 16) % 2 == 0 else "confirm"


def fit_evidence_calibration(records, *, min_groups=8, min_domains=2):
    if (type(min_groups) is not int or min_groups < 2
            or type(min_domains) is not int or min_domains < 2):
        raise ValueError("at least two independent groups and domains required")
    scopes = defaultdict(list)
    seen = set()
    group_domains = {}
    for row in records:
        if row.get("role") != "source":
            raise ValueError("only source outcomes may calibrate guidance")
        key = (row["scope"], row["sample_id"], row["strength"])
        if key in seen:
            raise ValueError("duplicate intervention record")
        seen.add(key)
        group, domain = row["group_id"], row["domain"]
        if group in group_domains and group_domains[group] != domain:
            raise ValueError("one independent group cannot belong to multiple source domains")
        group_domains[group] = domain
        if (not np.isfinite([row["gain"], row["strength"]]).all()
                or not -1 <= row["gain"] <= 1 or row["strength"] <= 0):
            raise ValueError("invalid continuous gain or strength")
        scopes[row["scope"]].append(row)
    cards = {}
    for scope, data in scopes.items():
        domains = sorted({r["domain"] for r in data})
        strengths = sorted({r["strength"] for r in data})
        kinds = sorted({r["domain_kind"] for r in data})
        card = {"qualified": False, "status": "insufficient_support", "strength": 0.0,
                "domain_kinds": kinds, "domains": domains, "source_only": True}
        cards[scope] = card
        if any(kind not in {"hospital", "center", "dataset"} for kind in kinds):
            card["status"] = "proxy_or_unverified_domains"
            continue
        # Require identical observed cases across strengths: a failed/missing
        # branch must not silently produce an easier calibration cohort.
        cohorts = [{(r["sample_id"], r["group_id"], r["domain"])
                    for r in data if r["strength"] == s} for s in strengths]
        if any(cohort != cohorts[0] for cohort in cohorts[1:]):
            raise ValueError("strengths require matching source cohorts")

        def gains(strength, partition, domain, data=data):
            groups = defaultdict(list)
            for r in data:
                if (r["strength"] == strength and r["domain"] == domain
                        and calibration_partition(r["group_id"]) == partition):
                    groups[r["group_id"]].append(r["gain"])
            return [float(np.mean(values)) for values in groups.values()]

        counts = {d: {p: len(gains(strengths[0], p, d)) for p in ("select", "confirm")}
                  for d in domains}
        card["independent_groups"] = counts
        if len(domains) < min_domains or any(n < min_groups for c in counts.values() for n in c.values()):
            continue
        # Choose on selection groups, domain-balanced worst-domain mean gain.
        # Smaller strengths break ties. Confirmation cannot change this choice.
        selected = max(strengths, key=lambda s: (min(np.mean(gains(s, "select", d))
                                                    for d in domains), -s))
        selection = {d: float(np.mean(gains(selected, "select", d))) for d in domains}
        confirmation = {d: float(np.mean(gains(selected, "confirm", d))) for d in domains}
        qualified = min(selection.values()) > 0 and min(confirmation.values()) > 0
        card.update(qualified=qualified, status="source_confirmed" if qualified else "nonpositive_gain",
                    strength=selected if qualified else 0.0, selected_strength=selected,
                    selection_gain=selection, confirmation_gain=confirmation)
    return {"schema": "source-guidance-calibration-v1", "cards": cards,
            "min_groups_per_domain_per_partition": min_groups, "min_domains": min_domains,
            "limitations": ["Empirical independent-group confirmation is not a statistical safety bound.",
                            "Dataset domains are not necessarily independent hospitals.",
                            "No LODO or native multi-view stability is implemented in this profile."]}
