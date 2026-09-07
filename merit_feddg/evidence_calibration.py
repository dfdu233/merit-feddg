"""Source-only evidence-action selection with independent group confirmation.

No target labels, regression controller, statistical coverage guarantee or
synthetic domain labels. Proxy groups are reported but cannot qualify a policy.
The optimized quantity is the medical-content effect against a format-null arm;
the selected arm must also improve on the untouched generalist.
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
        intervention = row.get("intervention")
        if intervention not in {"direct", "bounded"}:
            raise ValueError("intervention must be direct or bounded")
        key = (row["scope"], row["sample_id"], intervention, row["strength"])
        if key in seen:
            raise ValueError("duplicate intervention record")
        seen.add(key)
        group, domain = row["group_id"], row["domain"]
        if group in group_domains and group_domains[group] != domain:
            raise ValueError("one independent group cannot belong to multiple source domains")
        group_domains[group] = domain
        values = [
            row.get("gain"),
            row.get("output_gain"),
            row.get("control_gain"),
            row["strength"],
        ]
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
            raise ValueError("invalid content/output/control gain or strength")
        if (not np.isfinite(values).all() or any(not -1 <= value <= 1 for value in values[:3])
                or row["strength"] <= 0):
            raise ValueError("invalid content/output/control gain or strength")
        if not np.isclose(row["output_gain"] - row["control_gain"], row["gain"], atol=1e-9):
            raise ValueError("content gain must equal output gain minus control gain")
        scopes[row["scope"]].append(row)
    cards = {}
    for scope, data in scopes.items():
        domains = sorted({r["domain"] for r in data})
        actions = sorted(
            {(r["intervention"], float(r["strength"])) for r in data},
            key=lambda action: (action[0] != "direct", action[1]),
        )
        kinds = sorted({r["domain_kind"] for r in data})
        card = {"qualified": False, "status": "insufficient_support", "strength": 0.0,
                "domain_kinds": kinds, "domains": domains, "source_only": True}
        cards[scope] = card
        if any(kind not in {"hospital", "center", "dataset"} for kind in kinds):
            card["status"] = "proxy_or_unverified_domains"
            continue
        # Require identical observed cases across actions: a failed/missing
        # branch must not silently produce an easier calibration cohort.
        cohorts = [{(r["sample_id"], r["group_id"], r["domain"])
                    for r in data
                    if (r["intervention"], float(r["strength"])) == action}
                   for action in actions]
        if any(cohort != cohorts[0] for cohort in cohorts[1:]):
            raise ValueError("evidence actions require matching source cohorts")

        def gains(action, partition, domain, field="gain", data=data):
            groups = defaultdict(list)
            for r in data:
                if ((r["intervention"], float(r["strength"])) == action
                        and r["domain"] == domain
                        and calibration_partition(r["group_id"]) == partition):
                    groups[r["group_id"]].append(r[field])
            return [float(np.mean(values)) for values in groups.values()]

        counts = {d: {p: len(gains(actions[0], p, d)) for p in ("select", "confirm")}
                  for d in domains}
        card["independent_groups"] = counts
        if len(domains) < min_domains or any(n < min_groups for c in counts.values() for n in c.values()):
            continue
        # Select on the weaker of content-vs-null and output-vs-generalist gains.
        # Direct context wins exact ties because it needs one decoding branch;
        # smaller bounded strengths then break ties. Confirmation cannot reselect.
        def guard(action, domains=tuple(domains), gains=gains):
            content = min(np.mean(gains(action, "select", d, "gain")) for d in domains)
            output = min(np.mean(gains(action, "select", d, "output_gain")) for d in domains)
            return min(content, output)

        selected = max(
            actions,
            key=lambda action: (guard(action), action[0] == "direct", -action[1]),
        )
        selection_content = {
            d: float(np.mean(gains(selected, "select", d, "gain"))) for d in domains
        }
        confirmation_content = {
            d: float(np.mean(gains(selected, "confirm", d, "gain"))) for d in domains
        }
        selection_output = {
            d: float(np.mean(gains(selected, "select", d, "output_gain"))) for d in domains
        }
        confirmation_output = {
            d: float(np.mean(gains(selected, "confirm", d, "output_gain"))) for d in domains
        }
        qualified = min(
            *selection_content.values(),
            *confirmation_content.values(),
            *selection_output.values(),
            *confirmation_output.values(),
        ) > 0
        card.update(
            qualified=qualified,
            status="source_confirmed" if qualified else "nonpositive_content_or_output_gain",
            intervention=selected[0] if qualified else "none",
            strength=selected[1] if qualified else 0.0,
            selected_intervention=selected[0],
            selected_strength=selected[1],
            selection_content_gain=selection_content,
            confirmation_content_gain=confirmation_content,
            selection_output_gain=selection_output,
            confirmation_output_gain=confirmation_output,
            selection_guard=guard(selected),
        )
    return {"schema": "source-evidence-calibration-v2", "cards": cards,
            "min_groups_per_domain_per_partition": min_groups, "min_domains": min_domains,
            "limitations": ["Empirical independent-group confirmation is not a statistical safety bound.",
                            "Dataset domains are not necessarily independent hospitals.",
                            "Format-null envelopes are not tokenizer-length matched.",
                            "No LODO or native multi-view stability is implemented in this profile."]}
