"""Source-only, continuous generation-utility qualification (not error prediction)."""

from __future__ import annotations

import re
from collections import Counter, defaultdict

import numpy as np


def normalized_tokens(text: str) -> list[str]:
    # Deliberately preserve negations and numbers. Lexical F1 is not factuality.
    return re.findall(r"\w+", text.casefold())


def answer_metrics(text: str, references: list[str]) -> dict:
    if not references or any(not isinstance(r, str) or not r.strip() for r in references):
        raise ValueError("at least one nonempty textual reference required")
    predicted = normalized_tokens(text)
    scores, exact = [], []
    for reference in references:
        expected = normalized_tokens(reference)
        overlap = sum((Counter(predicted) & Counter(expected)).values())
        scores.append(2 * overlap / max(1, len(predicted) + len(expected)))
        exact.append(float(predicted == expected))
    return {"token_f1": max(scores), "exact_match": max(exact)}


def qualify_contribution(
    rows: list[dict], *, min_per_domain: int = 8, min_domains: int = 2, penalty: float = 1.645
) -> dict:
    """Worst-source mean-minus-SE margin. A heuristic, NOT a coverage guarantee.

    Rows are paired FULL source generations, using the fixed intervention strength.
    Grouping at image/patient level must be performed before calling this function.
    Missing support cannot authorize loading a target expert.
    """
    if min_per_domain < 2 or min_domains < 2 or penalty < 0:
        raise ValueError("qualification needs >=2 domains and >=2 independent units/domain")
    grouped = defaultdict(list)
    for row in rows:
        if row.get("role") != "source":
            raise ValueError("target rows cannot fit contribution qualification")
        gain = float(row["guided_f1"]) - float(row["base_f1"])
        if not np.isfinite(gain) or not -1 <= gain <= 1:
            raise ValueError("invalid paired gain")
        grouped[str(row["domain"])].append(gain)
    domains = {}
    for domain, gains in sorted(grouped.items()):
        x = np.asarray(gains)
        margin = float(x.mean() - penalty * x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else -1.0
        domains[domain] = {"n": len(x), "mean_gain": float(x.mean()), "margin": margin}
    supported = len(domains) >= min_domains and all(
        d["n"] >= min_per_domain for d in domains.values()
    )
    robust_gain = min((d["margin"] for d in domains.values()), default=-1.0)
    return {
        "qualified": supported and robust_gain > 0,
        "robust_gain": robust_gain,
        "domains": domains,
        "support_sufficient": supported,
        "estimator": "worst_source_mean_minus_se_heuristic",
        "metric": "token_f1",
    }
