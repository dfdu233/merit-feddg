"""Frozen same-vocabulary evidence guidance; no labels, fitting or model loading.

Independently implemented after studying ThinkOmni's public guidance decoder
(ICLR 2026). Not copied upstream code or a reproduction of its three-model rule.
KL bounds distribution movement, NOT clinical error or arbitrary-domain safety.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GuidanceConfig:
    strength: float = 0.5
    clip: float = 2.0
    token_kl: float = 0.02
    case_kl: float = 0.32
    evidence_tokens: int = 16
    control: str = "real"

    def __post_init__(self):
        values = (self.strength, self.clip, self.token_kl, self.case_kl)
        if any(isinstance(v, bool) for v in values) or not np.isfinite(values).all():
            raise ValueError("guidance budgets must be finite numbers")
        if min(values) < 0 or type(self.evidence_tokens) is not int or self.evidence_tokens < 1:
            raise ValueError("nonnegative budgets and positive evidence lifetime required")
        if self.control not in {"real", "format_only"}:
            raise ValueError("unknown evidence control")


def _log_probs(scores):
    x = np.asarray(scores, dtype=np.float64)
    if x.ndim != 1 or not x.size or np.isnan(x).any() or np.isposinf(x).any():
        raise ValueError("scores must be a nonempty vector without NaN/+inf")
    valid = np.isfinite(x)
    if not valid.any():
        raise ValueError("empty vocabulary support")
    y = x - x[valid].max()
    return y - np.log(np.exp(y[valid]).sum()), valid


def bounded_distribution(base_scores, evidence_scores, config, *, spent=0.0):
    """Return normalized log probabilities and small, JSON-safe audit metadata.

    Common -inf masks are preserved. Different supports fail, not intersected
    silently. The KL is monotone in strength for this fixed residual, enabling
    a feasible-side bisection. No renormalized expert confidence is used.
    """
    if not np.isfinite(spent) or spent < 0 or spent > config.case_kl + 1e-9:
        raise ValueError("invalid spent KL budget")
    base, mask = _log_probs(base_scores)
    budget = max(0.0, min(config.token_kl, config.case_kl - spent))
    info = {"strength": 0.0, "kl": 0.0, "spent": float(spent),
            "remaining": max(0.0, config.case_kl - spent), "reason": "NONE:budget_or_empty",
            "base_selected": int(np.argmax(base)), "residual_max": 0.0}
    if evidence_scores is None or min(budget, config.strength, config.clip) == 0:
        return base, info
    evidence, evidence_mask = _log_probs(evidence_scores)
    if evidence.shape != base.shape or not np.array_equal(mask, evidence_mask):
        raise ValueError("branches must share the same vocabulary support")
    residual = np.clip(evidence[mask] - base[mask], -config.clip, config.clip)
    info["residual_max"] = float(np.max(np.abs(residual)))

    def candidate(strength):
        values, _ = _log_probs(base[mask] + strength * residual)
        kl = max(0.0, float(np.sum(np.exp(values) * (values - base[mask]))))
        return values, kl

    strength = config.strength
    values, kl = candidate(strength)
    if kl > budget:
        low, high = 0.0, strength
        for _ in range(48):
            mid = (low + high) / 2
            if candidate(mid)[1] <= budget:
                low = mid
            else:
                high = mid
        strength = low
        values, kl = candidate(strength)
    output = base.copy()
    output[mask] = values
    info.update(strength=float(strength), kl=kl, spent=float(spent + kl),
                remaining=max(0.0, config.case_kl - spent - kl), reason="bounded_evidence")
    return output, info
