"""Strict base-relative consensus on isolated receiver branches.

This adapts the q=m probability rule in Narang et al. (2026) to evidence-
conditioned branches of one frozen receiver. It is not their trained-reference
model experiment. A relaxed quorum is intentionally not implemented here:
the paper and released code differ for its downward branch when q<m.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .evidence_decode import _log_probs


def strict_base_relative_consensus_step(base_scores, branch_scores):
    """Choose a token from the renormalized weakest unanimously signed shifts.

    All branches must have exactly the base vocabulary support. For each token,
    an upward shift uses the smallest branch probability, a downward shift uses
    the largest branch probability, and mixed/zero shifts retain the base
    probability. This is a probability rule, not a minimum of raw logits.
    """
    base_logp, mask = _log_probs(base_scores)
    valid_ids = np.flatnonzero(mask)
    base = np.exp(base_logp[mask])
    base_token = int(valid_ids[np.argmax(base)])
    if not branch_scores:
        return base_token, {
            "reason": "no_expert_branch",
            "base_token": base_token,
            "selected_token": base_token,
            "expert_count": 0,
            "quorum": 0,
            "changed_from_base": False,
        }

    branches = []
    for scores in branch_scores:
        logp, branch_mask = _log_probs(scores)
        if logp.shape != base_logp.shape or not np.array_equal(branch_mask, mask):
            raise ValueError("all consensus branches must share exact vocabulary support")
        branches.append(np.exp(logp[mask]))
    values = np.stack(branches, axis=0)
    shifts = values - base[None, :]
    all_up = np.all(shifts > 0.0, axis=0)
    all_down = np.all(shifts < 0.0, axis=0)
    raw = np.where(all_up, np.min(values, axis=0),
                   np.where(all_down, np.max(values, axis=0), base))
    total = float(raw.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("consensus distribution has no finite positive mass")
    selected = int(valid_ids[np.argmax(raw / total)])
    return selected, {
        "reason": "strict_base_relative_probability_consensus",
        "base_token": base_token,
        "selected_token": selected,
        "expert_count": len(branches),
        "quorum": len(branches),
        "changed_from_base": selected != base_token,
        "unanimous_up_tokens": int(all_up.sum()),
        "unanimous_down_tokens": int(all_down.sum()),
        "raw_mass_before_renormalization": total,
    }


def decode_strict_base_relative_consensus(base_session, expert_sessions: Mapping[str, object], *, max_tokens: int):
    """Free-decode one consensus trajectory at its own shared branch prefix."""
    if type(max_tokens) is not int or max_tokens < 1:
        raise ValueError("max_tokens must be a positive integer")
    names = list(expert_sessions)
    if len(names) != len(set(names)):
        raise ValueError("expert branch names must be unique")
    prefix, trace = [], []
    for step in range(max_tokens):
        base_scores = base_session.next_scores(prefix)
        branch_scores = [expert_sessions[name].next_scores(prefix) for name in names]
        token, audit = strict_base_relative_consensus_step(base_scores, branch_scores)
        trace.append({**audit, "step": step, "experts": names})
        prefix.append(token)
        if token in base_session.eos_ids:
            break
    return {
        "text": base_session.decode(prefix).strip(),
        "token_ids": prefix,
        "trace": trace,
        "expert_branches": names,
        "aggregation": "strict_base_relative_probability_q_equals_m",
        "bounded_commit": False,
    }
