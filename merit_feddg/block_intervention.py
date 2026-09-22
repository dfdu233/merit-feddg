"""Counterfactual semantic-block intervention for frozen generalist/specialist VLMs.

MERIT-Block treats expert information as a temporary contextual intervention:

1. Generalist and expert branches propose competing continuations from the exact
   same committed prefix.
2. The expert continuation must have absolute support under the real expert
   evidence.
3. That preference must also be larger than the preference induced by matched
   wrong-patient/control expert evidence (counterfactual specificity).
4. The complete block is committed atomically. In the default ephemeral mode,
   expert context is discarded immediately after the commit; only the committed
   token IDs survive.

This module does NOT claim to verify medical truth. It measures whether an
expert-induced continuation is specifically supported by the current expert
context rather than generic contextual entrainment, while limiting how long
that context can influence the autoregressive trajectory.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import median
from time import perf_counter

import numpy as np

from .block_decode import Block
from .vector_gate import token_log_probability


@dataclass(frozen=True)
class ExpertBlockBranch:
    """One real expert branch and its counterfactual controls."""

    expert_id: str
    real_session: object
    control_sessions: tuple[object, ...] = ()
    control_kind: str = "matched_wrong_patient"

    def __post_init__(self):
        if not self.expert_id:
            raise ValueError("expert_id cannot be empty")
        if self.control_kind not in {
            "matched_wrong_patient",
            "random_wrong_patient",
            "none",
        }:
            raise ValueError("unsupported counterfactual control kind")
        if self.control_kind == "none" and self.control_sessions:
            raise ValueError("no-control branch cannot carry control sessions")


@dataclass(frozen=True)
class BlockInterventionConfig:
    max_new_tokens: int = 64
    block_tokens: int = 8
    block_mode: str = "fixed"
    sentence_max_tokens: int = 32
    decision_rule: str = "conjunction"
    context_policy: str = "ephemeral"
    score_reduction: str = "mean"
    min_controls: int = 3
    numerical_epsilon: float = 1e-8

    def __post_init__(self):
        for name, value in (
            ("max_new_tokens", self.max_new_tokens),
            ("block_tokens", self.block_tokens),
            ("sentence_max_tokens", self.sentence_max_tokens),
            ("min_controls", self.min_controls),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.block_mode not in {"fixed", "sentence"}:
            raise ValueError("block_mode must be fixed or sentence")
        if self.decision_rule not in {"conjunction", "gamma_only", "real_only"}:
            raise ValueError("unsupported block admission rule")
        if self.context_policy not in {"ephemeral", "persistent"}:
            raise ValueError("context_policy must be ephemeral or persistent")
        if self.score_reduction not in {"mean", "sum"}:
            raise ValueError("score_reduction must be mean or sum")
        if (
            type(self.numerical_epsilon) not in (int, float)
            or not math.isfinite(self.numerical_epsilon)
            or self.numerical_epsilon < 0
        ):
            raise ValueError("numerical_epsilon must be finite and nonnegative")


_SENTENCE_ENDINGS = (".", "?", "!", "。", "？", "！", "\n")


def _sentence_prefix(session, block: Block) -> Block:
    """Return the earliest token prefix ending a semantic sentence."""
    if len(block.tokens) <= 1:
        return block
    for end in range(1, len(block.tokens) + 1):
        text = session.decode(block.tokens[:end]).strip()
        if text and text.endswith(_SENTENCE_ENDINGS):
            return Block(
                tokens=tuple(block.tokens[:end]),
                text=text,
                log_probability=block.log_probability,
                finished=bool(block.finished and end == len(block.tokens)),
            )
    return block


def _propose(session, prefix, config: BlockInterventionConfig, remaining):
    if config.block_mode == "fixed":
        length = min(config.block_tokens, remaining)
    else:
        length = min(config.sentence_max_tokens, remaining)
    block = session.propose(tuple(prefix), count=1, length=length)[0]
    if not block.tokens:
        raise ValueError("block proposal cannot be empty")
    if len(block.tokens) > remaining:
        raise ValueError("block proposal exceeded remaining token budget")
    if config.block_mode == "sentence":
        block = _sentence_prefix(session, block)
    return block


def _sequence_score(
    session,
    prefix,
    tokens,
    *,
    reduction,
    cache,
    audit,
):
    """Teacher-force one candidate continuation under one branch."""
    tokens = tuple(int(value) for value in tokens)
    if not tokens:
        raise ValueError("cannot score an empty block")
    if hasattr(session, "sequence_mean_logp"):
        audit["sequence_score_calls"] += 1
        mean_value = float(session.sequence_mean_logp(tuple(prefix), tokens))
        if not math.isfinite(mean_value):
            raise ValueError("nonfinite sequence score")
        return mean_value if reduction == "mean" else mean_value * len(tokens)

    total = 0.0
    for index, token in enumerate(tokens):
        context = tuple(prefix) + tokens[:index]
        key = (id(session), context)
        if key not in cache:
            audit["next_score_queries"] += 1
            cache[key] = session.next_scores(context)
        total += token_log_probability(cache[key], token)
    if not math.isfinite(total):
        raise ValueError("nonfinite teacher-forced block score")
    return total / len(tokens) if reduction == "mean" else total


def evaluate_expert_block(
    *,
    prefix,
    base_block,
    expert_block,
    branch: ExpertBlockBranch,
    config: BlockInterventionConfig,
    cache=None,
):
    """Evaluate one competing expert block at an exact shared prefix."""
    started = perf_counter()
    prefix = tuple(prefix)
    cache = {} if cache is None else cache
    audit = {
        "schema": "merit-counterfactual-block-v1",
        "expert_id": branch.expert_id,
        "control_kind": branch.control_kind,
        "decision_rule": config.decision_rule,
        "score_reduction": config.score_reduction,
        "accepted": False,
        "absolute_support": False,
        "counterfactual_specificity": False,
        "medical_correctness_guaranteed": False,
        "interpretation": (
            "patient-specific contextual support for a competing continuation; "
            "not a proof of medical truth"
        ),
        "next_score_queries": 0,
        "sequence_score_calls": 0,
    }
    try:
        if tuple(base_block.tokens) == tuple(expert_block.tokens):
            audit.update(reason="identical_continuation", seconds=perf_counter() - started)
            return audit

        real_expert = _sequence_score(
            branch.real_session,
            prefix,
            expert_block.tokens,
            reduction=config.score_reduction,
            cache=cache,
            audit=audit,
        )
        real_base = _sequence_score(
            branch.real_session,
            prefix,
            base_block.tokens,
            reduction=config.score_reduction,
            cache=cache,
            audit=audit,
        )
        real_margin = real_expert - real_base
        epsilon = config.numerical_epsilon
        absolute = real_margin > epsilon

        control_margins = []
        for session in branch.control_sessions:
            ctrl_expert = _sequence_score(
                session,
                prefix,
                expert_block.tokens,
                reduction=config.score_reduction,
                cache=cache,
                audit=audit,
            )
            ctrl_base = _sequence_score(
                session,
                prefix,
                base_block.tokens,
                reduction=config.score_reduction,
                cache=cache,
                audit=audit,
            )
            control_margins.append(ctrl_expert - ctrl_base)

        control_median = None
        gamma = None
        specificity = False
        if control_margins:
            control_median = float(median(control_margins))
            gamma = real_margin - control_median
            specificity = gamma > epsilon

        if config.decision_rule == "real_only":
            accepted = absolute
            reason = "positive_absolute_support" if accepted else "no_absolute_support"
        elif len(control_margins) < config.min_controls:
            accepted = False
            reason = "insufficient_counterfactual_controls"
        elif config.decision_rule == "gamma_only":
            accepted = specificity
            reason = (
                "positive_counterfactual_specificity"
                if accepted
                else "no_counterfactual_specificity"
            )
        else:
            accepted = absolute and specificity
            reason = (
                "absolute_support_and_counterfactual_specificity"
                if accepted
                else "missing_absolute_support"
                if not absolute
                else "missing_counterfactual_specificity"
            )

        audit.update(
            accepted=bool(accepted),
            reason=reason,
            real_expert_score=float(real_expert),
            real_base_score=float(real_base),
            real_margin=float(real_margin),
            absolute_support=bool(absolute),
            control_margins=[float(value) for value in control_margins],
            control_median=control_median,
            gamma=None if gamma is None else float(gamma),
            counterfactual_specificity=bool(specificity),
            control_count=len(control_margins),
            selection_score=float(gamma if gamma is not None else real_margin),
        )
    except (ValueError, TypeError, FloatingPointError) as exc:
        audit.update(
            accepted=False,
            reason=f"invalid_block_evidence:{type(exc).__name__}:{exc}",
        )
    audit["seconds"] = perf_counter() - started
    return audit


def decode_counterfactual_blocks(
    base_session,
    expert_branches: Mapping[str, ExpertBlockBranch],
    *,
    config: BlockInterventionConfig,
):
    """Decode with same-prefix expert proposals and atomic block commits."""
    if not expert_branches:
        raise ValueError("at least one expert block branch is required")
    if set(expert_branches) != {
        branch.expert_id for branch in expert_branches.values()
    }:
        raise ValueError("expert branch mapping keys must match expert IDs")

    prefix: tuple[int, ...] = ()
    trace = []
    started = perf_counter()
    incumbent_session = base_session
    persistent_expert = None

    while len(prefix) < config.max_new_tokens:
        remaining = config.max_new_tokens - len(prefix)
        base_block = _propose(incumbent_session, prefix, config, remaining)
        block_audits = []
        shared_cache = {}

        for expert_id, branch in expert_branches.items():
            expert_block = _propose(branch.real_session, prefix, config, remaining)
            audit = evaluate_expert_block(
                prefix=prefix,
                base_block=base_block,
                expert_block=expert_block,
                branch=branch,
                config=config,
                cache=shared_cache,
            )
            audit["candidate"] = {
                "text": expert_block.text,
                "token_ids": list(expert_block.tokens),
                "finished": expert_block.finished,
            }
            block_audits.append((expert_id, expert_block, audit))

        admissible = [
            row for row in block_audits if row[2].get("accepted", False)
        ]
        if admissible:
            selected_expert, selected_block, _selected_audit = max(
                admissible,
                key=lambda row: (
                    row[2].get("selection_score", float("-inf")),
                    row[2].get("real_margin", float("-inf")),
                    row[0],
                ),
            )
            accepted = True
        else:
            selected_expert = None
            selected_block = base_block
            accepted = False

        before = base_session.decode(prefix)
        trace.append(
            {
                "block": len(trace),
                "token_start": len(prefix),
                "token_end": len(prefix) + len(selected_block.tokens),
                "prefix_text": before,
                "block_mode": config.block_mode,
                "context_policy": config.context_policy,
                "incumbent_context_expert": persistent_expert,
                "base_candidate": {
                    "text": base_block.text,
                    "token_ids": list(base_block.tokens),
                    "finished": base_block.finished,
                },
                "experts": {
                    expert_id: audit
                    for expert_id, _block, audit in block_audits
                },
                "selected_expert": selected_expert,
                "expert_block_committed": accepted,
                "selected_block": {
                    "text": selected_block.text,
                    "token_ids": list(selected_block.tokens),
                    "finished": selected_block.finished,
                },
                "expert_context_discarded_after_commit": (
                    config.context_policy == "ephemeral"
                ),
            }
        )

        prefix = (*prefix, *selected_block.tokens)
        if selected_block.finished:
            break

        if config.context_policy == "ephemeral":
            incumbent_session = base_session
            persistent_expert = None
        elif accepted:
            incumbent_session = expert_branches[selected_expert].real_session
            persistent_expert = selected_expert

    return {
        "schema": "merit-block-intervention-run-v1",
        "text": base_session.decode(prefix).strip(),
        "token_ids": list(prefix),
        "trace": trace,
        "seconds": perf_counter() - started,
        "method": {
            "same_prefix_competing_blocks": True,
            "decision_rule": config.decision_rule,
            "context_policy": config.context_policy,
            "block_mode": config.block_mode,
            "block_tokens": config.block_tokens,
            "sentence_max_tokens": config.sentence_max_tokens,
            "score_reduction": config.score_reduction,
            "min_controls": config.min_controls,
            "trained_gate": False,
            "medical_correctness_guaranteed": False,
        },
    }


def _finite_distribution(scores):
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or not len(values) or np.isnan(values).any():
        raise ValueError("invalid score vector")
    finite = np.isfinite(values)
    if not finite.any():
        raise ValueError("score vector has no finite support")
    maximum = np.max(values[finite])
    probs = np.zeros_like(values)
    exp = np.exp(values[finite] - maximum)
    probs[finite] = exp / exp.sum()
    return probs


def _js_divergence(left, right):
    p = _finite_distribution(left)
    q = _finite_distribution(right)
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * (np.log(a[mask]) - np.log(b[mask]))))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def measure_same_prefix_context_drift(
    base_session,
    persistent_expert_session,
    *,
    prefix,
    horizon=16,
):
    """Measure residual expert-context influence after a committed block."""
    if type(horizon) is not int or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    shared = tuple(prefix)
    rows = []
    eos_ids = set(getattr(base_session, "eos_ids", ()))
    for step in range(horizon):
        base_scores = base_session.next_scores(shared)
        expert_scores = persistent_expert_session.next_scores(shared)
        token = int(np.argmax(_finite_distribution(base_scores)))
        rows.append(
            {
                "step": step,
                "prefix_length": len(shared),
                "js_divergence": _js_divergence(base_scores, expert_scores),
                "base_token": token,
                "expert_greedy_token": int(
                    np.argmax(_finite_distribution(expert_scores))
                ),
            }
        )
        shared = (*shared, token)
        if token in eos_ids:
            break
    divergences = [row["js_divergence"] for row in rows]
    return {
        "schema": "expert-context-drift-v1",
        "horizon": horizon,
        "steps": rows,
        "mean_js": float(np.mean(divergences)) if divergences else 0.0,
        "max_js": float(np.max(divergences)) if divergences else 0.0,
        "greedy_disagreement_rate": (
            float(
                np.mean(
                    [
                        row["base_token"] != row["expert_greedy_token"]
                        for row in rows
                    ]
                )
            )
            if rows
            else 0.0
        ),
        "interpretation": (
            "same-prefix residual context influence after an intervention; "
            "not a correctness metric"
        ),
    }


def native_expert_block_branches(
    native_session,
    *,
    real_items: Mapping[str, Sequence[object]],
    control_items: Mapping[str, Sequence[Sequence[object]]],
    control_kind="matched_wrong_patient",
):
    """Turn already-frozen native evidence packets into isolated block branches."""
    from .capability_runtime import NativeState

    branches = {}
    for expert_id, items in real_items.items():
        controls = control_items.get(expert_id, ())
        branches[expert_id] = ExpertBlockBranch(
            expert_id=expert_id,
            real_session=native_session._evidence_session(
                NativeState(items=tuple(items))
            ),
            control_sessions=tuple(
                native_session._evidence_session(
                    NativeState(items=tuple(values))
                )
                for values in controls
            ),
            control_kind=control_kind if controls else "none",
        )
    return branches
