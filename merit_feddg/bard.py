"""Byzantine-anchored residual decoding for untrusted heterogeneous experts.

Each specialist is evaluated in an isolated receiver branch at the exact
committed prefix.  The specialist's self-reported confidence is never used as a
correctness score.  Instead, BARD robustly aggregates the change each specialist
induces in the frozen receiver's next-token distribution relative to the
generalist, then uses a declared bounded-fault consensus rule before allowing a
departure from the incumbent generalist.

This is a robustness mechanism, not a medical correctness certificate.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from .evidence_decode import _log_probs


@dataclass(frozen=True)
class BARDConfig:
    """Fixed inference controls.

    fault_budget is a threat-model declaration, not a learned/tuned confidence
    threshold.  fault_probe_scale is used only by the label-free diagnostic
    stress test and never by production token selection.
    """

    fault_budget: int = 1
    aggregation: str = "geometric_median"
    median_iterations: int = 12
    median_tolerance: float = 1e-6
    fault_probe_scale: float = 8.0
    single_expert_policy: str = "baseline"
    pair_policy: str = "unanimous"

    def __post_init__(self):
        if type(self.fault_budget) is not int or self.fault_budget < 0:
            raise ValueError("fault_budget must be a nonnegative integer")
        if self.aggregation not in {"mean", "coordinate_median", "geometric_median"}:
            raise ValueError("unsupported BARD aggregation")
        if self.single_expert_policy not in {"baseline"}:
            raise ValueError("single expert is not identifiable without an external falsification test")
        if self.pair_policy not in {"unanimous"}:
            raise ValueError("two-expert BARD currently supports unanimous non-forcing commit only")
        if type(self.median_iterations) is not int or self.median_iterations < 1:
            raise ValueError("median_iterations must be positive")
        numeric = (self.median_tolerance, self.fault_probe_scale)
        if any(not np.isfinite(v) or v <= 0 for v in numeric):
            raise ValueError("numeric BARD controls must be finite and positive")


def _prepare(base_scores, expert_scores):
    """Return normalized base scores and base-measure-centered receiver residuals."""
    base, mask = _log_probs(base_scores)
    valid_base = base[mask]
    p0 = np.exp(valid_base)
    residuals = []
    for scores in expert_scores:
        expert, expert_mask = _log_probs(scores)
        if expert.shape != base.shape or not np.array_equal(mask, expert_mask):
            raise ValueError("all branches must share exactly the same vocabulary support")
        residual = expert[mask] - valid_base
        # Remove branch-wide log-normalization offsets.  What remains is the
        # expert-conditioned receiver's relative change in token preference.
        residual -= float(np.dot(p0, residual))
        if not np.isfinite(residual).all():
            raise ValueError("nonfinite receiver residual")
        residuals.append(residual)
    matrix = (
        np.stack(residuals, axis=0)
        if residuals
        else np.empty((0, valid_base.size), dtype=np.float64)
    )
    return base, mask, matrix


def geometric_median(points, *, max_iterations=12, tolerance=1e-6, epsilon=1e-12):
    """Stabilized Weiszfeld geometric median over expert residual vectors."""
    values = np.asarray(points, dtype=np.float64)
    if (
        values.ndim != 2
        or not values.shape[0]
        or not values.shape[1]
        or not np.isfinite(values).all()
    ):
        raise ValueError("geometric median needs a finite nonempty 2-D matrix")
    current = np.median(values, axis=0)
    for _ in range(max_iterations):
        distances = np.linalg.norm(values - current[None, :], axis=1)
        weights = 1.0 / np.maximum(distances, epsilon)
        candidate = np.average(values, axis=0, weights=weights)
        if np.linalg.norm(candidate - current) <= tolerance * max(1.0, np.linalg.norm(current)):
            current = candidate
            break
        current = candidate
    if not np.isfinite(current).all():
        raise ValueError("geometric median became nonfinite")
    return current


def aggregate_residual(residuals, config: BARDConfig, *, aggregation=None):
    values = np.asarray(residuals, dtype=np.float64)
    if values.ndim != 2 or not values.shape[0]:
        raise ValueError("at least one isolated expert residual is required")
    method = config.aggregation if aggregation is None else aggregation
    if method == "mean":
        return values.mean(axis=0)
    if method == "coordinate_median":
        return np.median(values, axis=0)
    if method == "geometric_median":
        return geometric_median(
            values,
            max_iterations=config.median_iterations,
            tolerance=config.median_tolerance,
        )
    raise ValueError("unsupported BARD aggregation")


def _decision_from_residuals(
    base_logp,
    mask,
    residuals,
    config: BARDConfig,
    *,
    aggregation=None,
    bounded_commit=True,
):
    valid_base = np.asarray(base_logp, dtype=np.float64)[mask]
    residuals = np.asarray(residuals, dtype=np.float64)
    n = residuals.shape[0]
    valid_ids = np.flatnonzero(mask)
    base_index = int(np.argmax(valid_base))
    base_token = int(valid_ids[base_index])
    if n == 0:
        return base_token, {
            "reason": "no_expert_branch",
            "committed": False,
            "base_token": base_token,
            "candidate_token": base_token,
            "selected_token": base_token,
            "expert_count": 0,
            "medical_correctness_guaranteed": False,
        }

    aggregate = aggregate_residual(residuals, config, aggregation=aggregation)
    combined = valid_base + aggregate
    candidate_index = int(np.argmax(combined))
    candidate_token = int(valid_ids[candidate_index])

    # Per-node margin of candidate versus incumbent after adding only that
    # node's isolated receiver residual.
    base_margin = float(valid_base[candidate_index] - valid_base[base_index])
    branch_margins = (
        base_margin
        + residuals[:, candidate_index]
        - residuals[:, base_index]
    )
    aggregate_margin = float(combined[candidate_index] - combined[base_index])

    declared_f = config.fault_budget
    # This is a centralized robust-aggregation problem, not distributed
    # Byzantine agreement.  The usable fault budget is therefore bounded by
    # f < n/2 rather than a hard 3f+1 communication requirement.
    effective_f = min(declared_f, max(0, (n - 1) // 2))
    required_experts = 2 * effective_f + 1 if effective_f else 1
    support_needed = n if effective_f == 0 else n - effective_f
    supporters = int(np.sum(branch_margins > 0.0))
    conservative_margin = (
        float(np.sort(branch_margins)[effective_f])
        if n > effective_f
        else float("-inf")
    )

    if candidate_token == base_token:
        committed, reason, mode = False, "aggregate_agrees_with_base", "anchor"
    elif not bounded_commit:
        committed, reason, mode = True, "unprotected_aggregate", "ablation"
    elif n == 1:
        # One fallible expert and one incumbent are observationally
        # non-identifiable without an additional falsification observation.
        committed, reason, mode = False, "single_expert_unidentifiable", "single"
    elif n == 2:
        # One arbitrary node cannot force a change: both isolated receiver
        # branches must independently move the same candidate over the anchor.
        committed = aggregate_margin > 0.0 and supporters == 2 and conservative_margin > 0.0
        reason = "unanimous_nonforcing_consensus" if committed else "pair_disagreement_or_weak_support"
        mode = "pair_unanimous"
    elif n < required_experts:
        committed, reason, mode = False, "insufficient_robust_majority", "robust"
    elif aggregate_margin <= 0.0 or supporters < support_needed or conservative_margin <= 0.0:
        committed, reason, mode = False, "insufficient_branch_consensus", "robust"
    else:
        committed, reason, mode = True, "centralized_bounded_fault_consensus", "robust"

    selected = candidate_token if committed else base_token
    return selected, {
        "reason": reason,
        "committed": bool(committed),
        "base_token": base_token,
        "candidate_token": candidate_token,
        "selected_token": int(selected),
        "expert_count": int(n),
        "fault_budget": int(declared_f),
        "effective_fault_budget": int(effective_f),
        "required_experts": int(required_experts),
        "consensus_mode": mode,
        "support_needed": int(support_needed),
        "supporters": supporters,
        "base_margin": base_margin,
        "aggregate_margin": aggregate_margin,
        "conservative_margin": conservative_margin,
        "aggregation": config.aggregation if aggregation is None else aggregation,
        "aggregate_residual_norm": float(np.linalg.norm(aggregate)),
        "branch_margins": [float(v) for v in branch_margins],
        "medical_correctness_guaranteed": False,
        "interpretation": "bounded-fault receiver consensus, not expert confidence or clinical truth",
    }


def bard_step(
    base_scores,
    expert_scores,
    config: BARDConfig,
    *,
    aggregation=None,
    bounded_commit=True,
):
    base, mask, residuals = _prepare(base_scores, expert_scores)
    return _decision_from_residuals(
        base,
        mask,
        residuals,
        config,
        aggregation=aggregation,
        bounded_commit=bounded_commit,
    )


def single_fault_probe(base_scores, expert_scores, config: BARDConfig):
    """Label-free one-node stress test over already-computed residuals.

    One node at a time is replaced by a strong sign-reversed residual.  This
    costs zero additional model forwards and is diagnostics only.
    """
    base, mask, residuals = _prepare(base_scores, expert_scores)
    clean_token, clean = _decision_from_residuals(base, mask, residuals, config)
    probes = []
    for index in range(residuals.shape[0]):
        corrupted = residuals.copy()
        row = corrupted[index]
        if np.linalg.norm(row) > 0:
            corrupted[index] = -config.fault_probe_scale * row
        else:
            valid_base = base[mask]
            order = np.argsort(valid_base)
            source = int(order[-1])
            target = int(order[-2]) if len(order) > 1 else source
            magnitude = (
                config.fault_probe_scale
                + abs(float(valid_base[source] - valid_base[target]))
            )
            corrupted[index, target] = magnitude
            corrupted[index, source] = -magnitude
        token, audit = _decision_from_residuals(base, mask, corrupted, config)
        probes.append(
            {
                "expert_index": index,
                "selected_token": int(token),
                "same_as_clean": bool(token == clean_token),
                "fallback_to_base": bool(token == clean["base_token"]),
                "reason": audit["reason"],
                "committed": audit["committed"],
            }
        )
    return {
        "clean_token": int(clean_token),
        "clean": clean,
        "single_faults": probes,
        "extra_model_forwards": 0,
        "fault_model": "one sign-reversed arbitrary receiver residual",
    }




def decode_bard_bundle(
    base_session,
    expert_sessions: Mapping[str, object],
    *,
    max_tokens: int,
    config: BARDConfig,
):
    """Run matched mean/geomedian/BARD arms while sharing identical-prefix scores.

    This is an experiment-efficiency optimization only.  Each arm keeps its own
    committed prefix.  Scores are reused only when two arms have exactly the
    same token prefix, so the mathematical definition of each arm is unchanged.
    """
    if type(max_tokens) is not int or max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    names = list(expert_sessions)
    if len(names) != len(set(names)):
        raise ValueError("expert branch names must be unique")
    policies = {
        "isolated_mean": {"aggregation": "mean", "bounded": False},
        "isolated_geomedian": {"aggregation": "geometric_median", "bounded": False},
        "bard": {"aggregation": "geometric_median", "bounded": True},
    }
    states = {
        name: {"prefix": [], "trace": [], "finished": False}
        for name in policies
    }
    score_calls = 0

    for _step in range(max_tokens):
        active = [name for name, state in states.items() if not state["finished"]]
        if not active:
            break
        grouped = {}
        for name in active:
            grouped.setdefault(tuple(states[name]["prefix"]), []).append(name)
        for prefix_tuple, members in grouped.items():
            prefix = list(prefix_tuple)
            base_scores = base_session.next_scores(prefix)
            need_experts = any(
                not (
                    policies[name]["bounded"]
                    and len(names) == 1
                    and config.single_expert_policy == "baseline"
                )
                for name in members
            )
            expert_scores = (
                [expert_sessions[name].next_scores(prefix) for name in names]
                if need_experts
                else []
            )
            score_calls += 1 + len(expert_scores)
            for method in members:
                bounded = policies[method]["bounded"]
                if bounded and len(names) == 1 and not expert_scores:
                    base_logp, _ = _log_probs(base_scores)
                    token = int(np.argmax(base_logp))
                    audit = {
                        "reason": "single_expert_unidentifiable",
                        "committed": False,
                        "base_token": token,
                        "candidate_token": token,
                        "selected_token": token,
                        "expert_count": 1,
                        "fault_budget": config.fault_budget,
                        "effective_fault_budget": 0,
                        "required_experts": 2,
                        "consensus_mode": "single",
                        "medical_correctness_guaranteed": False,
                    }
                else:
                    token, audit = bard_step(
                        base_scores,
                        expert_scores,
                        config,
                        aggregation=policies[method]["aggregation"],
                        bounded_commit=bounded,
                    )
                    if method == "bard" and expert_scores:
                        audit["fault_probe"] = single_fault_probe(
                            base_scores, expert_scores, config
                        )
                audit["step"] = len(states[method]["prefix"])
                audit["experts"] = names
                states[method]["trace"].append(audit)
                states[method]["prefix"].append(int(token))
                if token in base_session.eos_ids or len(states[method]["prefix"]) >= max_tokens:
                    states[method]["finished"] = True

    outputs = {}
    for method, state in states.items():
        outputs[method] = {
            "text": base_session.decode(state["prefix"]).strip(),
            "token_ids": list(state["prefix"]),
            "trace": state["trace"],
            "expert_branches": names,
            "structural_fallback": bool(
                method == "bard"
                and len(names) == 1
                and config.single_expert_policy == "baseline"
            ),
            "fault_budget": config.fault_budget,
            "aggregation": policies[method]["aggregation"],
            "bounded_commit": policies[method]["bounded"],
            "shared_score_calls_bundle": score_calls,
        }
    return outputs



def decode_bard_incremental(
    base_stream,
    expert_streams: Mapping[str, object],
    *,
    max_tokens: int,
    config: BARDConfig,
    fault_probe=False,
):
    """Persistent-KV BARD for one committed trajectory.

    This fast path is intentionally separate from the replay reference decoder.
    Callers must establish real-model parity before using it for benchmark
    generation.
    """
    if type(max_tokens) is not int or max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    names = list(expert_streams)
    prefix, trace = [], []
    structural_single = len(names) == 1 and config.single_expert_policy == "baseline"

    for step in range(max_tokens):
        base_scores = base_stream.current_scores()
        if structural_single:
            base_logp, _ = _log_probs(base_scores)
            token = int(np.argmax(base_logp))
            expert_scores = []
            audit = {
                "reason": "single_expert_unidentifiable",
                "committed": False,
                "base_token": token,
                "candidate_token": token,
                "selected_token": token,
                "expert_count": 1,
                "fault_budget": config.fault_budget,
                "effective_fault_budget": 0,
                "required_experts": 2,
                "consensus_mode": "single",
                "medical_correctness_guaranteed": False,
            }
        else:
            expert_scores = [
                expert_streams[name].current_scores() for name in names
            ]
            token, audit = bard_step(
                base_scores,
                expert_scores,
                config,
                aggregation=config.aggregation,
                bounded_commit=True,
            )
        audit["step"] = step
        audit["experts"] = names
        audit["receiver_backend"] = "persistent_kv"
        if fault_probe and expert_scores:
            audit["fault_probe"] = single_fault_probe(
                base_scores, expert_scores, config
            )
        trace.append(audit)
        prefix.append(int(token))
        if token in base_stream.eos_ids:
            break
        base_stream.commit(token)
        for stream in expert_streams.values():
            stream.commit(token)

    return {
        "text": base_stream.decode(prefix).strip(),
        "token_ids": prefix,
        "trace": trace,
        "expert_branches": names,
        "structural_fallback": structural_single,
        "fault_budget": config.fault_budget,
        "aggregation": config.aggregation,
        "bounded_commit": True,
        "receiver_backend": "persistent_kv",
    }

def decode_bard(
    base_session,
    expert_sessions: Mapping[str, object],
    *,
    max_tokens: int,
    config: BARDConfig,
    aggregation=None,
    bounded_commit=True,
    fault_probe=False,
):
    """Decode with exact shared committed tokens across isolated receiver branches."""
    if type(max_tokens) is not int or max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    names = list(expert_sessions)
    if len(names) != len(set(names)):
        raise ValueError("expert branch names must be unique")
    prefix, trace = [], []

    # Only the truly unidentifiable single-expert case is structurally
    # short-circuited.  Two experts can use unanimous non-forcing consensus;
    # three or more use centralized robust aggregation.
    structural_fallback = bounded_commit and len(names) == 1
    for step in range(max_tokens):
        base_scores = base_session.next_scores(prefix)
        if structural_fallback:
            base_logp, _ = _log_probs(base_scores)
            token = int(np.argmax(base_logp))
            expert_scores = []
            audit = {
                "reason": "single_expert_unidentifiable",
                "committed": False,
                "base_token": token,
                "candidate_token": token,
                "selected_token": token,
                "expert_count": len(names),
                "fault_budget": config.fault_budget,
                "effective_fault_budget": 0,
                "required_experts": 2,
                "consensus_mode": "single",
                "medical_correctness_guaranteed": False,
                "interpretation": (
                    "single-expert structural fallback; an external falsification "
                    "observation is required before allowing this node to force a change"
                ),
            }
        else:
            expert_scores = [
                expert_sessions[name].next_scores(prefix) for name in names
            ]
            token, audit = bard_step(
                base_scores,
                expert_scores,
                config,
                aggregation=aggregation,
                bounded_commit=bounded_commit,
            )
        audit["step"] = step
        audit["experts"] = names
        if fault_probe and expert_scores:
            audit["fault_probe"] = single_fault_probe(
                base_scores, expert_scores, config
            )
        trace.append(audit)
        prefix.append(int(token))
        if token in base_session.eos_ids:
            break
    return {
        "text": base_session.decode(prefix).strip(),
        "token_ids": prefix,
        "trace": trace,
        "expert_branches": names,
        "structural_fallback": structural_fallback,
        "fault_budget": config.fault_budget,
        "aggregation": aggregation or config.aggregation,
        "bounded_commit": bounded_commit,
    }
