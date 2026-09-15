"""Research reference: uncertainty-discounted SAME-PREFIX dual decoding.

No training, calibration, model loading, medical validator, or repository changes.
ACD follows Kim et al., Findings EMNLP 2024, equations (2)-(3):
https://aclanthology.org/2024.findings-emnlp.136/

source_only and source_acd are UNVALIDATED heuristic ablations, not official ACD.
Input scores must be finite, full-vocabulary scores before truncation/constraints.
A source uncertainty must describe the exact admitted claim, not an arbitrary
confidence field. Missing source uncertainty is explicit and never treated as 0.

The session protocol matches MERIT's next_scores(prefix), decode(tokens), and
propose((), count=1, length=max_tokens) interfaces. This module was tested only
with synthetic logits/mock sessions, not medical weights or a live MERIT run.
"""
from __future__ import annotations

import math
from time import perf_counter
from typing import Any, Literal, Protocol, Sequence

import numpy as np

Mode = Literal['fixed', 'acd', 'source_only', 'source_acd']
MODES = {'fixed', 'acd', 'source_only', 'source_acd'}
SOURCE_MODES = {'source_only', 'source_acd'}


class Session(Protocol):
    def next_scores(self, prefix: tuple[int, ...]) -> np.ndarray: ...
    def decode(self, tokens: Sequence[int]) -> str: ...
    def propose(self, prefix: tuple[int, ...], *, count: int, length: int) -> Any: ...


def _unit_value(value: float, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f'{name} must be numeric, not boolean')
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f'{name} must be finite in [0, 1]')
    return result


def categorical_uncertainty(probabilities: Sequence[float]) -> float:
    """H(q)/log(K), for a declared, mutually exclusive native label space.

    The caller must not substitute arbitrary similarity scores or normalize
    independent sigmoid labels into a categorical distribution.
    """
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 1 or len(p) < 2 or not np.isfinite(p).all() or (p < 0).any():
        raise ValueError('expected >=2 finite nonnegative categorical probabilities')
    if not np.isclose(p.sum(), 1.0, rtol=1e-6, atol=1e-8):
        raise ValueError('native probabilities must sum to one')
    p = p / p.sum()  # Floating point roundoff only, not a temperature fit.
    positive = p > 0
    h = -float(np.dot(p[positive], np.log(p[positive])))
    return float(np.clip(h / np.log(len(p)), 0.0, 1.0))


def binary_uncertainty(probability: float) -> float:
    """Entropy of ONE relevant binary attribute, not a mean over all diseases."""
    p = _unit_value(probability, 'probability')
    return categorical_uncertainty([p, 1.0 - p])


def semantic_frequency_uncertainty(cluster_ids: Sequence[int]) -> float:
    """Empirical cluster entropy / log(M), M fixed in the experiment protocol.

    Cluster assignments must come from a frozen semantic comparison procedure.
    This function does not perform NLI or validate medical statements. log(M)
    is a proposed sample-size normalization, NOT the original SE algorithm's
    guaranteed population-entropy scale. M=1 and unknown cluster -1 are rejected.
    """
    ids = np.asarray(cluster_ids)
    if (ids.ndim != 1 or len(ids) < 2 or ids.dtype.kind not in 'iu'
            or (ids < 0).any()):
        raise ValueError('need >=2 known nonnegative integer cluster assignments')
    _, counts = np.unique(ids, return_counts=True)
    p = counts.astype(np.float64) / len(ids)
    return float(np.clip(-np.dot(p, np.log(p)) / np.log(len(ids)), 0.0, 1.0))


def distribution_stats(logits: Sequence[float]) -> tuple[np.ndarray, np.ndarray, float]:
    """Stable float64 reference; logarithms are natural and entropy is in nats."""
    z = np.asarray(logits, dtype=np.float64)
    if z.ndim != 1 or len(z) < 2 or not np.isfinite(z).all():
        raise ValueError('need a finite full-vocabulary vector of length >=2')
    with np.errstate(over='raise', invalid='raise'):
        shifted = z - z.max()
    logp = shifted - np.log(np.exp(shifted).sum())
    p = np.exp(logp)
    positive = p > 0  # Avoid 0 * (-inf) for underflowed probabilities.
    h = max(0.0, -float(np.dot(p[positive], logp[positive])))
    return z, logp, h


def _validate_policy(mode: str, applicable: bool, source_u: float | None,
                     fixed_weight: float) -> float | None:
    if mode not in MODES or type(applicable) is not bool:
        raise ValueError('unknown mode or nonboolean applicability')
    if not math.isfinite(fixed_weight) or fixed_weight < 0:
        raise ValueError('fixed weight must be finite and nonnegative')
    return None if source_u is None else _unit_value(source_u, 'source_u')


def adaptive_step(base_logits: Sequence[float], evidence_logits: Sequence[float], *,
                  mode: Mode = 'source_acd', source_u: float | None = None,
                  applicable: bool = True, fixed_weight: float = 0.5
                  ) -> tuple[np.ndarray, dict[str, Any]]:
    """Return mixed SCORES and audit, not calibrated clinical probabilities.

    s = H0/(H0+HE)                        for acd
    s = 1-source_u                       for source_only
    s = (1-source_u) * H0/(H0+HE)         for source_acd
    z = (1-s)*z0 + s*zE

    Applicability=false or a missing source_u in source modes yields s=0.
    Entropy's zero/zero limit uses 0.5, an explicit numeric convention.
    fixed_weight=1.5 supplies a fixed CAD alpha=0.5 score baseline;
    adaptive modes stay in [0,1] and do NOT amplify beyond the evidence branch.
    """
    source_u = _validate_policy(mode, applicable, source_u, fixed_weight)
    z0, _, h0 = distribution_stats(base_logits)
    ze, _, he = distribution_stats(evidence_logits)
    if z0.shape != ze.shape:
        raise ValueError('both branches must have the exact same vocabulary')
    denominator = h0 + he
    rho = h0 / denominator if denominator > np.finfo(np.float64).eps else 0.5
    if not applicable:
        weight, reason = 0.0, 'not_applicable'
    elif mode in SOURCE_MODES and source_u is None:
        weight, reason = 0.0, 'source_uncertainty_missing'
    elif mode == 'fixed':
        weight, reason = float(fixed_weight), 'fixed'
    elif mode == 'acd':
        weight, reason = rho, 'acd_entropy_ratio'
    elif mode == 'source_only':
        weight, reason = 1.0 - source_u, 'native_uncertainty_discount'
    else:
        weight = (1.0 - source_u) * rho
        reason = 'native_discount_times_acd'
    # Preserve exact endpoint scores, including tie-breaking; no clipping/filter.
    mixed = z0.copy() if weight == 0 else ze.copy() if weight == 1 else (
        (1.0 - weight) * z0 + weight * ze)
    if not np.isfinite(mixed).all():
        raise FloatingPointError('nonfinite fused scores')
    return mixed, {
        'base_entropy_nats': h0, 'evidence_entropy_nats': he,
        'acd_weight': float(rho), 'source_uncertainty': source_u,
        'effective_weight': float(weight), 'reason': reason,
        'mode': mode, 'calibrated_correctness_probability': False,
    }


def decode_adaptive(base: Session, conditioned: Session | None, *,
                    max_tokens: int, eos_ids: set[int], mode: Mode = 'source_acd',
                    source_u: float | None = None, applicable: bool = True,
                    fixed_weight: float = 0.5) -> dict[str, Any]:
    """Drop-in REFERENCE loop using existing sessions, no new model instances.

    Each experiment arm needs fresh branch-local KV state. Both branches see
    exactly the same already-committed token IDs. Greedy-only; upstream prompts,
    EOS conventions, image preprocessing and generation limits must be frozen.
    """
    if type(max_tokens) is not int or max_tokens < 1:
        raise ValueError('max_tokens must be a positive integer')
    if not isinstance(eos_ids, set) or not eos_ids or any(
            type(t) is not int or t < 0 for t in eos_ids):
        raise ValueError('eos_ids must be a nonempty set of nonnegative integers')
    source_u = _validate_policy(mode, applicable, source_u, fixed_weight)
    start = perf_counter()
    static_weight = None
    reason = None
    if not applicable:
        static_weight, reason = 0.0, 'not_applicable'
    elif mode in SOURCE_MODES and source_u is None:
        static_weight, reason = 0.0, 'source_uncertainty_missing'
    elif mode == 'fixed':
        static_weight, reason = fixed_weight, 'fixed'
    elif mode == 'source_only':
        static_weight, reason = 1.0 - source_u, 'native_uncertainty_discount'
    elif mode == 'source_acd' and source_u == 1.0:
        static_weight, reason = 0.0, 'maximal_source_uncertainty'
    if static_weight in (0.0, 1.0):
        session = base if static_weight == 0 else conditioned
        if session is None:
            raise ValueError('evidence branch required for weight=1')
        block = session.propose((), count=1, length=max_tokens)[0]
        return {
            'text': block.text, 'token_ids': list(block.tokens),
            'steps': [], 'endpoint_weight': static_weight, 'reason': reason,
            'score_calls': 0, 'endpoint_generation_calls': 1,
            'seconds': perf_counter() - start, 'clinical_verification': False,
        }
    if conditioned is None:
        raise ValueError('an evidence session is required')
    prefix: list[int] = []
    steps = []
    for _ in range(max_tokens):
        a = base.next_scores(tuple(prefix))
        b = conditioned.next_scores(tuple(prefix))
        scores, audit = adaptive_step(a, b, mode=mode, source_u=source_u,
                                     applicable=applicable, fixed_weight=fixed_weight)
        token = int(np.argmax(scores))
        steps.append({**audit, 'base_top': int(np.argmax(a)),
                      'evidence_top': int(np.argmax(b)), 'selected': token})
        prefix.append(token)
        if token in eos_ids:
            break
    return {'text': base.decode(prefix).strip(), 'token_ids': prefix, 'steps': steps,
            'score_calls': 2 * len(steps), 'endpoint_generation_calls': 0,
            'seconds': perf_counter() - start, 'clinical_verification': False}
