"""Continuous, source-only marginal tool value with empirical shift pessimism.

The caller supplies pre-action state x descriptor features and paired continuations
from an identical prefix. Neither the future tool result nor reference answers may
be features. Empty executed observations remain training outcomes. Independent RGB
image/patient groups, not states or question IDs, determine weights and support.

This is a lightweight empirical baseline, NOT a conformal confidence interval,
clinical safety guarantee, or guarantee for arbitrary unseen domains. A singleton
or one-history estimate does not certify an unobserved multi-tool composition.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from numbers import Real

import numpy as np

_VERSION = 1
_KINDS = {"initial", "continuation", "after_tool"}


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f"{name} must be finite numeric data")
    return float(value)


def _features(values: object, dimension: int | None = None) -> list[float]:
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError("features must be a nonempty finite list")
    result = [_number(value, "feature") for value in values]
    if dimension is not None and len(result) != dimension:
        raise ValueError("feature dimension differs from source-fitted policy")
    return result


def _condition(row: dict) -> tuple[str, str, tuple[str, ...]]:
    action = _text(row.get("action_key"), "action_key")
    kind = row.get("state_kind", "initial")
    if kind not in _KINDS:
        raise ValueError("unknown state_kind")
    history = row.get("history_actions", [])
    if not isinstance(history, list):
        raise TypeError("history_actions must be an ordered list")
    history = tuple(_text(item, "history action") for item in history)
    if kind == "initial" and history:
        raise ValueError("initial state cannot have tool history")
    if kind == "after_tool" and not history:
        raise ValueError("after_tool state requires tool history")
    return action, kind, history


def _condition_key(condition: tuple[str, str, tuple[str, ...]]) -> str:
    # JSON encoding is unambiguous even if a tool ID itself contains a separator.
    return json.dumps(condition, ensure_ascii=False, separators=(",", ":"))


def _validate_records(records: list[dict]) -> tuple[list[dict], int, int]:
    normalized, seen, samples, groups = [], {}, {}, {}
    dimension, duplicates = 0, 0
    for original in records:
        if original.get("role") != "source":
            raise ValueError("only source records may fit a value policy")
        row = {key: _text(original.get(key), key) for key in (
            "sample_id", "group_id", "domain", "state_id"
        )}
        action, kind, history = _condition(original)
        row.update(action_key=action, state_kind=kind, history_actions=list(history))
        row["features"] = _features(original.get("features"), dimension or None)
        dimension = len(row["features"])
        row["gain"] = _number(original.get("gain"), "gain")
        row["cost"] = _number(original.get("cost"), "cost")
        if not -1 <= row["gain"] <= 1 or row["cost"] < 0:
            raise ValueError("gain must be in [-1, 1] and cost must be nonnegative")
        for flag in ("executed", "adopted"):
            if not isinstance(original.get(flag), bool):
                raise TypeError(f"{flag} must be bool")
            row[flag] = original[flag]
        if row["adopted"] and not row["executed"]:
            raise ValueError("unexecuted tool cannot have adopted evidence")
        identity = (row["domain"], row["group_id"])
        if row["sample_id"] in samples and samples[row["sample_id"]] != identity:
            raise ValueError("sample_id appears in inconsistent source domains/groups")
        if row["group_id"] in groups and groups[row["group_id"]] != row["domain"]:
            raise ValueError("independent group crosses source domains")
        samples[row["sample_id"]], groups[row["group_id"]] = identity, row["domain"]
        key = (row["domain"], row["sample_id"], row["state_id"], action)
        if key in seen:
            if seen[key] != row:
                raise ValueError("conflicting duplicate source state/action record")
            duplicates += 1
            continue
        seen[key] = row
        normalized.append(row)
    return normalized, dimension, duplicates


def _weights(rows: list[dict]) -> np.ndarray:
    """Equal mass per domain and per independent image group within domain."""
    group_sizes = Counter((row["domain"], row["group_id"]) for row in rows)
    domain_sizes = Counter(domain for domain, _ in group_sizes)
    total_groups = len(group_sizes)
    return np.asarray([
        total_groups / (len(domain_sizes) * domain_sizes[row["domain"]]
                        * group_sizes[(row["domain"], row["group_id"])])
        for row in rows
    ])


def _fit_ridge(rows: list[dict], ridge: float) -> dict | None:
    if not rows:
        return None
    x = np.asarray([row["features"] for row in rows], dtype=float)
    y = np.asarray([[row["gain"], row["cost"]] for row in rows], dtype=float)
    weight = _weights(rows)
    offset = np.average(x, axis=0, weights=weight)
    scale = np.sqrt(np.average((x - offset) ** 2, axis=0, weights=weight))
    scale[scale < 1e-8] = 1.0
    design = np.column_stack((np.ones(len(rows)), (x - offset) / scale))
    regularizer = np.eye(design.shape[1]) * ridge
    regularizer[0, 0] = 0.0
    matrix = design.T @ (weight[:, None] * design) + regularizer
    rhs = design.T @ (weight[:, None] * y)
    coefficients = np.linalg.solve(matrix, rhs)
    result = {
        "offset": offset.tolist(), "scale": scale.tolist(),
        "coefficients": coefficients.tolist(),
        "training_domains": sorted({row["domain"] for row in rows}),
        "independent_groups": len({(row["domain"], row["group_id"]) for row in rows}),
    }
    json.dumps(result, allow_nan=False)
    return result


def _predict(model: dict, features: list[float]) -> tuple[float, float]:
    x = np.asarray(features, dtype=float)
    offset, scale = np.asarray(model["offset"]), np.asarray(model["scale"])
    coefficients = np.asarray(model["coefficients"])
    if (offset.shape != x.shape or scale.shape != x.shape
            or coefficients.shape != (len(x) + 1, 2)
            or not np.all(np.isfinite(offset)) or not np.all(np.isfinite(scale))
            or not np.all(np.isfinite(coefficients)) or np.any(scale <= 0)):
        raise ValueError("invalid source-fitted model dimensions/parameters")
    with np.errstate(over="ignore", invalid="ignore"):
        gain, cost = np.concatenate(([1.0], (x - offset) / scale)) @ coefficients
    if not np.isfinite(gain) or not np.isfinite(cost):
        raise ValueError("nonfinite prediction; check source/target feature scale")
    return float(np.clip(gain, -1, 1)), float(max(0.0, cost))


def fit_value_policy(
    records: list[dict], provenance: dict, *, ridge: float = 1.0,
    min_cases_per_domain: int = 8, min_domains: int = 2,
    residual_quantile: float = 0.9, cost_weight: float = 0.0,
) -> dict:
    """Fit a shared ridge gain/cost head and source-LODO overestimation penalty.

    Gain is a continuous paired quality difference, not a binary error label.
    All *executed* records fit the mean, including empty/non-adopted outcomes.
    Support additionally requires nonempty adopted observations in independent
    groups for this action/state-kind/ordered-history condition. Runtime failures
    (executed=False) are counted separately, never relabelled as zero gain.

    Residuals use one source domain held out of BOTH normalization and regression.
    The largest within-group overestimate is one calibration observation; the
    penalty is the worst-domain empirical upper quantile, clipped below at zero.
    It is NOT a calibrated confidence bound for the refitted head or a new target.
    """
    ridge = _number(ridge, "ridge")
    residual_quantile = _number(residual_quantile, "residual_quantile")
    cost_weight = _number(cost_weight, "cost_weight")
    if ridge <= 0 or not 0.5 <= residual_quantile <= 1 or cost_weight < 0:
        raise ValueError("need positive ridge, quantile in [0.5, 1], nonnegative cost_weight")
    if (type(min_cases_per_domain) is not int or min_cases_per_domain < 1
            or type(min_domains) is not int or min_domains < 2):
        raise ValueError("support requires >=1 independent group and >=2 source domains")
    if not isinstance(provenance, dict):
        raise TypeError("provenance must be a JSON object")
    provenance = json.loads(json.dumps(provenance, allow_nan=False))
    rows, dimension, duplicates = _validate_records(records)
    executed = [row for row in rows if row["executed"]]
    domains = sorted({row["domain"] for row in rows})
    mean_model = _fit_ridge(executed, ridge)
    folds, residuals = {}, defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for held_out in domains:
        train = [row for row in executed if row["domain"] != held_out]
        model = _fit_ridge(train, ridge)
        if model is None:
            continue
        folds[held_out] = model
        for row in executed:
            if row["domain"] != held_out:
                continue
            predicted_gain, predicted_cost = _predict(model, row["features"])
            overestimate = ((predicted_gain - cost_weight * predicted_cost)
                            - (row["gain"] - cost_weight * row["cost"]))
            key = _condition_key(_condition(row))
            residuals[key][held_out][row["group_id"]].append(overestimate)

    conditions = defaultdict(list)
    for row in rows:
        conditions[_condition_key(_condition(row))].append(row)
    cards = {}
    for key, condition_rows in sorted(conditions.items()):
        card_domains = {}
        for domain in sorted({row["domain"] for row in condition_rows}):
            subset = [row for row in condition_rows if row["domain"] == domain]
            values = [max(gains) for gains in residuals[key][domain].values()]
            penalty = float(max(0.0, np.quantile(
                values, residual_quantile, method="higher"
            ))) if values else None
            card_domains[domain] = {
                "observed_groups": len({row["group_id"] for row in subset}),
                "executed_groups": len({row["group_id"] for row in subset if row["executed"]}),
                "adopted_groups": len({row["group_id"] for row in subset if row["adopted"]}),
                "rows": len(subset),
                "runtime_failures": sum(not row["executed"] for row in subset),
                "empty_executions": sum(row["executed"] and not row["adopted"] for row in subset),
                "residual_groups": len(values), "overestimate_penalty": penalty,
            }
        support = len(card_domains) >= min_domains and all(
            domain["adopted_groups"] >= min_cases_per_domain
            for domain in card_domains.values()
        )
        calibrated = len(card_domains) >= min_domains and all(
            domain["residual_groups"] >= min_cases_per_domain
            and domain["overestimate_penalty"] is not None for domain in card_domains.values()
        )
        action, kind, history = _condition(condition_rows[0])
        cards[key] = {
            "action_key": action, "state_kind": kind, "history_actions": list(history),
            "support_sufficient": support, "residual_calibrated": calibrated,
            "domains": card_domains,
            "penalty": max((domain["overestimate_penalty"] or 0.0
                            for domain in card_domains.values()), default=0.0),
        }
    artifact = {
        "schema_version": _VERSION, "estimator": "group_balanced_ridge_source_lodo",
        "uncertainty": "empirical_worst_source_overestimation_not_confidence_bound",
        "guarantee": "none_for_arbitrary_target_shift_or_unseen_composition",
        "support_unit": "domain_and_independent_group_id",
        "history_policy": "exact_ordered_history_and_state_kind",
        "target_labels_used": False, "provenance": provenance, "feature_dim": dimension,
        "parameters": {"ridge": ridge, "min_cases_per_domain": min_cases_per_domain,
                       "min_domains": min_domains, "residual_quantile": residual_quantile,
                       "cost_weight": cost_weight},
        "mean_model": mean_model, "lodo_models": folds, "conditions": cards,
        "source_summary": {
            "domains": domains, "records": len(rows), "duplicates_removed": duplicates,
            "executed_records": len(executed),
            "runtime_failures": len(rows) - len(executed),
            "empty_executions": sum(not row["adopted"] for row in executed),
            "independent_groups": len({(row["domain"], row["group_id"]) for row in rows}),
            "sample_ids": sorted({row["sample_id"] for row in rows}),
        },
    }
    json.dumps(artifact, allow_nan=False)
    return artifact


def score_value_policy(artifact: dict, candidates: list[dict], *, robust: bool = True) -> list[dict]:
    """Read-only inference: score continuous gain minus cost and optional penalty.

    NONE has reference value zero. The caller must require supported=True and a
    strictly positive score before selecting a tool. Unsupported actions return
    score=0 (never +/-infinity) plus an explicit reason. Mean ablation removes
    pessimism only; it does not bypass missing image/history support.
    """
    if artifact.get("schema_version") != _VERSION or artifact.get("target_labels_used") is not False:
        raise ValueError("not a recognized source-only value artifact")
    dimension = artifact.get("feature_dim")
    if type(dimension) is not int or dimension < 0:
        raise ValueError("invalid feature dimension in policy")
    results = []
    for candidate in candidates:
        if any(key in candidate for key in ("gain", "reference", "references", "target_label")):
            raise ValueError("prediction candidates must not contain outcomes or references")
        condition = _condition(candidate)
        features = _features(candidate.get("features"), dimension or None)
        action, kind, history = condition
        result = {"action_key": action, "state_kind": kind, "history_actions": list(history),
                  "supported": False, "score": 0.0, "mean_gain": None,
                  "predicted_cost": None, "mean_net_gain": None, "penalty": None}
        card = artifact.get("conditions", {}).get(_condition_key(condition))
        if card is None:
            known_actions = {item["action_key"] for item in artifact.get("conditions", {}).values()}
            reason = "unseen_state_history" if action in known_actions else "unknown_action"
        elif not card["support_sufficient"]:
            reason = "insufficient_independent_support"
        elif robust and not card["residual_calibrated"]:
            reason = "insufficient_held_source_residuals"
        elif artifact.get("mean_model") is None:
            reason = "no_executed_source_data"
        else:
            gain, cost = _predict(artifact["mean_model"], features)
            weight = _number(artifact["parameters"]["cost_weight"], "cost_weight")
            penalty = _number(card["penalty"], "penalty") if robust else 0.0
            net = gain - weight * cost
            score = net - penalty
            if not np.isfinite(score) or weight < 0 or penalty < 0:
                raise ValueError("invalid policy cost/penalty")
            reason = "positive_utility" if score > 0 else "nonpositive_utility"
            result.update(supported=True, score=score, mean_gain=gain, predicted_cost=cost,
                          mean_net_gain=net, penalty=penalty)
        result["reason"] = reason
        results.append(result)
    return results
