"""Source-only expert portfolio selection for MERIT-Tx.

This module addresses a failure mode that per-expert qualification cannot solve:
two specialists can each look acceptable in isolation while their joint use is
redundant or harmful.  Portfolio selection therefore evaluates the *set* of
qualified verifier fault groups on frozen source/development transactions.

The empty portfolio is always a valid candidate.  If no non-empty set has
positive conservative source utility under the frozen harm constraints, MERIT
keeps the immutable Generalist rather than adding experts for coverage alone.

No target/test label is accepted by this module.
"""
from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from .expert_policy import qualification_for, role_card


@dataclass(frozen=True)
class ExpertPortfolioCard:
    modality: str
    task: str
    claim_type: str
    expert_ids: tuple[str, ...]
    fault_groups: tuple[str, ...]
    n_groups: int
    domains: tuple[str, ...]
    action_domains: tuple[str, ...]
    commit_n: int
    helpful_n: int
    harmful_n: int
    neutral_n: int
    utility_mean: float
    utility_lcb: float
    action_utility_mean: float
    action_utility_lcb: float
    harm_ucb: float
    precision_lcb: float
    coverage_lcb: float
    estimated_cost_units: float
    feasible: bool
    reason: str
    source_only: bool = True

    def __post_init__(self):
        if not self.source_only:
            raise ValueError("target/test outcomes cannot define an expert portfolio")
        if not self.modality or not self.task or not self.claim_type:
            raise ValueError("portfolio identity cannot be empty")
        if self.n_groups < 1:
            raise ValueError("portfolio needs source groups")
        if len(self.expert_ids) != len(set(self.expert_ids)):
            raise ValueError("portfolio expert IDs must be unique")
        if len(self.fault_groups) != len(set(self.fault_groups)):
            raise ValueError("portfolio fault groups must be unique")
        if self.commit_n < 0 or self.helpful_n < 0 or self.harmful_n < 0 or self.neutral_n < 0:
            raise ValueError("portfolio counts must be nonnegative")
        for value in (
            self.utility_mean,
            self.utility_lcb,
            self.action_utility_mean,
            self.action_utility_lcb,
            self.harm_ucb,
            self.precision_lcb,
            self.coverage_lcb,
            self.estimated_cost_units,
        ):
            if not math.isfinite(value):
                raise ValueError("portfolio statistics must be finite")

    @property
    def key(self):
        return self.modality, self.task, self.claim_type


def _wilson(successes, n, z, *, upper):
    if n < 1:
        return 1.0 if upper else 0.0
    p = successes / n
    z2 = z * z
    center = (p + z2 / (2 * n)) / (1 + z2 / n)
    radius = (
        z
        * math.sqrt((p * (1 - p) + z2 / (4 * n)) / n)
        / (1 + z2 / n)
    )
    return min(1.0, center + radius) if upper else max(0.0, center - radius)


def _mean_lcb(values, z):
    values = tuple(float(value) for value in values)
    if not values:
        return -1.0
    center = mean(values)
    if len(values) == 1:
        return -1.0
    variance = sum((value - center) ** 2 for value in values) / (len(values) - 1)
    return max(-1.0, center - z * math.sqrt(variance / len(values)))


def _bucket(row):
    return (
        str(row["modality"]),
        str(row["task"]),
        str(row.get("claim_type") or "*"),
    )


def _proposer_fault_groups(row, specs):
    groups = set()
    values = row.get("proposer_expert_ids") or ()
    if isinstance(values, str):
        values = (values,)
    single = row.get("proposer_expert_id")
    if single:
        values = tuple(values) + (single,)
    for expert_id in values:
        if expert_id in specs:
            groups.add(role_card(expert_id, specs[expert_id]).fault_group)
    return groups


def _qualified_action(
    row,
    *,
    specs,
    qualification_cards,
    policy,
):
    expert_id = str(row["expert_id"])
    if expert_id not in specs:
        return None
    descriptor_scope = str(row["scope"])
    qcard = qualification_for(
        qualification_cards,
        expert_id=expert_id,
        capability=str(row["capability"]),
        scope=descriptor_scope,
        modality=str(row["modality"]),
        task=str(row["task"]),
        claim_type=str(row.get("claim_type") or "*"),
    )
    if qcard is None:
        return None
    effect = float(row["differential_effect"])
    card = role_card(expert_id, specs[expert_id])
    if not card.patient_specific or card.commit_authority != "source_qualified":
        return None
    if effect > 0 and qcard.authorizes_commit(
        min_domains=int(policy["qualification_min_domains"]),
        max_harm_ucb=float(policy["qualification_max_harm_ucb"]),
        min_support_precision_lcb=float(
            policy["qualification_min_support_precision_lcb"]
        ),
        min_consequential=int(policy["qualification_min_consequential"]),
    ):
        return "support"
    if effect < 0 and qcard.authorizes_veto(
        min_domains=int(policy["qualification_min_domains"]),
        min_veto_precision_lcb=float(
            policy["qualification_min_veto_precision_lcb"]
        ),
        min_consequential=int(policy["qualification_min_consequential"]),
    ):
        return "veto"
    return None


def _simulate_subset(
    rows,
    expert_ids,
    *,
    specs,
    qualification_cards,
    policy,
    z,
):
    selected = frozenset(expert_ids)
    by_transaction = {}
    group_domains = {}
    for row in rows:
        key = str(row["group_id"]), str(row["transaction_id"])
        by_transaction.setdefault(key, []).append(row)
        group_domains.setdefault(str(row["group_id"]), set()).add(str(row["domain"]))

    group_commits = {}
    for (group_id, _transaction_id), observations in by_transaction.items():
        proposer_groups = _proposer_fault_groups(observations[0], specs)
        support_groups = set()
        veto_groups = set()
        outcome = float(observations[0]["outcome_delta"])
        for row in observations:
            expert_id = str(row["expert_id"])
            if expert_id not in selected:
                continue
            action = _qualified_action(
                row,
                specs=specs,
                qualification_cards=qualification_cards,
                policy=policy,
            )
            if action is None:
                continue
            fault_group = role_card(expert_id, specs[expert_id]).fault_group
            if action == "support":
                if (
                    bool(policy["require_independent_validator"])
                    and fault_group in proposer_groups
                ):
                    continue
                support_groups.add(fault_group)
            elif action == "veto":
                veto_groups.add(fault_group)

        commit = (
            len(support_groups) >= int(policy["min_support_groups"])
            and not (
                bool(policy["reject_on_qualified_contradiction"])
                and veto_groups
            )
        )
        if commit:
            group_commits.setdefault(group_id, []).append(outcome)

    all_groups = sorted(group_domains)
    utilities = []
    action_utilities = []
    action_domains = set()
    for group_id in all_groups:
        committed = group_commits.get(group_id, ())
        # Report-generation patients can yield several claim transactions.  One
        # harmful committed edit is sufficient to count the patient conservatively.
        utility = min(committed) if committed else 0.0
        utilities.append(float(utility))
        if committed:
            action_utilities.append(float(utility))
            action_domains.update(group_domains[group_id])

    commit_n = len(action_utilities)
    helpful_n = sum(value > 0 for value in action_utilities)
    harmful_n = sum(value < 0 for value in action_utilities)
    neutral_n = sum(value == 0 for value in action_utilities)
    consequential = helpful_n + harmful_n
    fault_groups = tuple(
        sorted({role_card(expert_id, specs[expert_id]).fault_group for expert_id in selected})
    )
    cost_units = sum(
        float(specs[expert_id].get("portfolio_cost_units", 1.0))
        for expert_id in selected
    )
    domains = tuple(sorted({domain for values in group_domains.values() for domain in values}))
    action_utility_mean = mean(action_utilities) if action_utilities else 0.0
    harm_ucb = _wilson(harmful_n, commit_n, z, upper=True)
    precision_lcb = _wilson(helpful_n, consequential, z, upper=False)
    coverage_lcb = _wilson(commit_n, len(all_groups), z, upper=False)

    reason = "eligible"
    feasible = True
    if not selected:
        feasible, reason = False, "empty-incumbent-portfolio"
    elif commit_n < int(policy["portfolio_min_actions"]):
        feasible, reason = False, "insufficient-source-actions"
    elif len(action_domains) < int(policy["portfolio_min_domains"]):
        feasible, reason = False, "insufficient-action-domain-coverage"
    elif _mean_lcb(utilities, z) <= 0:
        feasible, reason = False, "nonpositive-population-utility-lcb"
    elif _mean_lcb(action_utilities, z) <= 0:
        feasible, reason = False, "nonpositive-action-utility-lcb"
    elif harm_ucb > float(policy["portfolio_max_harm_ucb"]):
        feasible, reason = False, "portfolio-harm-ucb-too-high"
    elif precision_lcb < float(policy["portfolio_min_precision_lcb"]):
        feasible, reason = False, "portfolio-precision-lcb-too-low"

    return ExpertPortfolioCard(
        modality=str(rows[0]["modality"]),
        task=str(rows[0]["task"]),
        claim_type=str(rows[0].get("claim_type") or "*"),
        expert_ids=tuple(sorted(selected)),
        fault_groups=fault_groups,
        n_groups=len(all_groups),
        domains=domains,
        action_domains=tuple(sorted(action_domains)),
        commit_n=commit_n,
        helpful_n=helpful_n,
        harmful_n=harmful_n,
        neutral_n=neutral_n,
        utility_mean=float(mean(utilities)),
        utility_lcb=float(_mean_lcb(utilities, z)),
        action_utility_mean=float(action_utility_mean),
        action_utility_lcb=float(_mean_lcb(action_utilities, z)),
        harm_ucb=float(harm_ucb),
        precision_lcb=float(precision_lcb),
        coverage_lcb=float(coverage_lcb),
        estimated_cost_units=float(cost_units),
        feasible=bool(feasible),
        reason=reason,
    )


def fit_expert_portfolios(
    rows,
    *,
    specs,
    qualification_cards,
    policy,
    z=1.96,
):
    """Fit one conservative source-only verifier portfolio per task cell.

    The selection objective is lexicographic rather than a target-tuned weighted
    sum: first require positive conservative utility and bounded harm, then prefer
    larger utility/coverage, lower cost, and fewer experts.
    """
    rows = tuple(rows)
    if not rows:
        raise ValueError("portfolio observations are empty")
    if not math.isfinite(z) or z <= 0:
        raise ValueError("z must be positive and finite")
    forbidden_splits = {"test", "target", "evaluation", "eval"}
    for row in rows:
        if str(row.get("split", "source")).casefold() in forbidden_splits:
            raise ValueError("target/test observations cannot fit an expert portfolio")
        for key in (
            "expert_id",
            "capability",
            "scope",
            "modality",
            "task",
            "domain",
            "group_id",
            "transaction_id",
            "outcome_delta",
            "differential_effect",
        ):
            if key not in row:
                raise ValueError(f"portfolio observation missing {key}")

    grouped = {}
    for row in rows:
        grouped.setdefault(_bucket(row), []).append(row)

    buckets = []
    for key in sorted(grouped):
        values = grouped[key]
        eligible = []
        for expert_id in sorted({str(row["expert_id"]) for row in values}):
            if expert_id not in specs:
                continue
            card = role_card(expert_id, specs[expert_id])
            if not card.patient_specific or card.commit_authority != "source_qualified":
                continue
            has_action = any(
                str(row["expert_id"]) == expert_id
                and _qualified_action(
                    row,
                    specs=specs,
                    qualification_cards=qualification_cards,
                    policy=policy,
                )
                is not None
                for row in values
            )
            if has_action:
                eligible.append(expert_id)

        max_size = min(int(policy["portfolio_max_experts"]), len(eligible))
        candidates = []
        by_ids = {}
        for size in range(1, max_size + 1):
            for subset in itertools.combinations(eligible, size):
                # Correlated tools do not become two portfolio votes.
                groups = [
                    role_card(expert_id, specs[expert_id]).fault_group
                    for expert_id in subset
                ]
                if len(groups) != len(set(groups)):
                    continue
                card = _simulate_subset(
                    values,
                    subset,
                    specs=specs,
                    qualification_cards=qualification_cards,
                    policy=policy,
                    z=z,
                )
                candidates.append(card)
                by_ids[card.expert_ids] = card

        feasible = [card for card in candidates if card.feasible]
        if feasible:
            feasible.sort(
                key=lambda card: (
                    card.utility_lcb,
                    card.action_utility_lcb,
                    card.coverage_lcb,
                    -card.harm_ucb,
                    -card.estimated_cost_units,
                    -len(card.expert_ids),
                    card.expert_ids,
                ),
                reverse=True,
            )
            selected = feasible[0]
            selection_reason = "best-feasible-source-portfolio"
        else:
            # Empty is intentionally a first-class outcome: source evidence did
            # not justify allowing any verifier set to alter the incumbent.
            template = _simulate_subset(
                values,
                (),
                specs=specs,
                qualification_cards=qualification_cards,
                policy=policy,
                z=z,
            )
            selected = template
            selection_reason = "no-safe-nonempty-source-portfolio"

        leave_one_out = []
        if selected.expert_ids:
            for expert_id in selected.expert_ids:
                ablated_ids = tuple(
                    value for value in selected.expert_ids if value != expert_id
                )
                ablated = by_ids.get(ablated_ids)
                if ablated is None:
                    ablated = _simulate_subset(
                        values,
                        ablated_ids,
                        specs=specs,
                        qualification_cards=qualification_cards,
                        policy=policy,
                        z=z,
                    )
                leave_one_out.append(
                    {
                        "expert_id": expert_id,
                        "utility_mean_delta_when_present": (
                            selected.utility_mean - ablated.utility_mean
                        ),
                        "utility_lcb_delta_when_present": (
                            selected.utility_lcb - ablated.utility_lcb
                        ),
                        "harm_ucb_delta_when_present": (
                            selected.harm_ucb - ablated.harm_ucb
                        ),
                        "coverage_lcb_delta_when_present": (
                            selected.coverage_lcb - ablated.coverage_lcb
                        ),
                        "removing_expert_improves_mean_utility": (
                            ablated.utility_mean > selected.utility_mean
                        ),
                    }
                )

        pair_interactions = []
        selected_or_eligible = selected.expert_ids or tuple(eligible)
        for left, right in itertools.combinations(selected_or_eligible, 2):
            pair = by_ids.get(tuple(sorted((left, right))))
            left_card = by_ids.get((left,))
            right_card = by_ids.get((right,))
            if pair is None or left_card is None or right_card is None:
                continue
            pair_interactions.append(
                {
                    "experts": [left, right],
                    "interaction_utility": (
                        pair.utility_mean
                        - left_card.utility_mean
                        - right_card.utility_mean
                    ),
                    "pair_harm_ucb": pair.harm_ucb,
                }
            )

        buckets.append(
            {
                "modality": key[0],
                "task": key[1],
                "claim_type": key[2],
                "selection_reason": selection_reason,
                "selected": _card_dict(selected),
                "eligible_experts": eligible,
                "leave_one_out": leave_one_out,
                "pair_interactions": pair_interactions,
                "candidate_portfolios": [
                    _card_dict(card)
                    for card in sorted(
                        candidates,
                        key=lambda card: (
                            -card.utility_lcb,
                            card.harm_ucb,
                            card.estimated_cost_units,
                            card.expert_ids,
                        ),
                    )
                ],
            }
        )

    return {
        "schema": "merit-expert-portfolio-v1",
        "source_only": True,
        "selection_rule": {
            "feasibility": (
                "positive population/action utility LCB, bounded harm UCB, "
                "conditional precision, action/domain support"
            ),
            "ranking": (
                "lexicographic utility LCB -> action utility LCB -> coverage LCB "
                "-> lower harm/cost -> fewer experts"
            ),
            "empty_portfolio": "allowed and preserves the immutable incumbent",
            "target_test_selection": False,
            "z": float(z),
        },
        "buckets": buckets,
    }


def _card_dict(card):
    return {
        "modality": card.modality,
        "task": card.task,
        "claim_type": card.claim_type,
        "expert_ids": list(card.expert_ids),
        "fault_groups": list(card.fault_groups),
        "n_groups": card.n_groups,
        "domains": list(card.domains),
        "action_domains": list(card.action_domains),
        "commit_n": card.commit_n,
        "helpful_n": card.helpful_n,
        "harmful_n": card.harmful_n,
        "neutral_n": card.neutral_n,
        "utility_mean": card.utility_mean,
        "utility_lcb": card.utility_lcb,
        "action_utility_mean": card.action_utility_mean,
        "action_utility_lcb": card.action_utility_lcb,
        "harm_ucb": card.harm_ucb,
        "precision_lcb": card.precision_lcb,
        "coverage_lcb": card.coverage_lcb,
        "estimated_cost_units": card.estimated_cost_units,
        "feasible": card.feasible,
        "reason": card.reason,
        "source_only": card.source_only,
    }


def load_expert_portfolio(path):
    if path is None:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != "merit-expert-portfolio-v1":
        raise ValueError("unsupported expert portfolio schema")
    if payload.get("source_only") is not True:
        raise ValueError("expert portfolio must be fitted on source-only observations")
    result = {}
    for bucket in payload.get("buckets", ()):
        selected = bucket.get("selected", {})
        key = (
            str(bucket["modality"]),
            str(bucket["task"]),
            str(bucket["claim_type"]),
        )
        if key in result:
            raise ValueError(f"duplicate expert portfolio bucket: {key}")
        result[key] = tuple(str(value) for value in selected.get("expert_ids", ()))
    return result


def portfolio_experts_for(policy, *, modality, task, claim_type):
    if not policy:
        return None
    exact = (str(modality), str(task), str(claim_type))
    wildcard = (str(modality), str(task), "*")
    if exact in policy:
        return policy[exact]
    if wildcard in policy:
        return policy[wildcard]
    # A supplied portfolio policy is fail-closed: an unseen source cell does not
    # regain authority merely because a model exists in the global pool.
    return ()
