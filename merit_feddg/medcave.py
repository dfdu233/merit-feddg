"""Frozen free-answer intervention policy. Source validation is calibration-only.

No learned verifier is invented here: absent candidate-bound observations remain
unknown. A certificate validates a frozen trajectory policy, not a risk score
as a probability and not safety under arbitrary distribution shift.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from time import perf_counter

from .capabilities import CapabilityRequest, tool_descriptors, validate_result
from .intervention_risk import InterventionSignals


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass(frozen=True)
class FrozenPolicy:
    version: str = "medcave-free-v1"
    accept_threshold: float = 0.1
    max_expert_calls: int = 2
    max_candidates: int = 2
    max_probe_calls: int = 0
    minimum_utility: float = 0.0
    target_harm: float = 0.05
    alpha: float = 0.05
    minimum_accepted_groups: int = 60

    def __post_init__(self):
        for k in ("accept_threshold", "target_harm", "alpha", "minimum_utility"):
            if not math.isfinite(getattr(self, k)):
                raise ValueError("finite policy values required")
        if not 0 <= self.accept_threshold <= 1 or not 0 < self.target_harm < 1:
            raise ValueError("invalid risk threshold")
        if not 0 < self.alpha < 1 or self.minimum_utility < 0:
            raise ValueError("invalid confidence/utility")
        for k in ("max_expert_calls", "max_candidates", "minimum_accepted_groups"):
            if type(getattr(self, k)) is not int or getattr(self, k) < 1:
                raise ValueError("positive integer budgets required")
        if self.max_expert_calls > 2 or self.max_candidates > 2 or self.max_probe_calls != 0:
            raise ValueError("v1 supports at most two tools/candidates; visual probes unavailable")


def binomial_upper(harms, n, alpha):
    """One-sided exact Clopper-Pearson upper bound for fixed IID Bernoulli units."""
    if type(n) is not int or type(harms) is not int or not 0 <= harms <= n:
        raise ValueError("invalid counts")
    if not 0 < alpha < 1:
        raise ValueError("invalid alpha")
    if not n or harms == n:
        return 1.0
    low, high = 0.0, 1.0
    for _ in range(70):
        p = (low + high) / 2
        logs = [
            math.lgamma(n + 1)
            - math.lgamma(k + 1)
            - math.lgamma(n - k + 1)
            + k * math.log(p)
            + (n - k) * math.log1p(-p)
            for k in range(harms + 1)
        ]
        peak = max(logs)
        cdf = math.exp(peak) * sum(math.exp(v - peak) for v in logs)
        if cdf > alpha:
            low = p
        else:
            high = p
    return high


def certify(trajectories, scores, binding, policy, dev_units):
    """No threshold search. One preselected (minimum ID) trajectory per group.

    scores are produced OFFLINE after the complete source-cal trajectories have
    finished. The caller must audit source-dev/cal disjointness first.
    """
    groups = {}
    for t in sorted(trajectories, key=lambda x: x["id"]):
        if t["binding"] != binding or t["phase"] != "source-cal" or not t["complete"]:
            raise ValueError("certificate requires complete same-policy source-cal trajectories")
        if t["group_id"] in dev_units["groups"] or t["image_sha256"] in dev_units["images"]:
            raise ValueError("source-dev/cal group or image leakage")
        groups.setdefault(t["group_id"], t)
    accepted = [t for t in groups.values() if t["policy_action"] == "accept"]
    harms = sum(scores[t["id"]]["candidate"] < scores[t["id"]]["baseline"] for t in accepted)
    upper = binomial_upper(harms, len(accepted), policy.alpha)
    enabled = len(accepted) >= policy.minimum_accepted_groups and upper <= policy.target_harm
    result = {
        "schema": "medcave-source-certificate-v1",
        "binding": binding,
        "acceptance_disabled": not enabled,
        "policy": asdict(policy),
        "accepted_groups": len(accepted),
        "groups": len(groups),
        "harmful_groups": harms,
        "conditional_harm": harms / len(accepted) if accepted else None,
        "joint_harm": harms / len(groups) if groups else None,
        "harm_upper": upper,
        "method": "fixed-policy-exact-binomial-upper",
        "unit_rule": "minimum_case_id_per_predeclared_independent_group",
        "sampling_assumption": "independent identically sampled groups from source population",
        "target_domain_guarantee": False,
        "dev_units": dev_units,
        "cal_units": {
            "groups": sorted(groups),
            "images": sorted({t["image_sha256"] for t in trajectories}),
        },
        "trajectory_digest": digest(trajectories),
    }
    result["digest"] = digest(result)
    return result


def certificate_valid(certificate, binding):
    if not certificate or certificate.get("schema") != "medcave-source-certificate-v1":
        return False
    body = {k: v for k, v in certificate.items() if k != "digest"}
    if certificate.get("digest") != digest(body) or certificate.get("binding") != binding:
        return False
    try:
        p = FrozenPolicy(**certificate["policy"])
        n, h = certificate["accepted_groups"], certificate["harmful_groups"]
        return (
            certificate.get("acceptance_disabled") is False
            and n >= p.minimum_accepted_groups
            and binomial_upper(h, n, p.alpha) <= p.target_harm
        )
    except (KeyError, TypeError, ValueError):
        return False


def answer_valid(answer):
    return (
        isinstance(answer, dict)
        and isinstance(answer.get("text"), str)
        and bool(answer["text"].strip())
        and isinstance(answer.get("token_ids"), list)
        and bool(answer["token_ids"])
        and all(type(t) is int and t >= 0 for t in answer["token_ids"])
        and isinstance(answer.get("generation_config"), dict)
    )


class CandidateSignalBuilder:
    """EvidenceBridge coverage and explicitly bound verifier observations.

    A full free-text candidate is the requested proposition. Catalog scores do
    not match it automatically. Candidate support must be provided by a real
    semantic adapter, with the exact candidate/state identity; no keyword-based
    normality/absence inference is performed. Qualifications and checks are
    caller-owned audited artifacts, never scalar defaults from tool prose.
    """

    def __call__(self, row, candidate, items, state_hash, qualification=None, checks=None):
        from .claims import CandidateProposition, ClaimSpec
        from .evidence_bridges import EvidenceBridgeRegistry
        from .med_defer import NativeEvidence

        fields = {
            k: {"status": "unknown", "value": None, "reason": "not_measured"}
            for k in (
                "coverage",
                "source_reliability",
                "pre_ood",
                "post_ood",
                "conflict",
                "instability",
                "specificity",
                "visual_consistency",
            )
        }
        fields["applicability"] = {
            "status": "observed",
            "value": 1.0,
            "reason": "request_and_tool_contract_validated",
        }
        fields["expert_confidence"] = {
            "status": "not_applicable",
            "value": None,
            "reason": "native_score_is_not_correctness_probability",
        }
        hard, bridge_rows = [], []
        for item in items:
            if item.capability == "retrieval":
                bridge_rows.append(
                    {"expert": item.expert_id, "reason": "source_case_not_patient_fact"}
                )
                continue
            semantic = item.provenance.get("semantic_candidate_support")
            if not semantic or semantic.get("state_hash") != state_hash:
                bridge_rows.append({"expert": item.expert_id, "reason": "semantic_mapping_unknown"})
                continue
            if semantic.get("candidate_hash") != digest(candidate):
                hard.append("candidate_binding_mismatch")
                continue
            claim = ClaimSpec(
                claim_id=row["id"],
                question=row["question"],
                modality=row["modality"],
                required_capabilities=(item.capability,),
                propositions=(
                    CandidateProposition("candidate", candidate["text"], candidate["text"]),
                ),
                closed_set=False,
            )
            native = NativeEvidence(
                item.expert_id,
                item.capability,
                semantic.get("proposition_scores", {}),
                0.0,
                generated_text=item.summary or None,
                provenance={"score_semantics": semantic.get("score_semantics", "logit")},
            )
            bridge = EvidenceBridgeRegistry().convert(claim, native)
            bridge_rows.append(
                {
                    "expert": item.expert_id,
                    "reason": bridge.reason,
                    "coverage": min(c.coverage for c in bridge.candidates),
                }
            )
        # A new expert never erases unknown/contradictory old evidence.
        if bridge_rows and all(v.get("coverage") == 1 for v in bridge_rows):
            fields["coverage"] = {
                "status": "observed",
                "value": 1.0,
                "reason": "all_candidate_propositions_covered_by_bridge",
            }
        if not any(i.capability in {"segmentation", "detection"} for i in items):
            fields["visual_consistency"]["status"] = "not_applicable"
            fields["visual_consistency"]["reason"] = "no_spatial_check_contract"
        # These envelopes must be constructed by a verifier, with complete bindings.
        for artifact in (qualification, checks):
            if not artifact:
                continue
            if (
                artifact.get("state_hash") != state_hash
                or artifact.get("candidate_hash") != digest(candidate)
                or not artifact.get("source_artifact")
            ):
                hard.append("verification_binding_mismatch")
                continue
            for key, value in artifact.get("signals", {}).items():
                if key not in fields or key in {"coverage", "applicability", "expert_confidence"}:
                    continue
                if not math.isfinite(value) or not 0 <= value <= 1:
                    hard.append("invalid_verifier_signal")
                    continue
                fields[key] = {
                    "status": "observed",
                    "value": value,
                    "source": artifact["source_artifact"],
                }
        # No visual measurement is invented for a classifier without a mask.
        required = [k for k in fields if fields[k]["status"] != "not_applicable"]
        unknown = [k for k in required if fields[k]["status"] != "observed"]
        signal_args = {k: v["value"] for k, v in fields.items() if k != "specificity"}
        signals = InterventionSignals(**signal_args, metadata={"hard_failures": hard})
        observed = signals.risk_components()
        values = [v for v in observed.values() if v is not None]
        return {
            "fields": fields,
            "unknown": unknown,
            "hard_failures": hard,
            "score": max(values) if values else None,
            "bridge": bridge_rows,
            "candidate_hash": digest(candidate),
            "state_hash": state_hash,
            "raw_expert_confidences": [i.confidence for i in items],
        }


def run_case(
    row,
    *,
    binding,
    policy,
    descriptors,
    fingerprints,
    pool,
    generate,
    certificate=None,
    phase="source-smoke",
    verifier=None,
):
    """Two real acquisitions, cumulative evidence, immutable baseline/final token IDs."""
    start = perf_counter()
    baseline = copy.deepcopy(generate(()))
    if not answer_valid(baseline):
        raise ValueError("invalid original answer")
    result = {
        "id": row["id"],
        "group_id": row["group_id"],
        "image_sha256": row["image_sha256"],
        "binding": binding,
        "phase": phase,
        "input_hash": digest(row),
        "baseline": baseline,
        "final": copy.deepcopy(baseline),
        "candidate": None,
        "action": "fallback",
        "policy_action": "fallback",
        "steps": [],
        "logical_calls": 0,
        "actual_tool_calls": 0,
        "cache_hits": 0,
        "candidate_calls": 0,
        "probe_calls": 0,
        "complete": True,
        "reason": "no_qualified_utility",
        "acceptance_disabled": not certificate_valid(certificate, binding),
    }
    used, items = set(), ()
    for descriptor in descriptors:
        if result["logical_calls"] >= policy.max_expert_calls:
            result["reason"] = "acquisition_budget_exhausted"
            break
        if result["candidate_calls"] >= policy.max_candidates:
            result["reason"] = "candidate_budget_exhausted"
            break
        expert = descriptor["expert"]
        model_source = fingerprints.get(expert)
        utility = descriptor.get("utility")
        if (
            not model_source
            or model_source in used
            or utility is None
            or not math.isfinite(utility)
            or utility <= policy.minimum_utility
        ):
            continue
        used.add(model_source)
        result["logical_calls"] += 1
        request = CapabilityRequest(
            row["id"],
            row["image"],
            row["question"],
            row["modality"],
            row["task"],
            row["domain"],
            row["group_id"],
            descriptor["capability"],
            query=row["question"],
            scope=descriptor["scope"],
        )
        step = {
            "expert": expert,
            "model_source": model_source,
            "request_hash": digest(asdict(request)),
            "capability": request.capability,
            "status": "attempted",
            "utility": utility,
        }
        result["steps"].append(step)
        stamp = perf_counter()
        try:
            output = pool.infer(expert, request)
            origin = getattr(pool, "last_origin", "live_native_output")
            cache_hit = origin in {"cached_native_output", "reused_compatible_native_output"}
            result["cache_hits" if cache_hit else "actual_tool_calls"] += 1
            step["cache_hit"] = cache_hit
            validate_result(output, expert, request)
            if not output.items:
                step.update(status="unavailable", reason=output.reason)
                continue
            items += tuple(output.items)
            state_hash = digest([asdict(i) for i in items])
            step["evidence"] = [asdict(i) for i in items]
            result["candidate_calls"] += 1
            candidate = generate(items)
            if not answer_valid(candidate):
                raise ValueError("empty_or_invalid_candidate")
            transport = candidate.get("evidence_transport", {})
            presented = {(i["expert_id"], i["evidence_id"]) for i in transport.get("presented", [])}
            expected = {(i.expert_id, i.evidence_id) for i in items}
            result["candidate"] = copy.deepcopy(candidate)
            audit = (verifier or CandidateSignalBuilder())(row, candidate, items, state_hash)
            if audit["candidate_hash"] != digest(candidate) or audit["state_hash"] != state_hash:
                raise ValueError("verifier_did_not_evaluate_this_candidate_and_state")
            if presented != expected:
                audit["hard_failures"].append("evidence_not_fully_delivered")
            step.update(
                status="evaluated",
                verification=audit,
                candidate=copy.deepcopy(candidate),
                state_hash=state_hash,
            )
            eligible = (
                not audit["unknown"]
                and not audit["hard_failures"]
                and audit["score"] is not None
                and math.isfinite(audit["score"])
                and 0 <= audit["score"] <= policy.accept_threshold
            )
            if eligible:
                result["policy_action"] = "accept"
                step["decision"] = "accept"
                if not result["acceptance_disabled"]:
                    result.update(
                        action="accept", final=copy.deepcopy(candidate), reason="certified_policy"
                    )
                else:
                    result["reason"] = "acceptance_disabled"
                break  # Verify precisely this candidate; no extra generation after accept.
            step["decision"] = "fallback" if audit["hard_failures"] else "acquire"
            if audit["hard_failures"]:
                result["reason"] = "structural_or_delivery_failure"
                break
            result["reason"] = "missing_signals_or_risk_budget"
        except (
            RuntimeError,
            ValueError,
            TypeError,
            OSError,
            LookupError,
            AttributeError,
            ArithmeticError,
        ) as exc:
            if "cache_hit" not in step:
                result["actual_tool_calls"] += 1
            step.update(status="failed", error_type=type(exc).__name__)
            result["reason"] = "runtime_failure"
            result["complete"] = False
            break
        finally:
            step["seconds"] = perf_counter() - stamp
    result["seconds"] = perf_counter() - start
    return result


def admissible_descriptors(specs, row, utilities):
    """Hard capability/intent filtering before risk aggregation or tool invocation."""
    from .request_scope import assess_request

    descriptors = []
    for d in tool_descriptors(specs, row):
        if (
            d["requires_region"]
            or not assess_request(row["question"], specs[d["expert"]], d["capability"])["allowed"]
        ):
            continue
        descriptors.append({**d, "utility": utilities.get(d["expert"])})
    return sorted(descriptors, key=lambda d: (-(d["utility"] or 0), d["expert"]))


class SourceArtifactVerifier:
    """Reuse qualification/OOD artifacts when their complete runtime identity matches.

    Optional adapter feature payloads require request and encoder provenance.
    Missing artifacts/features stay unknown. Explicit contradictory assertions
    are compared only within the same current-image proposition and location.
    """

    def __init__(self, specs, fingerprints):
        from pathlib import Path

        from .qualification import QualificationArtifact

        self.specs, self.fingerprints = specs, fingerprints
        self.artifacts = {
            k: QualificationArtifact.from_json(Path(s["qualification_path"]).read_text())
            for k, s in specs.items()
            if s.get("qualification_path")
        }

    def __call__(self, row, candidate, items, state_hash):
        from .qualification import ExpertIdentity, QualificationGate

        audit = CandidateSignalBuilder()(row, candidate, items, state_hash)
        gate, reliabilities, pre, post, provenance = QualificationGate(), [], [], [], []
        assertions = {}
        for item in items:
            spec = self.specs[item.expert_id]
            artifact = self.artifacts.get(item.expert_id)
            identity = ExpertIdentity(
                item.expert_id,
                self.fingerprints[item.expert_id],
                digest(
                    {
                        "adapter": spec.get("adapter"),
                        "factory": spec.get("factory"),
                        "kwargs": spec.get("factory_kwargs", {}),
                    }
                ),
                row["modality"],
                item.capability,
                row["task"],
            )
            decision = gate.authorize(identity, artifact)
            provenance.append({"expert": item.expert_id, "qualification": decision.reason})
            if decision.allowed:
                reliabilities.append(artifact.performance_lcb)
                features = item.provenance.get("risk_features", {})
                if features.get("input_hash") == digest(row):
                    for stage, output, method in (
                        ("pre", pre, gate.pre_call_ood),
                        ("post", post, gate.post_call_native_ood),
                    ):
                        feature = features.get(stage)
                        if feature:
                            assessment = method(
                                identity,
                                artifact,
                                feature["values"],
                                feature["encoder_fingerprint"],
                            )
                            if assessment.allowed:
                                output.append(assessment.score)
            if item.capability == "retrieval":
                continue
            for statement in item.payload.get("assertions", []):
                if (
                    statement.get("image_sha256") != row["image_sha256"]
                    or statement.get("scope") != item.scope
                    or statement.get("polarity") not in {"positive", "negative"}
                    or not statement.get("proposition")
                    or "location" not in statement
                ):
                    continue
                key = digest([statement["proposition"], statement["location"]])
                assertions.setdefault(key, set()).add(statement["polarity"])
        for name, values, combine in (
            ("source_reliability", reliabilities, min),
            ("pre_ood", pre, max),
            ("post_ood", post, max),
        ):
            if len(values) == len(items) and values:
                audit["fields"][name] = {
                    "status": "observed",
                    "value": combine(values),
                    "source": "QualificationGate",
                    "details": provenance,
                }
        if any(len(polarities) > 1 for polarities in assertions.values()):
            audit["fields"]["conflict"] = {
                "status": "observed",
                "value": 1.0,
                "source": "same_proposition_location_polarity",
            }
            audit["hard_failures"].append("explicit_same_proposition_conflict")
        audit["unknown"] = [k for k, v in audit["fields"].items() if v["status"] == "unknown"]
        values = {k: v["value"] for k, v in audit["fields"].items() if k != "specificity"}
        risks = InterventionSignals(**values).risk_components()
        audit["components"] = risks
        audit["score"] = max(v for v in risks.values() if v is not None)
        audit["qualification_audit"] = provenance
        return audit


def replay_trajectory(trajectory, row, binding, policy):
    """Replay only exact full-state verdicts; never synthesize missing joint answers."""
    t = trajectory
    if (
        not t
        or t.get("binding") != binding
        or t.get("input_hash") != digest(row)
        or not t.get("complete")
        or not answer_valid(t.get("baseline"))
        or t.get("logical_calls", 0) > policy.max_expert_calls
        or t.get("candidate_calls", 0) > policy.max_candidates
    ):
        return {"status": "replay_unavailable"}
    previous = []
    for step in t.get("steps", []):
        if step.get("status") != "evaluated":
            continue
        evidence = step.get("evidence", [])
        if evidence[: len(previous)] != previous:
            return {"status": "replay_unavailable", "reason": "joint_evidence_missing"}
        previous = evidence
        audit = step.get("verification", {})
        if (
            not answer_valid(step.get("candidate"))
            or audit.get("candidate_hash") != digest(step["candidate"])
            or audit.get("state_hash") != digest(evidence)
        ):
            return {"status": "replay_unavailable", "reason": "candidate_or_state_mismatch"}
    return {
        "status": "exact_trajectory_available",
        "control_flow_only": True,
        "recorded_action": t["action"],
        "logical_calls": t["logical_calls"],
        "new_generation_calls": 0,
        "accuracy": None,
    }
