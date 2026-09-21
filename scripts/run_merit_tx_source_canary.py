"""Run a frozen source/development MERIT-Tx canary.

Candidate outputs and immutable Generalist baselines must already exist before
this program runs.  Target/test rows and labels in the generation manifest are
forbidden.  Optional source references are read only after every transaction
decision and final output have been frozen, solely for post-hoc canary scoring.
"""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from merit_feddg.capability_study import _filter_optional_experts
from merit_feddg.contribution import answer_metrics
from merit_feddg.expert_policy import load_qualification_cards
from merit_feddg.expert_portfolio import load_expert_portfolio
from merit_feddg.io import load_experiment_yaml
from merit_feddg.open_experts import OpenExpertPool
from merit_feddg.open_study import atomic_json
from merit_feddg.transactional_claims import (
    RadGraphClaimizer,
    apply_transactions,
    candidate_transactions,
    claim_truth_key,
)
from merit_feddg.transactional_runtime import (
    claimize_pair,
    normalize_proposer_expert_ids,
    verify_transaction,
)

_LABEL_KEYS = frozenset(
    {"answer", "answers", "label", "labels", "reference", "references", "ground_truth"}
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path):
    rows = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        required = ("id", "image", "question", "modality", "task", "domain", "group_id")
        missing = [key for key in required if not str(row.get(key, "")).strip()]
        if missing:
            raise ValueError(f"manifest line {line_number}: missing {missing}")
        if _LABEL_KEYS.intersection(row):
            raise ValueError(f"manifest line {line_number}: labels/references are forbidden")
        split = str(row.get("split", row.get("role", "source"))).casefold()
        if split not in {"source", "train", "training", "development", "dev"}:
            raise ValueError(f"manifest line {line_number}: target/test rows are forbidden")
        rows.append(row)
    if not rows:
        raise ValueError("source manifest is empty")
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("source manifest IDs must be unique")
    return rows


def load_outputs(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("outputs"), dict):
        payload = payload["outputs"]
    if not isinstance(payload, dict) or not payload:
        raise TypeError("output file must be a non-empty sample-id mapping")
    result = {}
    for sample_id, row in payload.items():
        if not isinstance(row, dict) or not isinstance(row.get("text"), str):
            raise TypeError(f"invalid output row for {sample_id}")
        result[str(sample_id)] = row["text"]
    return result


def load_references(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("references must be a sample-id mapping")
    result = {}
    for sample_id, value in payload.items():
        values = [value] if isinstance(value, str) else value
        if not isinstance(values, list) or not values or any(not isinstance(v, str) for v in values):
            raise TypeError(f"invalid reference row for {sample_id}")
        result[str(sample_id)] = values
    return result


def report_claim_f1(text, references, claimizer):
    predicted = {claim_truth_key(claim) for claim in claimizer(text)}
    best = 0.0
    for reference in references:
        truth = {claim_truth_key(claim) for claim in claimizer(reference)}
        if not predicted and not truth:
            score = 1.0
        elif not predicted or not truth:
            score = 0.0
        else:
            overlap = len(predicted & truth)
            precision = overlap / len(predicted)
            recall = overlap / len(truth)
            score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        best = max(best, score)
    return best


def resolved_claim_type(row, requested):
    if requested != "auto":
        return requested
    return str(row.get("question_type") or "*")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--control-manifest")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--qualification-cards")
    parser.add_argument("--portfolio-policy")
    parser.add_argument("--config", default="configs/merit_tx.yaml")
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--proposer-expert-id")
    parser.add_argument(
        "--proposer-expert-ids",
        nargs="*",
        default=(),
        help=(
            "All experts that may have influenced the frozen candidate. "
            "Accepts repeated values or comma-separated groups."
        ),
    )
    parser.add_argument("--claim-type", default="auto")
    parser.add_argument("--max-calls", type=int)
    parser.add_argument("--knockoff-controls", type=int)
    parser.add_argument("--references")
    parser.add_argument("--metric", choices=("token_f1", "exact_match"), default="token_f1")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.max_calls is not None and args.max_calls < 1:
        parser.error("--max-calls must be positive")
    if args.knockoff_controls is not None and args.knockoff_controls < 1:
        parser.error("--knockoff-controls must be positive")
    if not args.candidate_name.strip():
        parser.error("--candidate-name cannot be empty")

    rows = load_manifest(args.manifest)
    controls = load_manifest(args.control_manifest or args.manifest)
    baseline = load_outputs(args.baseline)
    candidate = load_outputs(args.candidate)
    sample_ids = [str(row["id"]) for row in rows]
    if set(baseline) != set(sample_ids) or set(candidate) != set(sample_ids):
        raise ValueError("baseline/candidate IDs must exactly match source manifest")

    config = load_experiment_yaml(args.config)
    specs, excluded = _filter_optional_experts(config["experts"], args.artifacts)
    specs.pop("source_cases", None)
    proposer_ids = normalize_proposer_expert_ids(
        args.proposer_expert_id,
        args.proposer_expert_ids,
    )
    unknown_proposers = [expert_id for expert_id in proposer_ids if expert_id not in specs]
    if unknown_proposers:
        raise ValueError(
            "proposer experts are unavailable under the frozen config: "
            + ", ".join(unknown_proposers)
        )
    tx_policy = config.get("merit_tx", {})
    card_path = args.qualification_cards or tx_policy.get("qualification_cards")
    if not card_path or not Path(card_path).is_file():
        raise FileNotFoundError(
            "frozen source qualification cards are required before the MERIT-Tx canary"
        )
    qualification_payload = json.loads(Path(card_path).read_text(encoding="utf-8"))
    qualification_groups = {
        str(value) for value in qualification_payload.get("source_group_ids", ())
    }
    if not qualification_groups:
        raise ValueError("v3 qualification cards must record source_group_ids")
    cards = load_qualification_cards(card_path)
    portfolio_path = args.portfolio_policy or tx_policy.get("portfolio_cards")
    if not portfolio_path or not Path(portfolio_path).is_file():
        raise FileNotFoundError(
            "frozen source expert portfolio is required before the MERIT-Tx v3 canary"
        )
    portfolio_payload = json.loads(Path(portfolio_path).read_text(encoding="utf-8"))
    portfolio_groups = {
        str(value) for value in portfolio_payload.get("source_group_ids", ())
    }
    if not portfolio_groups:
        raise ValueError("v3 expert portfolio must record source_group_ids")
    canary_groups = {str(row["group_id"]) for row in rows}
    if qualification_groups & portfolio_groups:
        raise ValueError("qualification and portfolio source groups overlap")
    if canary_groups & qualification_groups:
        raise ValueError("canary source groups overlap qualification source groups")
    if canary_groups & portfolio_groups:
        raise ValueError("canary source groups overlap portfolio-selection source groups")
    portfolio_policy = load_expert_portfolio(portfolio_path)
    max_calls = (
        args.max_calls
        if args.max_calls is not None
        else int(config.get("capability_value", {}).get("generation", {}).get("max_expert_calls", 6))
    )
    knockoff_count = (
        args.knockoff_controls
        if args.knockoff_controls is not None
        else int(tx_policy.get("matched_knockoff_controls", 4))
    )

    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output}")
    (output / "cases").mkdir(parents=True, exist_ok=True)
    protocol = {
        "schema": "merit-tx-source-canary-v1",
        "manifest": str(Path(args.manifest).resolve()),
        "manifest_sha256": sha256(args.manifest),
        "control_manifest": str(Path(args.control_manifest or args.manifest).resolve()),
        "control_manifest_sha256": sha256(args.control_manifest or args.manifest),
        "baseline": str(Path(args.baseline).resolve()),
        "baseline_sha256": sha256(args.baseline),
        "candidate": str(Path(args.candidate).resolve()),
        "candidate_sha256": sha256(args.candidate),
        "candidate_name": args.candidate_name,
        "proposer_expert_id": args.proposer_expert_id,
        "proposer_expert_ids": list(proposer_ids),
        "config": str(Path(args.config).resolve()),
        "config_sha256": sha256(args.config),
        "qualification_cards": str(Path(card_path).resolve()),
        "qualification_cards_sha256": sha256(card_path),
        "portfolio_policy": str(Path(portfolio_path).resolve()),
        "portfolio_policy_sha256": sha256(portfolio_path),
        "portfolio_selection_source_only": True,
        "qualification_portfolio_canary_group_disjoint": True,
        "qualification_source_groups_sha256": qualification_payload.get("source_groups_sha256"),
        "portfolio_source_groups_sha256": portfolio_payload.get("source_groups_sha256"),
        "max_calls": max_calls,
        "knockoff_controls": knockoff_count,
        "immutable_incumbent": True,
        "candidate_frozen_before_verification": True,
        "references_available_during_decisions": False,
        "target_test_selection": False,
        "excluded_optional_experts": excluded,
    }
    atomic_json(output / "protocol.json", protocol)

    pool = OpenExpertPool(specs, args.artifacts)
    radgraph = None
    final_outputs = {}
    case_summaries = []
    try:
        for row in rows:
            sample_id = str(row["id"])
            if row["task"] == "report_generation" and radgraph is None:
                radgraph = RadGraphClaimizer(
                    model_type=str(tx_policy.get("radiology_grounding", "modern-radgraph-xl"))
                )
            baseline_claims, candidate_claims = claimize_pair(
                row,
                baseline[sample_id],
                candidate[sample_id],
                radgraph_claimizer=radgraph,
            )
            transactions = candidate_transactions(
                task=row["task"],
                question=row["question"],
                baseline_text=baseline[sample_id],
                candidate_text=candidate[sample_id],
                baseline_claims=baseline_claims,
                candidate_claims=candidate_claims,
                proposer_expert_id=args.proposer_expert_id,
                proposer_expert_ids=proposer_ids,
                transaction_prefix=f"{sample_id}:{args.candidate_name}",
            )
            decisions = []
            verification = []
            claim_type = resolved_claim_type(row, args.claim_type)
            for transaction in transactions:
                decision, evidences, audit = verify_transaction(
                    row=row,
                    transaction=transaction,
                    baseline_claims=baseline_claims,
                    source_controls=controls,
                    specs=specs,
                    qualification_cards=cards,
                    pool=pool,
                    tx_policy=tx_policy,
                    claim_type=claim_type,
                    max_calls=max_calls,
                    knockoff_count=knockoff_count,
                    portfolio_policy=portfolio_policy,
                )
                decisions.append(decision)
                verification.append(
                    {
                        "transaction": asdict(transaction),
                        "evidence": evidences,
                        "decision": asdict(decision),
                        "audit": audit,
                    }
                )

            try:
                rendered = apply_transactions(
                    baseline[sample_id],
                    baseline_claims,
                    transactions,
                    decisions,
                )
                rendering_error = None
            except ValueError as exc:
                if "overlap" not in str(exc):
                    raise
                rendered = {
                    "text": baseline[sample_id],
                    "baseline_text": baseline[sample_id],
                    "committed_transactions": [],
                    "suppressed_partial_patch_transactions": [
                        transaction.transaction_id
                        for transaction, decision in zip(
                            transactions, decisions, strict=True
                        )
                        if decision.commit
                    ],
                    "fallback_exact": True,
                    "untouched_baseline_preserved": True,
                }
                rendering_error = str(exc)

            final_outputs[sample_id] = {
                "text": rendered["text"],
                "baseline_text": baseline[sample_id],
                "candidate_text": candidate[sample_id],
                "committed_transactions": rendered["committed_transactions"],
                "suppressed_partial_patch_transactions": rendered.get(
                    "suppressed_partial_patch_transactions", []
                ),
                "fallback_exact": rendered["fallback_exact"],
                "untouched_baseline_preserved": rendered["untouched_baseline_preserved"],
                "rendering_error": rendering_error,
            }
            case_payload = {
                "id": sample_id,
                "task": row["task"],
                "modality": row["modality"],
                "claim_type": claim_type,
                "baseline_claims": [asdict(claim) for claim in baseline_claims],
                "candidate_claims": [asdict(claim) for claim in candidate_claims],
                "verification": verification,
                "final": final_outputs[sample_id],
                "references_read": False,
            }
            atomic_json(output / "cases" / f"{sample_id}.json", case_payload)
            case_summaries.append(case_payload)
            pool.reset_case()
    finally:
        for model in pool.models.values():
            close = getattr(model, "close", None)
            if callable(close):
                close()

    # Freeze all generated outputs before references are opened.
    atomic_json(output / "outputs.json", final_outputs)
    generation_summary = {
        "n": len(rows),
        "changed_cases": sum(
            value["text"] != value["baseline_text"] for value in final_outputs.values()
        ),
        "fallback_cases": sum(value["fallback_exact"] for value in final_outputs.values()),
        "committed_transactions": sum(
            len(value["committed_transactions"]) for value in final_outputs.values()
        ),
        "suppressed_partial_patch_transactions": sum(
            len(value["suppressed_partial_patch_transactions"])
            for value in final_outputs.values()
        ),
        "references_read": False,
    }
    atomic_json(output / "generation-summary.json", generation_summary)

    if args.references:
        references = load_references(args.references)
        if set(references) != set(sample_ids):
            raise ValueError("reference IDs must exactly match source manifest")
        scores = {"baseline": [], "candidate": [], "merit_tx": []}
        report_claimizer = radgraph
        for row in rows:
            sample_id = str(row["id"])
            if row["task"] == "report_generation":
                if report_claimizer is None:
                    report_claimizer = RadGraphClaimizer(
                        model_type=str(
                            tx_policy.get("radiology_grounding", "modern-radgraph-xl")
                        )
                    )
                case_references = references[sample_id]
                scores["baseline"].append(
                    report_claim_f1(
                        baseline[sample_id], case_references, report_claimizer
                    )
                )
                scores["candidate"].append(
                    report_claim_f1(
                        candidate[sample_id], case_references, report_claimizer
                    )
                )
                scores["merit_tx"].append(
                    report_claim_f1(
                        final_outputs[sample_id]["text"],
                        case_references,
                        report_claimizer,
                    )
                )
            else:
                for name, text in (
                    ("baseline", baseline[sample_id]),
                    ("candidate", candidate[sample_id]),
                    ("merit_tx", final_outputs[sample_id]["text"]),
                ):
                    scores[name].append(
                        answer_metrics(text, references[sample_id])[args.metric]
                    )
        evaluation = {
            "schema": "merit-tx-source-canary-evaluation-v1",
            "n": len(rows),
            "metric": (
                "mixed: RadGraph claim-set F1 for reports, " + args.metric + " for VQA"
                if any(row["task"] == "report_generation" for row in rows)
                else args.metric
            ),
            "means": {
                name: sum(values) / len(values)
                for name, values in scores.items()
            },
            "references_loaded_after_outputs_frozen": True,
            "references_used_for_transaction_decisions": False,
            "target_test_selection": False,
        }
        atomic_json(output / "evaluation.json", evaluation)

    print(output)


if __name__ == "__main__":
    main()
