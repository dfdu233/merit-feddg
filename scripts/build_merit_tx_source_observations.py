"""Build source-only MERIT-Tx expert qualification observations.

The candidate output set must already be frozen without access to source
references. References are joined only after candidate generation to measure
whether the proposed immutable-baseline transaction helped or harmed.

One invocation evaluates one proposal distribution. This avoids counting the
same patient multiple times inside a qualification cell merely because several
candidate generators were tried.
"""
import argparse
import json
from pathlib import Path

from merit_feddg.capability_study import _filter_optional_experts
from merit_feddg.contribution import answer_metrics
from merit_feddg.expert_policy import role_card, transaction_descriptors
from merit_feddg.io import load_experiment_yaml
from merit_feddg.knockoff import select_matched_knockoffs
from merit_feddg.merit_tx import differential_margin_controls
from merit_feddg.open_experts import OpenExpertPool
from merit_feddg.transactional_claims import (
    RadGraphClaimizer,
    candidate_transactions,
    claim_truth_key,
    claimize_vqa,
)
from merit_feddg.transactional_runtime import (
    normalize_proposer_expert_ids,
    transaction_claim_spec,
)


def parse_named(value):
    name, separator, path = value.partition("=")
    if not separator or not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError("candidate must use NAME=/path/to/output.json")
    return name.strip(), Path(path.strip())


def load_rows(path):
    rows = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        required = ("id", "image", "question", "modality", "task", "domain", "group_id")
        missing = [key for key in required if not str(row.get(key, "")).strip()]
        if missing:
            raise ValueError(f"manifest line {line_number}: missing {missing}")
        split = str(row.get("split", row.get("role", "source"))).casefold()
        if split not in {"source", "train", "training", "development", "dev"}:
            raise ValueError(
                f"manifest line {line_number}: target/test rows are forbidden ({split})"
            )
        forbidden = {
            "answer", "answers", "label", "labels", "reference", "references", "ground_truth"
        }
        if forbidden.intersection(row):
            raise ValueError(
                f"manifest line {line_number}: labels must stay in the reference file"
            )
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
    if not isinstance(payload, dict):
        raise TypeError("output file must be a sample-id mapping")
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


def native_scores(pool, expert_id, image, claim):
    evidence = pool.evidence_function(expert_id, image)(claim, "")
    queries = [item.proposition for item in claim.propositions]
    try:
        values = [float(evidence.concept_scores[query]) for query in queries]
    except KeyError as exc:
        raise ValueError(
            f"{expert_id} did not return native scores for both transaction claims"
        ) from exc
    return values


def report_outcome(transaction, baseline_claims, reference_claims):
    """Source-only claim utility for the exact transaction operation."""
    reference_keys = {claim_truth_key(claim) for claim in reference_claims}
    after = int(claim_truth_key(transaction.proposed_claim) in reference_keys)
    if transaction.operation == "ADD":
        # A correct addition recovers an omitted clinical claim; an unsupported
        # addition is an explicit false-positive and therefore harmful.
        return 1.0 if after else -1.0
    if transaction.operation != "REPLACE":
        raise ValueError("source qualification supports ADD/REPLACE only")
    baseline = {
        claim.claim_id: claim
        for claim in baseline_claims
    }.get(transaction.baseline_claim_id)
    if baseline is None:
        raise ValueError("report transaction baseline claim is missing")
    before = int(claim_truth_key(baseline) in reference_keys)
    return float(after - before)


def collapse_group_observations(rows):
    """Use one conservative qualification observation per source patient/study group."""
    rows = tuple(rows)
    group_fields = (
        "expert_id",
        "capability",
        "scope",
        "modality",
        "task",
        "claim_type",
        "domain",
        "group_id",
    )
    grouped = {}
    for observation in rows:
        key = tuple(observation[field] for field in group_fields)
        grouped.setdefault(key, []).append(observation)
    collapsed = []
    for key in sorted(grouped):
        values = grouped[key]
        chosen = min(
            values,
            key=lambda row: (
                float(row["outcome_delta"]),
                float(row["differential_effect"]),
                str(row["transaction_id"]),
            ),
        )
        collapsed.append(
            {
                **chosen,
                "within_group_transactions": len(values),
                "within_group_aggregation": "worst-outcome-then-specificity",
            }
        )
    return collapsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True, type=parse_named)
    parser.add_argument("--references", required=True)
    parser.add_argument("--config", default="configs/merit_tx.yaml")
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--proposer-expert-id")
    parser.add_argument("--proposer-map")
    parser.add_argument(
        "--proposer-expert-ids",
        nargs="*",
        default=(),
        help=(
            "All experts that may have influenced the frozen candidate. "
            "Accepts repeated values or comma-separated groups."
        ),
    )
    parser.add_argument("--metric", choices=("token_f1", "exact_match"), default="token_f1")
    parser.add_argument("--claim-type", default="*")
    parser.add_argument("--knockoff-controls", type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.knockoff_controls is not None and args.knockoff_controls < 1:
        parser.error("--knockoff-controls must be positive")

    rows = load_rows(args.manifest)
    row_by_id = {str(row["id"]): row for row in rows}
    baseline = load_outputs(args.baseline)
    candidate_name, candidate_path = args.candidate
    candidate = load_outputs(candidate_path)
    references = load_references(args.references)
    ids = list(row_by_id)
    for name, mapping in (
        ("baseline", baseline),
        ("candidate", candidate),
        ("references", references),
    ):
        if set(mapping) != set(ids):
            raise ValueError(f"{name} IDs must exactly match the source manifest")

    proposer_map = json.loads(Path(args.proposer_map).read_text()) if args.proposer_map else {}
    if args.proposer_map and set(proposer_map) != set(ids):
        raise ValueError("proposer map IDs must match manifest")
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
    tx_config = config.get("merit_tx", {})
    knockoff_count = (
        args.knockoff_controls
        if args.knockoff_controls is not None
        else int(tx_config.get("matched_knockoff_controls", 4))
    )
    pool = OpenExpertPool(specs, args.artifacts)
    radgraph = None
    output_rows = []
    skipped = []
    expert_counts = {}

    try:
        for row in rows:
            sample_id = str(row["id"])
            if args.proposer_map:
                proposer_ids = normalize_proposer_expert_ids(None, proposer_map[sample_id])
                if any(name not in specs for name in proposer_ids):
                    raise ValueError("unavailable proposer")
            task = str(row["task"])
            if task == "report_generation":
                if radgraph is None:
                    radgraph = RadGraphClaimizer(
                        model_type=str(tx_config.get("radiology_grounding", "modern-radgraph-xl"))
                    )
                baseline_claims = radgraph(baseline[sample_id])
                candidate_claims = radgraph(candidate[sample_id])
                reference_claims = tuple(
                    claim
                    for reference in references[sample_id]
                    for claim in radgraph(reference)
                )
            else:
                baseline_claims = claimize_vqa(row["question"], baseline[sample_id])
                candidate_claims = claimize_vqa(row["question"], candidate[sample_id])
                reference_claims = ()

            transactions = candidate_transactions(
                task=task,
                question=row["question"],
                baseline_text=baseline[sample_id],
                candidate_text=candidate[sample_id],
                baseline_claims=baseline_claims,
                candidate_claims=candidate_claims,
                proposer_expert_id=args.proposer_expert_id,
                proposer_expert_ids=proposer_ids,
                transaction_prefix=f"{sample_id}:{candidate_name}",
            )
            if not transactions:
                skipped.append({"id": sample_id, "reason": "candidate-identical-to-incumbent"})
                continue

            descriptors = transaction_descriptors(specs, row)
            verifiers = []
            for descriptor in descriptors:
                card = role_card(descriptor["expert"], specs[descriptor["expert"]])
                if (
                    card.evidence_role == "direct_visual_verifier"
                    and card.commit_authority == "source_qualified"
                    and descriptor["capability"] == "classification"
                ):
                    verifiers.append(descriptor)
            if not verifiers:
                skipped.append({"id": sample_id, "reason": "no-native-direct-visual-verifier"})
                continue

            match_fields = ("question_type",) if row.get("question_type") else ()
            for transaction in transactions:
                if transaction.operation == "DELETE":
                    skipped.append(
                        {
                            "id": sample_id,
                            "transaction_id": transaction.transaction_id,
                            "reason": "delete-not-qualified-by-omission",
                        }
                    )
                    continue
                try:
                    claim = transaction_claim_spec(row, transaction, baseline_claims)
                except ValueError as exc:
                    skipped.append(
                        {
                            "id": sample_id,
                            "transaction_id": transaction.transaction_id,
                            "reason": "transaction-has-no-verifiable-counterfactual",
                            "detail": str(exc),
                        }
                    )
                    continue
                if task == "report_generation":
                    outcome_delta = report_outcome(
                        transaction, baseline_claims, reference_claims
                    )
                else:
                    before = answer_metrics(
                        baseline[sample_id], references[sample_id]
                    )[args.metric]
                    after = answer_metrics(
                        candidate[sample_id], references[sample_id]
                    )[args.metric]
                    outcome_delta = float(after - before)

                for descriptor in verifiers:
                    expert_id = descriptor["expert"]
                    try:
                        incumbent_real, candidate_real = native_scores(
                            pool, expert_id, row["image"], claim
                        )
                    except ValueError as exc:
                        skipped.append(
                            {
                                "id": sample_id,
                                "transaction_id": transaction.transaction_id,
                                "expert_id": expert_id,
                                "reason": "claim-not-natively-scorable",
                                "detail": str(exc),
                            }
                        )
                        continue

                    try:
                        controls = select_matched_knockoffs(
                            rows,
                            row,
                            expert_id=expert_id,
                            count=knockoff_count,
                            match_fields=match_fields,
                        )
                    except ValueError as exc:
                        if not str(exc).startswith("insufficient matched source controls"):
                            raise
                        skipped.append({"id": sample_id, "expert_id": expert_id,
                                        "reason": "insufficient-matched-controls"})
                        continue
                    knockoff_pairs = []
                    for control in controls:
                        incumbent_control, candidate_control = native_scores(
                            pool, expert_id, control["image"], claim
                        )
                        knockoff_pairs.append(
                            (incumbent_control, candidate_control)
                        )
                    effect = differential_margin_controls(
                        incumbent_real=incumbent_real,
                        candidate_real=candidate_real,
                        knockoff_pairs=knockoff_pairs,
                    )
                    output_rows.append(
                        {
                            "expert_id": expert_id,
                            "capability": descriptor["capability"],
                            "scope": descriptor["scope"],
                            "modality": row["modality"],
                            "task": task,
                            "claim_type": args.claim_type,
                            "domain": row["domain"],
                            "group_id": row["group_id"],
                            "outcome_delta": outcome_delta,
                            "real_effect": effect["real_margin"],
                            "knockoff_effect": effect["knockoff_margin"],
                            "differential_effect": effect["differential_effect"],
                            "knockoff_margins": list(effect["knockoff_margins"]),
                            "knockoff_ids": [control["id"] for control in controls],
                            "candidate_method": candidate_name,
                            "transaction_id": transaction.transaction_id,
                            "proposer_expert_id": args.proposer_expert_id,
                            "proposer_expert_ids": list(proposer_ids),
                            "split": str(row.get("split", row.get("role", "source"))),
                            "references_used_post_generation_only": True,
                            "target_test_selection": False,
                        }
                    )
                    expert_counts[expert_id] = expert_counts.get(expert_id, 0) + 1
            pool.reset_case()
    finally:
        for model in pool.models.values():
            close = getattr(model, "close", None)
            if callable(close):
                close()

    if not output_rows:
        raise RuntimeError(
            "no source qualification observations were produced; inspect the skip audit"
        )

    # Preserve transaction-level observations for interaction-aware portfolio
    # fitting.  The per-expert qualification file below is still collapsed to
    # one conservative row per patient/study and expert cell, but portfolio
    # simulation needs experts from the same frozen transaction aligned together.
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    transaction_target = Path(str(target) + ".transactions.jsonl")
    with transaction_target.open("w", encoding="utf-8") as handle:
        for observation in output_rows:
            handle.write(json.dumps(observation, ensure_ascii=False) + "\n")

    # Qualification samples are patient/study groups, not claim count.  A long
    # report may yield several transactions for one expert; counting them
    # independently would create pseudo-replication.
    output_rows = collapse_group_observations(output_rows)

    with target.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "schema": "merit-tx-source-observations-v1",
        "candidate_method": candidate_name,
        "candidate_path": str(candidate_path.resolve()),
        "proposer_expert_ids": list(proposer_ids),
        "n_manifest": len(rows),
        "observations": len(output_rows),
        "transaction_observations_path": str(transaction_target.resolve()),
        "transaction_observations": sum(
            1
            for line in transaction_target.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ),
        "qualification_unit": "unique source group; worst transaction retained within group",
        "expert_counts": dict(sorted(expert_counts.items())),
        "knockoff_controls": knockoff_count,
        "excluded_optional_experts": excluded,
        "skipped": skipped,
        "references_used_only_for_source_outcome_measurement": True,
        "target_test_selection": False,
    }
    target.with_suffix(target.suffix + ".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(target)


if __name__ == "__main__":
    main()
