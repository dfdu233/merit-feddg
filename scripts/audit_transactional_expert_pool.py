"""Answer-blind audit of the literature-grounded MERIT-Tx expert pool."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from merit_feddg.capability_study import _filter_optional_experts
from merit_feddg.expert_policy import load_qualification_cards
from merit_feddg.io import load_experiment_yaml
from merit_feddg.matched_evaluation import load_manifest
from merit_feddg.transactional_protocol import plan_transaction_experts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", default="configs/merit_tx.yaml")
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--qualification-cards")
    parser.add_argument("--max-calls", type=int, default=6)
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.max_calls < 1:
        parser.error("--max-calls must be positive")

    config = load_experiment_yaml(args.config)
    specs, excluded = _filter_optional_experts(config["experts"], args.artifacts)
    specs.pop("source_cases", None)
    configured_cards = config.get("merit_tx", {}).get("qualification_cards")
    card_path = args.qualification_cards or configured_cards
    if card_path and Path(card_path).is_file():
        cards = load_qualification_cards(card_path)
        card_status = "loaded"
    else:
        cards = {}
        card_status = "missing"
    rows = load_manifest(args.manifest, include_answer_type=False)
    tx_policy = config.get("merit_tx", {})
    qualification_min_domains = int(tx_policy.get("qualification_min_domains", 2))
    qualification_max_harm_ucb = float(tx_policy.get("qualification_max_harm_ucb", 0.25))
    qualification_min_specificity_lcb = float(
        tx_policy.get("qualification_min_specificity_lcb", 0.5)
    )

    role_counts = Counter()
    commit_role_counts = Counter()
    group_hist = Counter()
    patient_hist = Counter()
    modality = defaultdict(Counter)
    cases = []
    for row in rows:
        plan = plan_transaction_experts(
            row,
            specs,
            qualification_cards=cards,
            max_calls=args.max_calls,
            qualification_min_domains=qualification_min_domains,
            qualification_max_harm_ucb=qualification_max_harm_ucb,
            qualification_min_specificity_lcb=qualification_min_specificity_lcb,
        )
        audit = plan["audit"]
        groups = sorted({entry["fault_group"] for entry in audit})
        patient_groups = sorted(
            {
                entry["fault_group"]
                for entry in audit
                if entry["patient_specific"]
            }
        )
        authorized = sorted(
            {
                entry["fault_group"]
                for entry in audit
                if entry["commit_authorized"]
            }
        )
        for entry in audit:
            role_counts[entry["evidence_role"]] += 1
            if entry["commit_authorized"]:
                commit_role_counts[entry["evidence_role"]] += 1
        bucket = str(len(groups)) if len(groups) < 4 else "4+"
        pbucket = str(len(patient_groups)) if len(patient_groups) < 4 else "4+"
        group_hist[bucket] += 1
        patient_hist[pbucket] += 1
        modality[row["modality"]]["n"] += 1
        modality[row["modality"]][f"fault_groups_{bucket}"] += 1
        modality[row["modality"]][f"patient_groups_{pbucket}"] += 1
        modality[row["modality"]]["commit_authorized_cases"] += int(bool(authorized))
        cases.append(
            {
                "id": row["id"],
                "modality": row["modality"],
                "task": row["task"],
                "selected": audit,
                "fault_groups": groups,
                "patient_specific_fault_groups": patient_groups,
                "commit_authorized_fault_groups": authorized,
                "has_commit_authority": bool(authorized),
            }
        )

    payload = {
        "schema": "merit-tx-expert-pool-audit-v1",
        "n": len(rows),
        "selection_principle": (
            "literature-grounded capability complementarity; patient-specific "
            "verification before proposal/knowledge; commit authority source-only"
        ),
        "qualification_cards_loaded": bool(cards),
        "qualification_policy": {
            "min_domains": qualification_min_domains,
            "max_harm_ucb": qualification_max_harm_ucb,
            "min_specificity_lcb": qualification_min_specificity_lcb,
        },
        "qualification_card_status": card_status,
        "qualification_card_path": str(card_path) if card_path else None,
        "fault_group_histogram": dict(sorted(group_hist.items())),
        "patient_specific_fault_group_histogram": dict(sorted(patient_hist.items())),
        "selected_role_counts": dict(sorted(role_counts.items())),
        "commit_authorized_role_counts": dict(sorted(commit_role_counts.items())),
        "by_modality": {key: dict(value) for key, value in sorted(modality.items())},
        "excluded_optional_experts": excluded,
        "cases": cases,
        "target_answers_used": False,
        "references_used": False,
        "expert_inference_executed": False,
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
