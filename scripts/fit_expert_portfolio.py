"""Fit interaction-aware MERIT-Tx expert portfolios on source data only.

The input is the transaction-level sidecar produced by
scripts/build_merit_tx_source_observations.py.  Per-expert qualification is a
prerequisite but not the final selection criterion: all feasible independent
verifier subsets are replayed over the same frozen source transactions so that
redundancy and harmful interactions are measured explicitly.

The empty portfolio is allowed and is selected whenever no non-empty subset has
positive conservative utility under the frozen harm constraints.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from merit_feddg.expert_policy import load_qualification_cards
from merit_feddg.expert_portfolio import fit_expert_portfolios, require_disjoint_source_groups
from merit_feddg.io import load_experiment_yaml


def read_rows(path):
    rows=[]
    for line_number,line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(),1):
        if not line.strip():
            continue
        row=json.loads(line)
        if str(row.get("split","source")).casefold() in {"test","target","evaluation","eval"}:
            raise ValueError(f"line {line_number}: target/test observations are forbidden")
        if row.get("target_test_selection") is not False:
            raise ValueError(
                f"line {line_number}: transaction observation must explicitly forbid target selection"
            )
        rows.append(row)
    if not rows:
        raise ValueError("portfolio observation file is empty")
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",required=True)
    parser.add_argument("--qualification-cards",required=True)
    parser.add_argument("--config",default="configs/merit_tx.yaml")
    parser.add_argument("--output",required=True)
    parser.add_argument("--z",type=float,default=1.96)
    args=parser.parse_args()
    if not math.isfinite(args.z) or args.z<=0:
        parser.error("--z must be positive and finite")

    config=load_experiment_yaml(args.config)
    specs=dict(config["experts"])
    specs.pop("source_cases",None)
    qualification_payload=json.loads(Path(args.qualification_cards).read_text(encoding="utf-8"))
    qualification_groups=set(str(value) for value in qualification_payload.get("source_group_ids", ()))
    if not qualification_groups:
        raise ValueError("v3 qualification cards must record source_group_ids")
    cards=load_qualification_cards(args.qualification_cards)
    policy=dict(config.get("merit_tx",{}))
    required={
        "qualification_min_domains",
        "qualification_max_harm_ucb",
        "qualification_min_support_precision_lcb",
        "qualification_min_veto_precision_lcb",
        "qualification_min_consequential",
        "min_support_groups",
        "require_independent_validator",
        "reject_on_qualified_contradiction",
        "portfolio_max_experts",
        "portfolio_min_actions",
        "portfolio_min_domains",
        "portfolio_max_harm_ucb",
        "portfolio_min_precision_lcb",
    }
    missing=sorted(required-set(policy))
    if missing:
        raise ValueError(f"MERIT-Tx portfolio policy missing fields: {missing}")

    input_rows=read_rows(args.input)
    portfolio_groups={str(row["group_id"]) for row in input_rows}
    require_disjoint_source_groups(
        qualification=qualification_groups,
        portfolio_selection=portfolio_groups,
    )
    payload=fit_expert_portfolios(
        input_rows,
        specs=specs,
        qualification_cards=cards,
        policy=policy,
        z=args.z,
    )
    payload["qualification_cards"]=str(Path(args.qualification_cards).resolve())
    payload["qualification_schema"]="merit-expert-qualification-v3"
    payload["qualification_source_groups_sha256"]=qualification_payload.get("source_groups_sha256")
    source_group_ids=sorted(portfolio_groups)
    payload["source_group_ids"]=source_group_ids
    payload["source_groups_sha256"]=hashlib.sha256(
        json.dumps(source_group_ids,separators=(",",":"),ensure_ascii=False).encode()
    ).hexdigest()
    payload["source_groups_disjoint_from_qualification"]=True
    payload["input"]=str(Path(args.input).resolve())
    payload["target_test_selection"]=False

    target=Path(args.output)
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(
        json.dumps(payload,ensure_ascii=False,indent=2)+"\n",
        encoding="utf-8",
    )
    selected={
        f"{row['modality']}|{row['task']}|{row['claim_type']}":
        row["selected"]["expert_ids"]
        for row in payload["buckets"]
    }
    print(json.dumps(selected,ensure_ascii=False,sort_keys=True))


if __name__=="__main__":
    main()
