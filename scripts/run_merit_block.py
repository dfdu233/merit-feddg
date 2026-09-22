from __future__ import annotations

import argparse
import json
from pathlib import Path

from merit_feddg.block_ablation import build_block_ablation_plan
from merit_feddg.block_intervention import (
    BlockInterventionConfig,
    decode_counterfactual_blocks,
    native_expert_block_branches,
)
from merit_feddg.capabilities import EvidenceItem
from merit_feddg.capability_runtime import NativeSession, NativeState, ValueGenerationConfig
from merit_feddg.generalist_factory import load_generalist
from merit_feddg.io import load_experiment_yaml, load_yaml
from merit_feddg.matched_evaluation import generation_prompt, load_manifest


def _read_packets(path):
    result = {}
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        if any(key in row for key in ("answer", "answers", "reference", "references", "label")):
            raise ValueError(f"line {line_number}: evidence packets cannot contain labels")
        sample_id = str(row.get("id") or "")
        if not sample_id or sample_id in result:
            raise ValueError("evidence packets require unique nonempty IDs")
        if not isinstance(row.get("experts"), dict):
            raise TypeError("each evidence packet needs an experts mapping")
        result[sample_id] = row["experts"]
    return result


def _items(values):
    if not isinstance(values, list):
        raise TypeError("evidence item collection must be a list")
    return tuple(EvidenceItem(**value) for value in values)


def _arm(method_config, arm_name):
    plan = build_block_ablation_plan(method_config)
    if arm_name not in plan["arms"]:
        raise ValueError(f"unknown MERIT-Block arm: {arm_name}")
    return plan["arms"][arm_name]


def _branch_packets(experts, arm):
    fixture = str(arm.get("evidence_fixture", "native_real"))
    control_kind = str(arm.get("control_kind", "matched_wrong_patient"))
    control_count = int(arm.get("controls", 0))
    limit = arm.get("expert_limit", "all")

    selected_ids = sorted(experts)
    if limit != "all":
        if type(limit) is not int or limit < 1:
            raise ValueError("expert_limit must be all or a positive integer")
        selected_ids = selected_ids[:limit]

    real_items = {}
    control_items = {}
    for expert_id in selected_ids:
        payload = experts[expert_id]
        if fixture == "native_real":
            raw_real = payload.get("real")
        else:
            raw_real = (payload.get("fixtures") or {}).get(fixture)
        if raw_real is None:
            continue
        real_items[expert_id] = _items(raw_real)

        if control_kind == "none":
            control_items[expert_id] = ()
            continue
        field = (
            "matched_controls"
            if control_kind == "matched_wrong_patient"
            else "random_controls"
        )
        controls = payload.get(field) or ()
        if len(controls) < control_count:
            # Missing controls cause the decoder to abstain through its declared
            # min_controls rule; target images are never substituted.
            chosen = controls
        else:
            chosen = controls[:control_count]
        control_items[expert_id] = tuple(_items(values) for values in chosen)
    return real_items, control_items, control_kind


def _runtime_config(arm, task, *, drift_horizon=0):
    task_budgets = arm.get("task_max_new_tokens", {})
    max_new_tokens = int(task_budgets.get(task, arm["max_new_tokens"]))
    controls = int(arm.get("controls", 0))
    return BlockInterventionConfig(
        max_new_tokens=max_new_tokens,
        block_tokens=int(arm["block_tokens"]),
        block_mode=str(arm["block_mode"]),
        sentence_max_tokens=int(arm["sentence_max_tokens"]),
        decision_rule=str(arm["decision_rule"]),
        context_policy=str(arm["context_policy"]),
        score_reduction=str(arm["score_reduction"]),
        min_controls=max(1, controls),
        numerical_epsilon=float(arm["numerical_epsilon"]),
        drift_probe_horizon=int(drift_horizon),
    )


def _load_existing(path):
    output = Path(path)
    if not output.exists():
        return set()
    ids = set()
    for line in output.read_text(encoding="utf-8").splitlines():
        if line.strip():
            ids.add(json.loads(line)["id"])
    return ids


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run MERIT-Block with frozen real/control EvidenceItem packets. "
            "The generation manifest and packet file must contain no references."
        )
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--evidence-packets", required=True)
    parser.add_argument("--receiver-config", required=True)
    parser.add_argument("--method-config", default="configs/merit_block.yaml")
    parser.add_argument("--arm", default="full")
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--drift-horizon",
        type=int,
        default=0,
        help=(
            "Diagnostic only: after an accepted block, measure same-prefix "
            "persistent-context JSD for this many Generalist-forced steps."
        ),
    )
    args = parser.parse_args()

    receiver = load_experiment_yaml(args.receiver_config)
    method = load_yaml(args.method_config)
    arm = _arm(method, args.arm)
    rows = load_manifest(args.manifest)
    packets = _read_packets(args.evidence_packets)
    missing = [row["id"] for row in rows if row["id"] not in packets]
    if missing:
        raise ValueError(f"missing frozen evidence packets for {len(missing)} case(s)")

    decoder = ValueGenerationConfig(**receiver["capability_value"]["generation"])
    probe = load_generalist(receiver["generalist"], args.artifacts)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = _load_existing(output)

    with output.open("a", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            if row["id"] in completed:
                continue
            prompt = generation_prompt(row, receiver)
            native = NativeSession(
                probe,
                row["image"],
                prompt,
                row["question"],
                decoder,
            )
            base_session = native._evidence_session(NativeState())
            real_items, control_items, control_kind = _branch_packets(
                packets[row["id"]], arm
            )
            if not real_items:
                raise ValueError(f"{row['id']}: no expert evidence for selected arm")
            branches = native_expert_block_branches(
                native,
                real_items=real_items,
                control_items=control_items,
                control_kind=control_kind,
            )
            result = decode_counterfactual_blocks(
                base_session,
                branches,
                config=_runtime_config(
                    arm,
                    row["task"],
                    drift_horizon=args.drift_horizon,
                ),
            )
            record = {
                "id": row["id"],
                "text": result["text"],
                "token_ids": result["token_ids"],
                "trace": result["trace"],
                "seconds": result["seconds"],
                "method": result["method"],
                "arm": args.arm,
                "evidence_fixture": arm.get("evidence_fixture", "native_real"),
                "control_kind": control_kind,
                "expert_ids": sorted(branches),
                "references_read": False,
                "target_labels_used_for_decision": False,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"[{index + 1}/{len(rows)}] {row['id']}", flush=True)


if __name__ == "__main__":
    main()
