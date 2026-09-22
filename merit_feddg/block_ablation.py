from __future__ import annotations

from copy import deepcopy


def build_block_ablation_plan(config):
    """Build a frozen one-factor-at-a-time MERIT-Block experiment plan."""
    if "merit_block" not in config or "ablation_axes" not in config:
        raise ValueError("config must contain merit_block and ablation_axes")
    main = deepcopy(config["merit_block"])
    axes = config["ablation_axes"]
    required = {
        "granularity",
        "locality",
        "control",
        "decision",
        "control_count",
        "sequence_score",
        "expert_count",
        "evidence_condition",
    }
    missing = required - set(axes)
    if missing:
        raise ValueError(f"missing ablation axes: {sorted(missing)}")

    arms = {
        "full": {
            **main,
            "role": "primary_method",
            "changed_axis": None,
        }
    }
    for axis_name in (
        "granularity",
        "locality",
        "control",
        "decision",
        "control_count",
        "sequence_score",
        "expert_count",
        "evidence_condition",
    ):
        for override in axes[axis_name]:
            if "name" not in override:
                raise ValueError(f"{axis_name} ablation is missing name")
            arm_name = f"{axis_name}__{override['name']}"
            arm = deepcopy(main)
            arm.update(
                {key: value for key, value in override.items() if key != "name"}
            )
            arm["role"] = "one_factor_ablation"
            arm["changed_axis"] = axis_name
            arm["ablation_name"] = override["name"]
            arms[arm_name] = arm

    return {
        "schema": "merit-block-ablation-plan-v1",
        "primary_arm": "full",
        "selection_on_target_forbidden": True,
        "one_factor_at_a_time": True,
        "arms": arms,
        "required_reports": {
            "trajectory_drift": [
                "mean_same_prefix_js",
                "greedy_disagreement_rate",
                "downstream_divergence_by_intervention_position",
            ],
            "expert_condition": [
                "accept_rate_correct",
                "accept_rate_plausible_wrong",
                "accept_rate_irrelevant",
                "accept_rate_shuffled",
            ],
            "task": ["CE", "OE", "report"],
            "efficiency": [
                "generalist_forwards",
                "expert_branch_forwards",
                "control_forwards",
                "wall_seconds",
            ],
        },
    }
