import copy
import json

import numpy as np
import pytest

from merit_feddg.capability_value import fit_value_policy, score_value_policy


def rows_for(gains=(-0.6, 0.6), *, domains=("a", "b"), count=8, action="tool"):
    return [
        {"role": "source", "sample_id": f"{domain}-{i}", "group_id": f"image-{domain}-{i}",
         "domain": domain, "state_id": "start", "action_key": action,
         "features": [float(i % len(gains)), 1.0], "gain": gains[i % len(gains)],
         "cost": 0.1, "executed": True, "adopted": True}
        for domain in domains for i in range(count)
    ]


def candidates():
    return [{"action_key": "tool", "features": [value, 1.0]} for value in (0.0, 1.0)]


def test_contextual_continuous_gain_not_whole_card_rejection():
    artifact = fit_value_policy(rows_for(), {"model": "fixed"}, ridge=0.01)
    predicted = score_value_policy(artifact, candidates())
    assert all(row["supported"] for row in predicted)
    assert predicted[0]["score"] < 0 < predicted[1]["score"]
    assert predicted[1]["mean_gain"] == pytest.approx(0.6, abs=0.002)
    assert artifact["source_summary"]["independent_groups"] == 16
    json.dumps(artifact, allow_nan=False)
    json.dumps(predicted, allow_nan=False)


def test_pessimism_uses_unseen_source_residual_not_fitted_residual():
    rows = rows_for((0.8,), domains=("a",)) + rows_for((-0.2,), domains=("b",))
    artifact = fit_value_policy(rows, {})
    robust = score_value_policy(artifact, [candidates()[0]])[0]
    mean = score_value_policy(artifact, [candidates()[0]], robust=False)[0]
    assert mean["score"] == pytest.approx(0.3)
    assert robust["penalty"] == pytest.approx(1.0)
    assert robust["score"] < 0 < mean["score"]
    assert artifact["lodo_models"]["a"]["training_domains"] == ["b"]


def test_each_fold_fits_its_own_feature_normalization():
    rows = rows_for()
    for row in rows:
        row["features"][0] += 100 if row["domain"] == "b" else 0
    artifact = fit_value_policy(rows, {})
    assert artifact["mean_model"]["offset"][0] == pytest.approx(50.5)
    assert artifact["lodo_models"]["a"]["offset"][0] == pytest.approx(100.5)
    assert artifact["lodo_models"]["b"]["offset"][0] == pytest.approx(0.5)


def test_empty_executions_lower_value_but_do_not_inflate_support():
    rows = rows_for((0.5,))
    for row in rows[:7]:
        row.update(adopted=False, gain=-0.8)
    artifact = fit_value_policy(rows, {})
    assert artifact["source_summary"]["executed_records"] == 16
    assert artifact["source_summary"]["empty_executions"] == 7
    card = next(iter(artifact["conditions"].values()))
    assert card["domains"]["a"]["executed_groups"] == 8
    assert card["domains"]["a"]["adopted_groups"] == 1
    assert not score_value_policy(artifact, [candidates()[0]], robust=False)[0]["supported"]
    assert artifact["mean_model"]["coefficients"][0][0] < 0.0


def test_empty_outcomes_kept_after_other_independent_support_is_sufficient():
    rows = rows_for((0.5,), count=16)
    for row in rows:
        if int(row["sample_id"].split("-")[-1]) >= 8:
            row.update(adopted=False, gain=-0.5)
    artifact = fit_value_policy(rows, {})
    result = score_value_policy(artifact, [candidates()[0]], robust=False)[0]
    assert result["supported"] and result["mean_gain"] == pytest.approx(0.0)


def test_runtime_failures_are_reported_not_trained_as_zero():
    rows = rows_for((0.6,))
    failed = copy.deepcopy(rows[0])
    failed.update(sample_id="failed", group_id="failed-image", executed=False, adopted=False, gain=-1)
    artifact = fit_value_policy(rows + [failed], {})
    assert artifact["source_summary"]["runtime_failures"] == 1
    assert score_value_policy(artifact, [candidates()[0]], robust=False)[0]["mean_gain"] == pytest.approx(0.6)


def test_repeated_states_or_questions_do_not_create_independent_cases():
    rows = rows_for((0.5,), count=1)
    repeated = []
    for row in rows:
        for i in range(12):
            extra = copy.deepcopy(row)
            extra.update(state_id=f"state-{i}", sample_id=f"{row['sample_id']}-question-{i}")
            repeated.append(extra)
    artifact = fit_value_policy(repeated, {})
    assert artifact["source_summary"]["independent_groups"] == 2
    assert not score_value_policy(artifact, [candidates()[0]])[0]["supported"]


def test_group_and_domain_balancing_prevents_state_frequency_bias():
    rows = rows_for((0.0,), count=2)
    for row in rows:
        row["gain"] = 1.0 if row["sample_id"].endswith("0") else 0.0
    copies = []
    for i in range(20):
        extra = copy.deepcopy(rows[0])
        extra["state_id"] = f"extra-{i}"
        copies.append(extra)
    artifact = fit_value_policy(rows + copies, {}, min_cases_per_domain=2)
    assert score_value_policy(artifact, [candidates()[0]], robust=False)[0]["mean_gain"] == pytest.approx(0.5)


def test_exact_duplicate_deduplicated_conflicting_duplicate_rejected():
    rows = rows_for()
    artifact = fit_value_policy(rows + [copy.deepcopy(rows[0])], {})
    assert artifact["source_summary"]["duplicates_removed"] == 1
    conflict = copy.deepcopy(rows[0])
    conflict["gain"] = 0.1
    with pytest.raises(ValueError, match="conflicting duplicate"):
        fit_value_policy(rows + [conflict], {})


def test_history_and_stage_must_have_independent_source_support():
    rows = rows_for((0.6,))
    artifact = fit_value_policy(rows, {})
    later = {**candidates()[0], "state_kind": "continuation"}
    after = {**candidates()[0], "state_kind": "after_tool", "history_actions": ["A"]}
    assert score_value_policy(artifact, [later, after])[0]["reason"] == "unseen_state_history"
    after_rows = copy.deepcopy(rows)
    for row in after_rows:
        row.update(state_id="after-a", state_kind="after_tool", history_actions=["A"])
    artifact = fit_value_policy(rows + after_rows, {})
    assert score_value_policy(artifact, [after])[0]["supported"]
    assert not score_value_policy(artifact, [{**after, "history_actions": ["B"]}])[0]["supported"]


def test_history_order_is_not_collapsed():
    rows = rows_for((0.5,))
    for row in rows:
        row.update(state_kind="after_tool", history_actions=["A", "B"])
    artifact = fit_value_policy(rows, {})
    candidate = {**candidates()[0], "state_kind": "after_tool", "history_actions": ["B", "A"]}
    assert score_value_policy(artifact, [candidate])[0]["reason"] == "unseen_state_history"


def test_history_weak_support_cannot_borrow_from_initial_cases():
    rows = rows_for((0.5,))
    extra = copy.deepcopy(rows[0])
    extra.update(state_id="after", state_kind="after_tool", history_actions=["A"])
    artifact = fit_value_policy(rows + [extra], {})
    candidate = {**candidates()[0], "state_kind": "after_tool", "history_actions": ["A"]}
    assert score_value_policy(artifact, [candidate])[0]["reason"] == "insufficient_independent_support"


def test_cost_is_predicted_and_charged_once():
    rows = rows_for((0.6,))
    for row in rows:
        row["cost"] = 2.0
    artifact = fit_value_policy(rows, {}, cost_weight=0.2)
    predicted = score_value_policy(artifact, [candidates()[0]])[0]
    assert predicted["predicted_cost"] == pytest.approx(2.0)
    assert predicted["score"] == pytest.approx(0.2)
    assert predicted["penalty"] == pytest.approx(0.0)


def test_unknown_action_empty_training_and_json_serialization():
    artifact = fit_value_policy([], {})
    result = score_value_policy(artifact, [candidates()[0]])[0]
    assert result["reason"] == "unknown_action" and result["score"] == 0
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("field,value", [
    ("role", "target"), ("role", None), ("gain", float("nan")), ("gain", 1.1),
    ("cost", -1), ("features", [float("inf"), 0]), ("sample_id", ""),
    ("executed", 1), ("adopted", "yes"), ("history_actions", "A"),
    ("state_kind", "unknown"),
])
def test_invalid_source_records_rejected(field, value):
    rows = rows_for()
    rows[0][field] = value
    with pytest.raises((ValueError, TypeError)):
        fit_value_policy(rows, {})


def test_cross_domain_group_and_sample_leakage_rejected():
    rows = rows_for()
    rows[8]["group_id"] = rows[0]["group_id"]
    with pytest.raises(ValueError, match="group crosses"):
        fit_value_policy(rows, {})
    rows = rows_for()
    rows[8]["sample_id"] = rows[0]["sample_id"]
    with pytest.raises(ValueError, match="sample_id appears"):
        fit_value_policy(rows, {})


def test_dimensions_and_prediction_reference_leakage_rejected_without_mutation():
    artifact = fit_value_policy(rows_for(), {})
    before = json.dumps(artifact, sort_keys=True)
    with pytest.raises(ValueError, match="dimension"):
        score_value_policy(artifact, [{"action_key": "tool", "features": [1.0]}])
    with pytest.raises(ValueError, match="outcomes or references"):
        score_value_policy(artifact, [{**candidates()[0], "references": ["answer"]}])
    score_value_policy(artifact, candidates())
    assert json.dumps(artifact, sort_keys=True) == before


def test_artifact_roundtrip_and_predictions_are_deterministic():
    artifact = fit_value_policy(rows_for(), {"source": "sha256"})
    restored = json.loads(json.dumps(artifact, allow_nan=False))
    assert score_value_policy(artifact, candidates()) == score_value_policy(restored, candidates())
    assert np.asarray(artifact["mean_model"]["coefficients"]).shape == (3, 2)
