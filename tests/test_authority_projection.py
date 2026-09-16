import math

import numpy as np
import pytest

from merit_feddg.authority_projection import (
    NEGATIVE,
    POSITIVE,
    UNKNOWN,
    binary_finding_group,
    mapped_positive_mass,
    normalized_pool_distribution,
    project_binary_authority,
    project_free_text_candidates,
    transport_diagnostic,
    xrv_operating_coordinate,
)


def test_operating_coordinate_endpoints_and_threshold():
    assert xrv_operating_coordinate(0.0, 0.2) == 0.0
    assert xrv_operating_coordinate(0.2, 0.2) == pytest.approx(0.5)
    assert xrv_operating_coordinate(1.0, 0.2) == 1.0


def test_operating_coordinate_matches_piecewise_definition():
    assert xrv_operating_coordinate(0.1, 0.2) == pytest.approx(0.25)
    assert xrv_operating_coordinate(0.6, 0.2) == pytest.approx(0.75)


def test_operating_coordinate_rejects_bad_threshold():
    with pytest.raises(ValueError):
        xrv_operating_coordinate(0.3, 0.0)
    with pytest.raises(ValueError):
        xrv_operating_coordinate(0.3, 1.0)


def test_pool_distribution_is_stable():
    p = normalized_pool_distribution([10000.0, 9999.0])
    assert p.sum() == pytest.approx(1.0)
    assert p[0] > p[1]


def test_projection_sets_group_marginal_and_preserves_unknown_mass():
    result = project_binary_authority(
        [0.0, -0.5, -1.0, -2.0],
        [POSITIVE, POSITIVE, NEGATIVE, UNKNOWN],
        0.8,
    )
    assert result["status"] == "projected"
    p = result["base_probabilities"]
    q = result["projected_probabilities"]
    assert q.sum() == pytest.approx(1.0)
    assert q[3] == pytest.approx(p[3])
    assert mapped_positive_mass(q, [POSITIVE, POSITIVE, NEGATIVE, UNKNOWN]) == pytest.approx(0.8)


def test_projection_preserves_within_positive_odds():
    result = project_binary_authority(
        [0.0, -1.0, -2.0],
        [POSITIVE, POSITIVE, NEGATIVE],
        0.7,
    )
    p = result["base_probabilities"]
    q = result["projected_probabilities"]
    assert q[0] / q[1] == pytest.approx(p[0] / p[1])


def test_projection_preserves_within_negative_odds():
    result = project_binary_authority(
        [0.0, -1.0, -2.0],
        [POSITIVE, NEGATIVE, NEGATIVE],
        0.7,
    )
    p = result["base_probabilities"]
    q = result["projected_probabilities"]
    assert q[1] / q[2] == pytest.approx(p[1] / p[2])


def test_projection_kl_is_nonnegative():
    result = project_binary_authority([0.0, -1.0], [POSITIVE, NEGATIVE], 0.9)
    assert result["kl_q_to_p"] >= 0.0


def test_projection_is_identity_when_target_matches_base_mapped_mass():
    scores = [0.0, -1.0, -0.2]
    groups = [POSITIVE, NEGATIVE, UNKNOWN]
    p = normalized_pool_distribution(scores)
    target = mapped_positive_mass(p, groups)
    result = project_binary_authority(scores, groups, target)
    assert np.allclose(result["projected_probabilities"], p)
    assert result["kl_q_to_p"] == pytest.approx(0.0, abs=1e-12)


def test_projection_reports_missing_positive_side():
    result = project_binary_authority([0.0, -1.0], [NEGATIVE, UNKNOWN], 0.8)
    assert result["status"] == "infeasible_missing_semantic_side"
    assert np.allclose(result["projected_probabilities"], result["base_probabilities"])


def test_projection_reports_missing_negative_side():
    result = project_binary_authority([0.0, -1.0], [POSITIVE, UNKNOWN], 0.2)
    assert result["status"] == "infeasible_missing_semantic_side"


def test_free_text_selection_returns_original_text_verbatim():
    candidates = ["No pleural effusion is seen.", "A small pleural effusion is present."]
    result = project_free_text_candidates(
        candidates,
        [0.0, -0.2],
        [NEGATIVE, POSITIVE],
        0.95,
    )
    assert result["selected_text"] in candidates
    assert result["selected_text"] == candidates[result["selected_index"]]
    assert result["free_text_output_preserved"] is True


def test_binary_mapper_leading_yes_no():
    aliases = ["pneumothorax"]
    assert binary_finding_group("Yes, a small pneumothorax is present.", aliases) == POSITIVE
    assert binary_finding_group("No, there is no pneumothorax.", aliases) == NEGATIVE


@pytest.mark.parametrize(
    "text",
    [
        "There is no pleural effusion.",
        "No evidence of pleural effusion is seen.",
        "Pleural effusion is absent.",
        "The radiograph is negative for pleural effusion.",
        "The image does not show pleural effusion.",
    ],
)
def test_binary_mapper_negative_free_text(text):
    assert binary_finding_group(text, ["pleural effusion", "effusion"]) == NEGATIVE


def test_binary_mapper_positive_free_text():
    assert binary_finding_group(
        "There is a small right pleural effusion with adjacent atelectasis.",
        ["pleural effusion", "effusion"],
    ) == POSITIVE


def test_binary_mapper_requires_alias_without_leading_answer():
    assert binary_finding_group("The study is otherwise unchanged.", ["cardiomegaly"]) == UNKNOWN


def test_binary_mapper_handles_cardiomegaly_alias():
    aliases = ["cardiomegaly", "cardiac enlargement", "enlarged heart"]
    assert binary_finding_group("The enlarged heart remains unchanged.", aliases) == POSITIVE
    assert binary_finding_group("There is no cardiac enlargement.", aliases) == NEGATIVE


def test_transport_diagnostic_detects_wrong_direction():
    groups = [NEGATIVE, POSITIVE]
    result = transport_diagnostic([0.0, -0.1], [-1.0, 0.0], groups, 0.05)
    assert result["status"] == "measured"
    assert result["toward_source"] is False
    assert result["distance_after"] > result["distance_before"]


def test_transport_diagnostic_detects_right_direction():
    groups = [NEGATIVE, POSITIVE]
    result = transport_diagnostic([0.0, -0.1], [0.0, 1.0], groups, 0.9)
    assert result["toward_source"] is True
    assert result["distance_after"] < result["distance_before"]


def test_mapped_positive_mass_ignores_unknown_conditionally():
    p = np.asarray([0.2, 0.3, 0.5])
    assert mapped_positive_mass(p, [POSITIVE, NEGATIVE, UNKNOWN]) == pytest.approx(0.4)


def test_invalid_group_is_rejected():
    with pytest.raises(ValueError):
        project_binary_authority([0.0, 0.0], [POSITIVE, "maybe"], 0.5)


def test_boolean_source_mass_is_rejected():
    with pytest.raises(TypeError):
        project_binary_authority([0.0, 0.0], [POSITIVE, NEGATIVE], True)


def test_zero_and_one_targets_are_supported():
    zero = project_binary_authority([0.0, 0.0], [POSITIVE, NEGATIVE], 0.0)
    one = project_binary_authority([0.0, 0.0], [POSITIVE, NEGATIVE], 1.0)
    assert zero["projected_probabilities"][0] == pytest.approx(0.0)
    assert one["projected_probabilities"][1] == pytest.approx(0.0)
    assert math.isfinite(zero["kl_q_to_p"])
    assert math.isfinite(one["kl_q_to_p"])
