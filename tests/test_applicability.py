import copy

import numpy as np
import pytest

from merit_feddg.applicability import (
    ApplicabilityConfig,
    ApplicabilityGate,
    build_memory,
    local_risk,
)
from merit_feddg.capabilities import scoped_key


def records(expert="A", domains=4, losses=(0., .8)):
    result = []
    for domain in range(domains):
        for cluster in range(2):
            for n in range(2):
                identity = f"{domain}-{cluster}-{n}"
                result.append({"id": identity, "role": "source", "domain": f"d{domain}",
                    "domain_kind": "independent", "group_id": identity, "image_sha256": identity,
                    "scope_key": scoped_key(expert, "pathology", "vqa", "classification", "tissue"),
                    "feature": [1., 0.] if cluster == 0 else [0., 1.], "loss": losses[cluster]})
    return result


class Pool:
    def __init__(self, feature=(1., 0.)):
        self.feature, self.calls = feature, 0
    def domain_embedding(self, expert, image):
        self.calls += 1
        return self.feature


def query():
    return {"id": "q", "image": "q.png", "image_sha256": "q", "group_id": "q",
            "domain": "new", "modality": "pathology", "task": "vqa"}


def descriptor(expert="A"):
    return {"expert": expert, "capability": "classification", "scope": "tissue"}


def test_input_specific_local_risk():
    cfg = ApplicabilityConfig()
    good = local_risk(records(), [1, 0], cfg)
    bad = local_risk(records(), [0, 1], cfg)
    assert good["supported"] and bad["supported"]
    assert good["worst"] == 0 and bad["worst"] == pytest.approx(.8)


def test_same_image_different_experts_and_cache():
    memory = build_memory(records()+records("B", losses=(.9, .9)), ApplicabilityConfig(), {}, "test")
    pool = Pool()
    gate = ApplicabilityGate(memory, pool)
    assert gate.decide(query(), descriptor())["allowed"]
    assert not gate.decide(query(), descriptor("B"))["allowed"]
    gate.decide(query(), descriptor())
    assert pool.calls == 2


def test_distance_ablation_ignores_loss_but_local_does_not():
    memory = build_memory(records(losses=(.8, .8)), ApplicabilityConfig(), {}, "test")
    assert ApplicabilityGate(memory, Pool(), mode="distance").decide(query(), descriptor())["allowed"]
    assert not ApplicabilityGate(memory, Pool(), mode="local").decide(query(), descriptor())["allowed"]


def test_global_ablation_does_not_encode_query():
    pool = Pool()
    memory = build_memory(records(), ApplicabilityConfig(), {}, "test")
    result = ApplicabilityGate(memory, pool, mode="global").decide(query(), descriptor())
    assert result["risk"] == pytest.approx(.4) and pool.calls == 0


def test_remote_feature_has_insufficient_support():
    cfg = ApplicabilityConfig()
    result = local_risk(records(), [-1, 0], cfg)
    assert not result["supported"]


def test_lodo_excludes_whole_domain():
    data = records()
    for item in data:
        item["loss"] = 1. if item["domain"] == "d0" else 0.
    memory = build_memory(data, ApplicabilityConfig(), {}, "test")
    fold = memory["lodo"][data[0]["scope_key"]]["held_out"]
    d0 = [row for row in fold if row["domain"] == "d0"]
    assert all(row["predicted_risk"] == 0 and row["observed_loss"] == 1 for row in d0)


def test_source_crossfit_penalty_cannot_use_query_domain_labels():
    original = records()
    changed = copy.deepcopy(original)
    for item in changed:
        if item["domain"] == "d0":
            item["loss"] = .99
    q = {**query(), "domain": "d0"}
    outcomes = []
    for data in (original, changed):
        memory = build_memory(data, ApplicabilityConfig(), {}, "test")
        outcomes.append(ApplicabilityGate(memory, Pool(), exclude_domain=True).decide(q, descriptor()))
    assert outcomes[0] == outcomes[1]


def test_two_domains_cannot_calibrate_two_domain_lodo_penalty():
    memory = build_memory(records(domains=2), ApplicabilityConfig(), {}, "test")
    assert list(memory["penalties"].values()) == [None]
    result = ApplicabilityGate(memory, Pool()).decide(query(), descriptor())
    assert not result["allowed"]


@pytest.mark.parametrize("change", [{"role": "target"}, {"loss": float("nan")},
                                    {"loss": 1.1}, {"domain_kind": "proxy"}])
def test_reject_invalid_calibration(change):
    data = records()
    data[0].update(change)
    with pytest.raises(ValueError):
        build_memory(data, ApplicabilityConfig(), {}, "test")


def test_no_duplicate_patient_support():
    data = records()
    data[1]["group_id"] = data[0]["group_id"]
    with pytest.raises(ValueError, match="duplicate"):
        build_memory(data, ApplicabilityConfig(), {}, "test")


def test_missing_scope_does_not_load_expert():
    memory = build_memory(records(), ApplicabilityConfig(), {}, "test")
    pool = Pool()
    result = ApplicabilityGate(memory, pool).decide(query(), descriptor("missing"))
    assert not result["allowed"] and pool.calls == 0


def test_unit_embedding_dimension_validation():
    with pytest.raises(ValueError, match="dimension"):
        local_risk(records(), [1, 0, 0], ApplicabilityConfig())
    assert np.isfinite(local_risk(records(), [1, 0], ApplicabilityConfig())["mean"])
