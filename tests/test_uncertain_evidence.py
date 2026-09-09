import copy
import json
from dataclasses import replace

import pytest

from merit_feddg.capabilities import EvidenceItem
from merit_feddg.capability_runtime import NativeSession, NativeState, ValueGenerationConfig
from merit_feddg.evidence_need import evidence_memory
from merit_feddg.uncertain_evidence import (
    attach_alternatives,
    compile_uncertain_evidence,
    shared_observations,
)


def finding(score=.8, label="native-label"):
    return {"findings": [{"finding": label, "score": score}],
            "score_semantics": "uncalibrated_independent_sigmoid"}


def item(payload=None, capability="classification"):
    return EvidenceItem("e1", "expert", capability, "scope", payload or finding())


def packet(evidence):
    return compile_uncertain_evidence([evidence], "unseen arbitrary question", 10000)


def test_no_uncertainty_is_not_zero_uncertainty():
    p = packet(item())[0]["payload"]
    assert p["uncertainty"]["status"] == "unknown"
    assert not p["uncertainty"]["truth_coverage_guaranteed"]
    assert p["observations"][0]["polarity"] == "unknown"


def test_ranges_widen_and_original_always_participates():
    original = item()
    saved = copy.deepcopy(original)
    p = packet(attach_alternatives(original, [finding(.3), finding(.6)]))[0]["payload"]
    assert p["observations"][0]["observed_value_ranges"]["value"] == {
        "lower": .3, "upper": .8}
    assert original == saved
    assert not p["uncertainty"]["domain_applicability_certified"]


def test_missing_or_conflicting_semantics_do_not_fallback():
    assert packet(attach_alternatives(item(), [{}])) == []
    assert packet(attach_alternatives(item(), [finding(.99, "another-label")])) == []


def test_duplicate_objects_are_not_falsely_associated():
    n = {"kind": "object", "observation": "same-class", "value": .3}
    assert shared_observations([[n], [n, n]]) == []


@pytest.mark.parametrize("extra", [.01, .2, .9, .99])
def test_monotonic_range_property(extra):
    def n(v):
        return {"kind": "finding_score", "observation": "label", "value": v}
    a = shared_observations([[n(.4)], [n(.6)]])[0]
    b = shared_observations([[n(.4)], [n(.6)], [n(extra)]])[0]
    ar, br = a["observed_value_ranges"]["value"], b["observed_value_ranges"]["value"]
    assert br["lower"] <= ar["lower"] <= ar["upper"] <= br["upper"]


def test_spatial_ranges_are_not_lesion_masks():
    def payload(x):
        return {"structures": [{"anatomical_structure": "native-organ",
                                 "bbox_xyxy_normalized_original_image": [x, .1, .8, .9]}]}
    p = packet(attach_alternatives(item(payload(.1), "segmentation"), [payload(.2)]))[0]
    node = p["payload"]["observations"][0]
    assert node["kind"] == "anatomy_region"
    assert node["observed_value_ranges"]["bbox_xyxy_normalized"]["upper"][0] == .2
    assert "do not establish a diagnosis" in str(p)


def test_generation_conflict_does_not_invent_summary():
    original = item({"generated_text": "statement one"}, "generation")
    assert packet(attach_alternatives(original, [{"generated_text": "statement two"}])) == []


def test_no_calibration_claim_or_recursive_payload():
    with pytest.raises(ValueError):
        attach_alternatives(item(), [finding()], origin="calibrated")
    with pytest.raises(ValueError):
        attach_alternatives(item(), [{"native_uncertainty": {}}])
    with pytest.raises(ValueError):
        attach_alternatives(item(), [])


def test_budget_is_whole_packet_not_truncated_range():
    e = attach_alternatives(item(), [finding(.1)])
    assert compile_uncertain_evidence([e], "q", 2) == []
    assert json.loads(json.dumps(packet(e))) == packet(e)


def test_context_path_and_no_visual_bypass():
    config = ValueGenerationConfig(evidence_style="uncertainty", visual_views=0,
                                   max_evidence_chars=10000)
    session = NativeSession(None, "original.png", "question prompt", "q", config)
    assert session.context(NativeState())[1] == "question prompt"
    e = attach_alternatives(item(), [finding(.1)])
    images, prompt = session.context(NativeState(items=(e,)))
    assert "original.png" in str(images)
    assert "observed_value_ranges" in prompt
    assert not session.view_metadata
    with pytest.raises(ValueError, match="raw-mask bypass"):
        replace(config, visual_views=1)
    assert evidence_memory([e], "q", config) == compile_uncertain_evidence([e], "q", 10000)


def test_real_probe_api_exports_outputs_without_claiming_calibration():
    import numpy as np
    from PIL import Image

    from merit_feddg.behavior_probe import probe_xrv

    class Model:
        calls = 0

        def classify(self, image):
            self.calls += 1
            return ["a", "b"], np.array([self.calls / 10, .8]), "same-transform"

    model = Model()
    audit = probe_xrv(model, Image.new("RGB", (8, 8)), "classification")
    assert model.calls == audit["extra_model_calls"] == 3
    assert len(audit["native_alternatives"]) == 3
    assert audit["native_alternatives"][2]["findings"][0]["score"] == .3
    assert not audit["clinical_invariance_verified"]


def test_probe_attachment_requires_explicit_audit():
    with pytest.raises(ValueError, match="audit"):
        ValueGenerationConfig(uncertainty_from_probe=True)
