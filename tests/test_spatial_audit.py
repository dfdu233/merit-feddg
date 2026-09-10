import importlib.util
import json
from pathlib import Path


def test_audit_distinguishes_nonzero_features_from_changed_answers(tmp_path):
    spec = importlib.util.spec_from_file_location("spatial_audit", Path("scripts/audit_spatial_run.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path / "protocol.json").write_text(json.dumps({"identity": "run", "methods": ["generalist", "spatial_gate"]}))
    baseline = {"1": {"text": "same", "seconds": 1}}
    (tmp_path / "generalist.json").write_text(json.dumps(baseline))
    output = {"1": {"text": "same", "seconds": 2, "input_modality": "ct", "trace": [
        {"event": "tool", "expert": "A", "adopted": True, "reason": "ok",
         "vector_gate": {"accepted": True, "seconds": .2, "verifier_queries": 4}},
        {"event": "decode", "evidence_transport": {
            "presented": [{"expert_id": "A", "evidence_id": "e"}],
            "spatial_fusion": {"max_abs_token_delta": .1}}}]}}
    (tmp_path / "spatial_gate.json").write_text(json.dumps(output))
    audit = module.audit_run(tmp_path)
    arm = audit["arms"]["spatial_gate"]
    assert arm["cases_with_nonzero_token_intervention"] == 1
    assert arm["text_changed_vs_generalist"] == 0
    assert arm["presented"] == {"A": 1} and arm["gate_accepted"] == 1
    assert not audit["clinical_gain_measured"] and not audit["labels_loaded"]
