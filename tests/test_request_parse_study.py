"""Synthetic semantics/interface checks, not a real-parser accuracy benchmark."""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from merit_feddg.request_parse_study import (
    audit_parse, canonicalize, catalog_for, compile_request, digest, fidelity,
    parse_response, parser_prompt, select_questions,
)


def spec():
    return {"authority_contract": {"supports": ["finding_presence"],
        "entity_aliases": {"Effusion": ["pleural effusion"], "Pneumothorax": ["pneumothorax"]}}}


def response(atoms=None, relations=None, status="parsed", unresolved=None):
    return json.dumps({"status": status, "atoms": atoms or [],
                       "relations": relations or [], "unresolved": unresolved or []})


def atom(entity="pneumothorax", attribute="finding_presence", cue="Is there"):
    return {"entity": entity, "attribute": attribute, "cue": cue}


Q = "Is there pneumothorax or tuberculosis?"
ROW = {"id": "synthetic-1", "question": Q, "modality": "cxr", "task": "open_vqa"}


def test_unknown_entity_is_not_dropped_by_schema_validation():
    p = parse_response(response([atom(), atom("tuberculosis")]), Q)
    assert p["valid"] and len(p["parse"]["atoms"]) == 2
    assert not p["semantic_completeness_verified"]
    assert canonicalize("tuberculosis", spec()) == ("tuberculosis", "unmapped")


def test_schema_validation_cannot_certify_no_omission():
    gold = parse_response(response([atom(), atom("tuberculosis")]), Q)
    omitted = parse_response(response([atom()]), Q)
    assert omitted["valid"]  # Important: no fictitious completeness guarantee.
    metric = fidelity(omitted, gold, spec())
    assert not metric["exact"] and metric["gold_count"] == 2 and metric["tp"] == 1
    assert metric["missing_atoms"] == [("tuberculosis", "finding_presence")]


@pytest.mark.parametrize("raw", [
    "not json", "```json\n{}\n```", "[]", "{}",
    '{"status":"parsed","status":"unknown","atoms":[],"relations":[],"unresolved":[]}',
    response(), response([atom("invented")]), response([atom(cue="invented")]),
    response([atom(attribute="clinical_truth")]), response([atom(), atom()]),
    response([atom()], unresolved=["tuberculosis"]),
    response(status="unknown", unresolved=["invented"]),
    '{"status":"parsed","atoms":{},"relations":[],"unresolved":[]}',
])
def test_invalid_outputs_are_explicit_failures(raw):
    p = parse_response(raw, Q)
    assert not p["valid"] and p["parse"] is None


def test_relation_is_grounded_and_ordered():
    q = "Where is the tumor relative to the heart?"
    relation = {"subject": "tumor", "predicate": "relative_location", "object": "heart", "cue": "relative to"}
    p = parse_response(response(relations=[relation]), q)
    assert p["valid"] and p["parse"]["relations"][0]["subject"] == "tumor"
    backwards = dict(relation, subject="heart", object="tumor")
    wrong = parse_response(response(relations=[backwards]), q)
    assert wrong["valid"] and not fidelity(wrong, p, spec())["exact"]


def test_catalog_is_ablated_not_scores_or_images():
    card = catalog_for({**spec(), "checkpoint_path": "private.pt", "prediction": 0.9})
    a, b = parser_prompt(Q), parser_prompt(Q, catalog=card)
    assert "expert_catalog_context" not in a and "expert_catalog_context" in b
    assert "private.pt" not in b and '0.9' not in b
    assert "tuberculosis" in a and "tuberculosis" in b


def test_exact_alias_mapping_and_ambiguity():
    assert canonicalize("pleural effusion", spec()) == ("Effusion", "matched")
    s = {"authority_contract": {"entity_aliases": {"Left Lung": ["lung"], "Right Lung": ["lung"]}}}
    assert canonicalize("lung", s) == ("lung", "ambiguous")
    assert canonicalize("effusions", spec())[1] == "unmapped"  # No hidden fuzzy heuristic.


def test_selection_is_order_independent_and_answer_blind():
    rows = [dict(ROW, id=str(i), answer="should never be copied", image="private.png") for i in range(8)]
    chosen = select_questions(rows, 5)
    assert chosen == select_questions(list(reversed(rows)), 5)
    assert all(set(r) == {"id", "question", "modality", "task"} for r in chosen)
    assert chosen == select_questions([dict(r, answer="different") for r in rows], 5)


@pytest.mark.parametrize("limit", [0, 101, True, 1.2])
def test_invalid_limits(limit):
    with pytest.raises(ValueError):
        select_questions([ROW], limit)


def test_duplicate_ids_rejected():
    with pytest.raises(ValueError):
        select_questions([ROW, ROW])


def test_all_unknown_is_not_a_successful_parse_for_answerable_questions():
    gold = parse_response(response([atom()]), Q)
    pred = parse_response(response(status="unknown", unresolved=["pneumothorax"]), Q)
    m = fidelity(pred, gold, spec())
    assert not m["exact"] and m["tp"] == 0 and m["gold_count"] == 1


def test_frozensets_and_maps_have_stable_digest():
    assert digest({"a": 1, "b": 2}) == digest({"b": 2, "a": 1})


def load_script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / (name + ".py")
    module_spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def test_offline_evaluator_requires_review_and_all_predictions():
    evaluator = load_script("evaluate_request_parse_study")
    p = response([atom(), atom("tuberculosis")])
    frozen = {"identity": "fixture", "questions": [ROW], "specs": {"xrv": spec()}}
    records = [{"id": ROW["id"], "question": Q, "expert": "xrv", "mode": mode,
                "output": {"text": p}} for mode in ("question_first", "catalog_conditioned")]
    predictions = {"identity": "fixture", "records": records}
    gold = [{"id": ROW["id"], "question": Q, "reviewed": True, "reviewer": "synthetic fixture",
             "gold_parse": json.loads(p)}]
    report = evaluator.evaluate(frozen, predictions, gold)
    assert all(m["exact_rate"] == 1 for m in report["metrics"].values())
    assert not report["medical_accuracy_evaluated"]
    for bad_gold in ([], [dict(gold[0], reviewed=False)]):
        with pytest.raises(ValueError):
            evaluator.evaluate(frozen, predictions, bad_gold)
    with pytest.raises(ValueError):
        evaluator.evaluate(frozen, dict(predictions, records=records[:1]), gold)


@pytest.fixture
def admission_stub(monkeypatch):
    """Local unit double only; genuine integration checks follow separately."""
    class Request:
        def __init__(self, question, entities, dimensions, modality, task, complete):
            self.question, self.entities = question, entities
            self.dimensions, self.complete = dimensions, complete
        def to_json(self):
            return {"entities": self.entities, "dimensions": sorted(self.dimensions)}
    monkeypatch.setitem(sys.modules, "merit_feddg.evidence_admission", SimpleNamespace(EvidenceRequest=Request))


def test_bridge_preserves_unknown_and_never_drops_one_side(admission_stub):
    p = parse_response(response([atom(), atom("tuberculosis")]), Q)
    req, reason = compile_request(p, ROW, spec())
    assert req.entities == ("Pneumothorax", "tuberculosis")
    assert reason == "model_asserted_complete_not_verified"


def test_bridge_rejects_relations_without_geometry_operator(admission_stub):
    q = "Where is the tumor relative to the heart?"
    rel = {"subject": "tumor", "predicate": "relative_location", "object": "heart", "cue": "relative to"}
    p = parse_response(response(relations=[rel]), q)
    assert compile_request(p, dict(ROW, question=q), spec()) == (None, "relation_operator_not_implemented")


def test_real_bridge_to_existing_admission():
    pytest.importorskip("merit_feddg.evidence_admission")
    from merit_feddg.capabilities import EvidenceItem
    real_spec = {**spec(), "adapter": "xrv_classification", "modalities": ["cxr"],
                 "tasks": ["open_vqa"], "scope": "cxr_findings", "capabilities": ["classification"]}
    real_spec["authority_contract"] = {**spec()["authority_contract"],
        "native_variable": {"entity_type": "finding", "attribute": "presence", "values": ["present", "absent"],
                            "output_semantics": "independent_sigmoid_score"}}
    item = EvidenceItem("e", "xrv", "classification", "cxr_findings",
        {"findings": [{"finding": "Pneumothorax", "score": 0.3}],
         "score_semantics": "uncalibrated_independent_sigmoid"})
    parsed = parse_response(response([atom(), atom("tuberculosis")]), Q)
    audit = audit_parse(parsed, ROW, "xrv", real_spec, [item])
    assert audit["delivery_audit"]["delivered_count"] == 0
    assert not audit["active_intervention"]


def test_token_only_backend_uses_no_image_and_keeps_audit_tokens():
    torch = pytest.importorskip("torch")
    runner = load_script("run_request_parse_study")
    calls = []
    class Conv:
        roles = ("USER", "ASSISTANT")
        def copy(self):
            return self
        def append_message(self, role, text):
            pass
        def get_prompt(self):
            return "prompt"
    class Tokenizer:
        eos_token_id = 2
        def __call__(self, *args, **kwargs):
            return {"input_ids": torch.tensor([[1, 3]]), "attention_mask": torch.ones(1, 2)}
        def decode(self, ids, **kwargs):
            return "{}"
    class Model:
        config = SimpleNamespace(max_position_embeddings=4096)
        def get_input_embeddings(self):
            return SimpleNamespace(weight=torch.zeros(1))
        def generate(self, **kwargs):
            calls.append(kwargs)
            return torch.tensor([[4, 5, 2]])
    probe = SimpleNamespace(torch=torch, tokenizer=Tokenizer(), model=Model(), conv_mode="test",
                            runtime=SimpleNamespace(conv_templates={"test": Conv()}))
    out = runner.TextOnlyLlava(probe, 512)("parse only")
    assert calls[0]["images"] is None and calls[0]["do_sample"] is False
    assert out["output_tokens"] == 3 and out["text"] == "{}"


def test_evaluator_rejects_missing_image_binding_before_loading_packets():
    evaluator = load_script("evaluate_request_parse_study")
    frozen = {"identity": "fixture", "questions": [ROW], "specs": {"xrv": spec()}}
    p = response([atom()])
    predictions = {"identity": "fixture", "records": []}
    gold = [{"id": ROW["id"], "question": Q, "reviewed": True, "reviewer": "fixture",
             "gold_parse": json.loads(p)}]
    with pytest.raises(ValueError, match="packet identity/image"):
        evaluator.evaluate(frozen, predictions, gold, [{"id": ROW["id"], "items": []}])
