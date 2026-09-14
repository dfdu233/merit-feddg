import copy
from dataclasses import asdict, replace

import pytest

from merit_feddg.capabilities import CapabilityResult, EvidenceItem
from merit_feddg.intervention_risk import (
    InterventionAction,
    InterventionRiskConfig,
    InterventionRiskController,
    InterventionRiskScorer,
    InterventionSignals,
    fit_source_risk_thresholds,
)
from merit_feddg.med_defer import ClaimRequest, ExpertCard, LazyExpertPool, NativeEvidence
from merit_feddg.medcave import (
    CandidateSignalBuilder,
    FrozenPolicy,
    admissible_descriptors,
    binomial_upper,
    certificate_valid,
    certify,
    digest,
    run_case,
)
from merit_feddg.sequential_agent import SequentialAgentConfig, SequentialSpecialistAgent


def row():
    return {
        "id": "x",
        "image": "x.png",
        "question": "is the liver normal?",
        "modality": "ct",
        "task": "open_vqa",
        "domain": "source",
        "group_id": "p1",
        "image_sha256": "img1",
    }


def answer(text="baseline", items=()):
    return {
        "text": text,
        "token_ids": [1] if text == "baseline" else [2, len(items) + 3],
        "generation_config": {"max_new_tokens": 4},
        "evidence_transport": {
            "presented": [{"expert_id": i.expert_id, "evidence_id": i.evidence_id} for i in items]
        },
    }


class Pool:
    def __init__(self, cached=False, error=False, wrong=False):
        self.last_origin = "cached_native_output" if cached else "live_native_output"
        self.error, self.wrong = error, wrong

    def infer(self, expert, request):
        if self.error:
            raise RuntimeError("tool failed")
        return CapabilityResult(
            "wrong" if self.wrong else expert,
            request.capability,
            (
                EvidenceItem(
                    expert,
                    expert,
                    request.capability,
                    request.scope,
                    {"text": "liver observed"},
                    summary="liver observed",
                ),
            ),
        )


def descriptors():
    return [
        {"expert": e, "capability": "generation", "scope": "observation", "utility": 1.0}
        for e in ("A", "B")
    ]


def verification(row, candidate, items, state_hash):
    return {
        "fields": {},
        "unknown": [],
        "hard_failures": [],
        "score": 0.0,
        "candidate_hash": digest(candidate),
        "state_hash": state_hash,
    }


def certificate():
    policy = FrozenPolicy()
    ts = [
        {
            "id": str(i),
            "group_id": str(i),
            "image_sha256": str(i),
            "phase": "source-cal",
            "complete": True,
            "binding": "binding",
            "policy_action": "accept",
        }
        for i in range(100)
    ]
    scores = {t["id"]: {"candidate": 1, "baseline": 0} for t in ts}
    return certify(ts, scores, "binding", policy, {"groups": [], "images": []})


def run(**kwargs):
    defaults = {
        "binding": "binding",
        "policy": FrozenPolicy(),
        "descriptors": descriptors(),
        "fingerprints": {"A": "model1", "B": "model2"},
        "pool": Pool(),
        "generate": lambda items: answer("candidate" if items else "baseline", items),
    }
    return run_case(row(), **(defaults | kwargs))


def test_empty_calibration_and_zero_signals_cannot_accept():
    thresholds = fit_source_risk_thresholds([])
    s = InterventionSignals(
        applicability=1,
        source_reliability=1,
        pre_ood=0,
        post_ood=0,
        conflict=0,
        instability=0,
        coverage=1,
        visual_consistency=1,
        expert_confidence=1,
    )
    assert thresholds.acceptance_disabled
    assert InterventionRiskController(thresholds).decide(s).action != InterventionAction.ACCEPT
    assert not certificate_valid(
        certify([], {}, "binding", FrozenPolicy(), {"groups": [], "images": []}), "binding"
    )


@pytest.mark.parametrize("aggregation", ["max", "top2_mean"])
def test_structural_mismatch_cannot_be_accepted_at_threshold_one(aggregation):
    t = replace(
        fit_source_risk_thresholds([]),
        accept_max_risk=1,
        acquire_max_risk=1,
        acceptance_disabled=False,
    )
    c = InterventionRiskController(t, InterventionRiskScorer(InterventionRiskConfig(aggregation)))
    assert c.decide(InterventionSignals(applicability=0)).action == InterventionAction.FALLBACK


@pytest.mark.parametrize("v", [float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_signals_rejected(v):
    with pytest.raises(ValueError):
        InterventionSignals(coverage=v)


def test_unknown_signals_and_classifier_no_mask():
    i = EvidenceItem("e", "A", "classification", "anatomy", {"catalog": ["liver"]})
    a = CandidateSignalBuilder()(row(), answer(), (i,), digest(asdict(i)))
    assert "coverage" in a["unknown"]
    assert a["fields"]["visual_consistency"]["status"] == "not_applicable"
    assert not a["hard_failures"]


def test_fallback_exact_and_candidates_are_real():
    t = run()
    assert t["final"] == t["baseline"] == answer()
    assert t["candidate_calls"] == 2 and t["logical_calls"] == 2
    assert t["candidate"]["text"] == "candidate"
    assert t["acceptance_disabled"]


def test_accept_returns_verified_candidate_without_regeneration():
    generated = []

    def gen(items):
        generated.append(len(items))
        return answer("candidate" if items else "baseline", items)

    t = run(certificate=certificate(), verifier=verification, generate=gen)
    assert generated == [0, 1]
    assert t["action"] == "accept" and t["final"] == t["candidate"]
    assert t["final"]["token_ids"] == t["steps"][0]["candidate"]["token_ids"]


def test_second_expert_reassesses_entire_state_and_candidate():
    seen = []

    def verify(r, c, items, h):
        seen.append(([i.expert_id for i in items], copy.deepcopy(c)))
        return verification(r, c, items, h) | {"unknown": ["conflict"] if len(items) == 1 else []}

    t = run(certificate=certificate(), verifier=verify)
    assert [s[0] for s in seen] == [["A"], ["A", "B"]]
    assert seen[0][1] != seen[1][1]
    assert t["action"] == "accept" and t["final"] == seen[1][1]


def test_stale_candidate_verification_fails_closed():
    t = run(
        certificate=certificate(), verifier=lambda r, c, i, h: verification(r, answer("old"), i, h)
    )
    assert t["final"] == t["baseline"] and t["reason"] == "runtime_failure"


def test_aliases_not_counted_as_new_model_sources():
    t = run(fingerprints={"A": "same", "B": "same"})
    assert t["logical_calls"] == 1 and t["candidate_calls"] == 1


def test_cache_hits_still_consume_logical_budget():
    t = run(pool=Pool(cached=True), policy=FrozenPolicy(max_expert_calls=1))
    assert t["cache_hits"] == 1 and t["logical_calls"] == 1 and t["actual_tool_calls"] == 0
    assert t["reason"] == "acquisition_budget_exhausted"


@pytest.mark.parametrize("pool", [Pool(error=True), Pool(wrong=True)])
def test_tool_failure_or_wrong_identity_audited(pool):
    t = run(pool=pool)
    assert t["final"] == t["baseline"] and t["actual_tool_calls"] == 1
    assert t["steps"][0]["status"] == "failed" and not t["complete"]


def test_nonpositive_utility_skips_calls():
    t = run(descriptors=[dict(d, utility=0) for d in descriptors()])
    assert t["logical_calls"] == 0 and t["final"] == t["baseline"]


def test_semantic_delivery_failure_cannot_be_accepted():
    t = run(
        certificate=certificate(),
        verifier=verification,
        generate=lambda items: answer("candidate" if items else "baseline"),
    )
    assert t["final"] == t["baseline"]
    assert t["reason"] == "structural_or_delivery_failure"


def test_hard_modality_and_question_attribute_filter():
    spec = {
        "A": {
            "modalities": ["ct"],
            "tasks": ["open_vqa"],
            "capabilities": ["classification"],
            "scope": "anatomy",
            "question_types": ["anatomy"],
            "request_contract": {"mode": "named_concepts", "concept_aliases": {"liver": []}},
        }
    }
    assert admissible_descriptors(spec, row(), {"A": 1}) == []


def test_exact_binomial_zero_harms():
    assert binomial_upper(0, 100, 0.05) == pytest.approx(1 - 0.05**0.01)
    assert binomial_upper(0, 0, 0.05) == 1


def test_certificate_identity_and_tampering():
    cert = certificate()
    assert certificate_valid(cert, "binding")
    assert not certificate_valid(cert, "changed-model-or-policy")
    cert["harmful_groups"] = 50
    assert not certificate_valid(cert, "binding")


def test_grouped_calibration_never_counts_questions_as_independent():
    ts = [
        {
            "id": str(i),
            "group_id": "one-patient",
            "image_sha256": str(i),
            "phase": "source-cal",
            "complete": True,
            "binding": "binding",
            "policy_action": "accept",
        }
        for i in range(100)
    ]
    s = {t["id"]: {"candidate": 1, "baseline": 0} for t in ts}
    cert = certify(ts, s, "binding", FrozenPolicy(), {"groups": [], "images": []})
    assert cert["accepted_groups"] == 1 and cert["acceptance_disabled"]
    with pytest.raises(ValueError, match="leakage"):
        certify(ts, s, "binding", FrozenPolicy(), {"groups": ["one-patient"], "images": []})


def test_zero_bias_and_cross_request_cache():
    request = ClaimRequest("s", "c", "ct", ("classification",), ("no", "yes"), (1, 0), 0.5, {})
    evidence = NativeEvidence("A", "classification", {"no": 0, "yes": 1}, 0.5)
    agent = SequentialSpecialistAgent(
        InterventionRiskController(fit_source_risk_thresholds([])),
        config=SequentialAgentConfig(max_bias_norm=0),
    )
    assert agent._guided_logits(request, evidence) == request.base_logits
    pool = LazyExpertPool()
    pool.register(ExpertCard("A", ("ct",), ("classification",), 0.9), lambda r: evidence)
    assert pool.get("A", request)[1] is False
    assert pool.get("A", request)[1] is True
    assert pool.get("A", replace(request, question="different"))[1] is False
    assert pool.get("A", replace(request, image_sha256="different"))[1] is False


def test_replay_never_invents_joint_candidate():
    from merit_feddg.medcave import replay_trajectory
    t = run()
    assert replay_trajectory(t, row(), 'binding', FrozenPolicy())['status'] == 'exact_trajectory_available'
    t['steps'][1]['candidate'] = answer('unverified answer')
    assert replay_trajectory(t, row(), 'binding', FrozenPolicy())['status'] == 'replay_unavailable'
    assert replay_trajectory({'outputs': {'old_arm': answer()}}, row(), 'binding',
                             FrozenPolicy())['status'] == 'replay_unavailable'


def test_explicit_polarity_conflict_and_unknown_qualification():
    from merit_feddg.medcave import SourceArtifactVerifier
    items = tuple(EvidenceItem(e,e,'generation','observation',{'assertions':[
        {'image_sha256':'img1','scope':'observation','proposition':'liver normal',
         'location':'liver','polarity':p}]}) for e,p in [('A','positive'),('B','negative')])
    verifier = SourceArtifactVerifier({'A':{},'B':{}}, {'A':'modelA','B':'modelB'})
    audit = verifier(row(),answer(),items,digest([asdict(i) for i in items]))
    assert 'explicit_same_proposition_conflict' in audit['hard_failures']
    assert 'source_reliability' in audit['unknown']
    assert audit['fields']['conflict']['value'] == 1


def test_all_cli_stages_use_frozen_policy_and_fail_closed(tmp_path, monkeypatch):
    """CPU integration of executable stages; fake backend is not a medical result."""
    import json
    import sys
    from types import SimpleNamespace

    from merit_feddg import capability_experts, capability_runtime, generalist_factory, medcave_run

    config = {'experts':{}, 'generalist':{}, 'utilities':{}, 'prompt_suffix':'Answer briefly',
              'generation':{'visual_views':0,'token_budgeted_evidence':True}}
    current = [row()]
    def prepare(*args):
        return config,FrozenPolicy(),copy.deepcopy(current),[
            r|{'answer_type':'open'} for r in current], 'binding', {}, {}
    monkeypatch.setattr(medcave_run,'prepare',prepare)
    monkeypatch.setattr(generalist_factory,'load_generalist',lambda *a:object())
    monkeypatch.setattr(capability_experts,'CapabilityPool',lambda *a,**kw:Pool())
    class Session:
        def __init__(self,*a):
            self.last_transport={'presented':[]}
        def propose(self,*a):
            return SimpleNamespace(tokens=[1])
        def decode(self,*a):
            return 'baseline'
    monkeypatch.setattr(capability_runtime,'NativeSession',Session)
    refs=tmp_path/'refs.json'
    refs.write_text(json.dumps({'x':['baseline'],'cal':['baseline'],'test':['baseline']}))
    def invoke(stage,*args):
        output=tmp_path/stage
        monkeypatch.setattr(sys,'argv',['medcave_run','--stage',stage,'--config','unused',
            '--manifest','unused','--output',str(output),*map(str,args)])
        medcave_run.main()
        return next(output.iterdir())
    invoke('dry-run')
    dev=invoke('source-dev','--references',refs)
    frozen=dev/'policy.json'
    assert json.loads(frozen.read_text())['binding']=='binding'
    current[:]=[row()|{'id':'cal','group_id':'patient-cal','image_sha256':'image-cal'}]
    cal=invoke('source-cal','--policy',frozen,'--references',refs)
    cert=cal/'certificate.json'
    assert json.loads(cert.read_text())['acceptance_disabled']
    assert json.loads(cert.read_text())['conditional_harm'] is None
    replay=invoke('replay','--replay-run',cal)
    assert json.loads((replay/'replay.json').read_text())['rows'][0]['status']=='exact_trajectory_available'
    from merit_feddg import capability_routing
    route_calls = []
    def route_once(probe, sample):
        route_calls.append(sample['id'])
        return {'modality':'cxr', 'seconds':1.0}
    monkeypatch.setattr(capability_routing, 'infer_image_type', route_once)
    current[:] = [row() | {'id': str(i), 'modality':'mixed'} for i in range(2)]
    smoke = invoke('source-smoke')
    smoke_diag = json.loads((smoke/'diagnostics.json').read_text())
    assert len(route_calls) == 1
    assert smoke_diag['routing_calls'] == 1 and smoke_diag['routing_cache_hits'] == 1
    assert smoke_diag['candidate_coverage'] == 0
    assert smoke_diag['candidate_metrics'] == {}
    assert smoke_diag['timing_totals_seconds']['case_wall_seconds'] >= 0
    current[:]=[row()|{'id':'test','group_id':'patient-test','image_sha256':'image-test'}]
    evaluation=invoke('evaluate','--policy',frozen,'--certificate',cert,'--references',refs)
    diag=json.loads((evaluation/'diagnostics.json').read_text())
    assert diag['accepted']==0 and diag['harmful_accept_rate'] is None
    current[:]=[row()]
    with pytest.raises(ValueError,match='overlap'):
        invoke('source-cal','--policy',frozen,'--references',refs)
