import copy
from dataclasses import replace

import numpy as np
import pytest

from merit_feddg.answer_arbitration import arbitrate_output
from merit_feddg.capabilities import EvidenceItem
from merit_feddg.capability_runtime import ValueGenerationConfig
from merit_feddg.compact_evidence import compact_records
from merit_feddg.io import load_experiment_yaml
from merit_feddg.matched_evaluation import experiment_arms
from merit_feddg.revision_audit import audit_revisions
from merit_feddg.revision_policy import preserve_unverified_candidate
from merit_feddg.spatial_evidence import encode_soft_mask


def output(text, token):
    return {'text':text, 'token_ids':[token], 'seconds':.1, 'evidence':[], 'trace':[], 'finished':True}


def test_omitted_expert_is_not_used_to_disqualify_verifier():
    from merit_feddg.revision_policy import presented_expert_ids
    result = {'evidence':[{'expert_id':'omitted'}, {'expert_id':'shown'}],
        'trace':[{'event':'decode', 'evidence_transport':{'presented':[{'expert_id':'shown'}]}}]}
    assert presented_expert_ids(result) == {'shown'}


@pytest.mark.parametrize('decision', ['accept', 'reject', 'abstain', 'unchanged'])
def test_policy_preserves_exact_tokens_and_never_calls_abstention_verified(decision):
    base, cand = output('spleen', 1), output('liver', 2)
    if decision == 'unchanged':
        cand = copy.deepcopy(base)
    old = copy.deepcopy(base)
    old.update(candidate_answer={'text':cand['text'], 'token_ids':cand['token_ids']},
               answer_arbitration={'decision':decision, 'reason':'test'})
    saved = copy.deepcopy((base, cand, old))
    result = preserve_unverified_candidate(base, cand, old)
    assert result['token_ids'] == (base if decision == 'reject' else cand)['token_ids']
    assert result['answer_arbitration']['candidate_verified'] == (decision == 'accept')
    assert (base, cand, old) == saved


def test_same_model_correction_is_unverified_not_vetoed():
    from types import SimpleNamespace
    base, cand = output('spleen', 1), output('liver', 2)
    verifier = SimpleNamespace(model_id='biomedclip', modalities=['ct'])
    old = arbitrate_output(base, cand, image='unused', question='organ?', modality='ct',
                          verifier=verifier, generalist_id='g', source_model_ids=['biomedclip'])
    assert old['text'] == 'spleen'  # legacy path is unchanged
    result = preserve_unverified_candidate(base, cand, old)
    assert result['text'] == 'liver'
    assert result['answer_arbitration']['decision'] == 'abstain'
    assert result['answer_arbitration']['score_calls'] == 0
    audit = audit_revisions({'x':base}, {'x':cand}, {'x':result},
        scores={'x':{'baseline':0., 'candidate':1.}}, groups={'x':'hospital_A'})
    assert audit['events']['beneficial:abstain'] == 1
    assert audit['domains']['hospital_A']['gate_gain'] == 0
    with pytest.raises(ValueError, match='identical case'):
        audit_revisions({'x':base}, {}, {'x':result})


def test_geometry_is_measured_without_modifying_masks_or_asserting_presence():
    mask = np.zeros((4, 4), dtype=np.float32)
    mask[0, 3] = 1
    item = EvidenceItem('e', 'seg', 'segmentation', 'anatomy',
        {'soft_mask':encode_soft_mask(mask), 'mask_coordinate_system':'original_image'})
    saved = copy.deepcopy(item)
    old = compact_records([item])[0]['payload']['soft_mask']
    new = compact_records([item], geometry=True)[0]['payload']['soft_mask']
    assert 'geometry_summary' not in old and 'data' not in new
    assert new['geometry_summary']['weighted_centroid_xy'] == [.875, .125]
    assert not new['geometry_summary']['object_presence_established']
    assert item == saved


def test_new_arms_keep_old_arms_and_isolate_geometry_from_spatial_operator():
    config = load_experiment_yaml('configs/matched_evidence_revision.yaml')
    assert config['prompt_contract'] == 'anchor-ce-v1'
    decoder = ValueGenerationConfig(**config['capability_value']['generation'])
    old = experiment_arms(decoder, 'verified_packets')
    new = experiment_arms(decoder, 'evidence_revision')
    assert all(new[k] == v for k, v in old.items()) and len(new) == 8
    assert replace(new['compact_geometry'], compact_geometry=False) == new['compact_all']
    assert replace(new['compact_spatial'], semantic_spatial=False) == new['compact_geometry']
    assert new['compact_guarded'] == new['compact_all']


@pytest.mark.parametrize('enable_uncertainty', [False, True])
def test_complete_runner_reuses_candidate_and_skips_same_source_model_load(tmp_path, monkeypatch, enable_uncertainty):
    import json
    from dataclasses import asdict
    from types import SimpleNamespace

    from PIL import Image

    from merit_feddg import (
        capability_experts,
        capability_runtime,
        capability_study,
        generalist_factory,
    )
    from merit_feddg import matched_evaluation as runner
    from merit_feddg.spatial_evidence import SpatialEvidenceBridge

    image = tmp_path/'image.png'
    Image.new('RGB', (4, 4)).save(image)
    manifest = tmp_path/'manifest.jsonl'
    manifest.write_text(json.dumps({'id':'case', 'image':str(image), 'image_sha256':'h',
                                     'question':'Which organ?', 'answer_type':'open',
                                     'task':'open_vqa'}))
    config = load_experiment_yaml('configs/matched_evidence_revision.yaml')
    if enable_uncertainty:
        config['uncertainty_comparison'] = {'enabled':True, 'samples':5,
            'nli_checkpoint':'unused', 'candidates':['compact_all']}
        monkeypatch.setattr('merit_feddg.semantic_uncertainty.FrozenDebertaEntailment', lambda *a: object())
        monkeypatch.setattr('merit_feddg.semantic_uncertainty.estimate_answer_uncertainty', lambda *a, **kw:
            {'available':True, 'scores':{'predictive_entropy':1., 'semantic_entropy':0.,
                                        'discrete_semantic_entropy':0.}})
    config['experts'] = {'biomed_anatomy':config['experts']['biomed_anatomy']}
    monkeypatch.setattr(runner, 'load_experiment_yaml', lambda p: config)
    monkeypatch.setattr(runner, 'model_provenance', lambda spec, artifacts: {'id':spec['id']})
    monkeypatch.setattr(generalist_factory, 'generalist_provenance', lambda *a: {'id':'g'})
    monkeypatch.setattr(generalist_factory, 'load_generalist', lambda *a:
        SimpleNamespace(tensor_bridge=SpatialEvidenceBridge(1)))
    monkeypatch.setattr(capability_study, '_filter_optional_experts', lambda specs, artifacts: (specs, {}))
    monkeypatch.setattr(capability_study, '_route_records', lambda rows, *a:
        ([{**r, 'modality':'ct'} for r in rows], {}))
    monkeypatch.setattr(runner, 'NativeSession', lambda *args: None)
    monkeypatch.setattr(capability_experts, 'CapabilityPool', lambda *a, **k:
        SimpleNamespace(reset_case=lambda: None, clear=lambda: None))
    calls = []
    class Engine:
        def __init__(self, session, pool, row, specs, arm, encoder):
            assert 'answer_type' not in row
        def run(self, mode):
            calls.append(mode)
            result = output('spleen' if mode == 'generalist' else 'liver', 1 if mode == 'generalist' else 2)
            if mode != 'generalist':
                result['evidence'] = [asdict(EvidenceItem('e', 'biomed_anatomy', 'classification', 'anatomy', {}))]
            return result
    monkeypatch.setattr(capability_runtime, 'CapabilityRuntime', Engine)
    monkeypatch.setattr('merit_feddg.answer_arbitration.load_verifier',
                        lambda *a: pytest.fail('same-source verifier must not load'))
    root = runner.run(manifest, 'unused', tmp_path/'run', protocol='evidence_revision')
    guarded = json.loads((root/'compact_guarded.json').read_text())['case']
    old = json.loads((root/'compact_verified.json').read_text())['case']
    assert len(calls) == 6 and guarded['text'] == 'liver' and old['text'] == 'spleen'
    assert guarded['answer_arbitration']['decision'] == 'abstain'
    assert guarded['answer_arbitration']['score_calls'] == 0
    if enable_uncertainty:
        result = json.loads((root/'uncertainty__compact_all__semantic_entropy.json').read_text())['case']
        assert result['text'] == 'liver' and result['answer_arbitration']['decision'] == 'abstain'
