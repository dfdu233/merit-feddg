import copy
import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from merit_feddg.answer_arbitration import arbitrate_output, assess_revision, render_pairs
from merit_feddg.capabilities import EvidenceItem
from merit_feddg.capability_runtime import ValueGenerationConfig
from merit_feddg.compact_evidence import columnar, compact_prompt, compact_records, factor_shared
from merit_feddg.evidence_transport import pack_records
from merit_feddg.io import load_experiment_yaml
from merit_feddg.matched_evaluation import SharedExpertPool, experiment_arms
from merit_feddg.open_study import atomic_json, fingerprint
from merit_feddg.spatial_evidence import encode_soft_mask


def expand(value):
    if isinstance(value, list):
        return [expand(v) for v in value]
    if not isinstance(value, dict):
        return value
    if set(value) == {'columns', 'rows'}:
        return [dict(zip(value['columns'], [expand(v) for v in row], strict=True)) for row in value['rows']]
    if set(value) == {'shared_fields', 'entries'}:
        def merge(a, b):
            result = copy.deepcopy(a)
            for k, v in b.items():
                result[k] = merge(result[k], v) if k in result and isinstance(v, dict) else v
            return result
        return [merge(value['shared_fields'], expand(v)) for v in expand(value['entries'])]
    return {k: expand(v) for k, v in value.items()}


def test_nested_compaction_is_reversible_and_does_not_equate_missing_with_null():
    value = [{'name': 'a', 'score': .9, 'support': {'type': 'CAM', 'array': 0}},
             {'name': 'b', 'score': .001, 'support': {'type': 'CAM', 'array': 1}}]
    original = copy.deepcopy(value)
    assert expand(columnar(factor_shared(value))) == original
    assert value == original
    ragged = [{'name': 'a'}, {'name': 'b', 'score': None}]
    assert expand(columnar(factor_shared(ragged))) == ragged


@pytest.mark.parametrize('capability,key', [('classification', 'findings'), ('segmentation', 'structures'),
    ('detection', 'objects'), ('retrieval', 'references'), ('generation', 'observations')])
def test_all_native_scalar_fields_survive_across_capabilities(capability, key):
    payload = {key: [{'label': str(i), 'score': i/100, 'uncertainty': 'native',
                      'custom_attribute': {'units': 'mm', 'value': i}} for i in range(18)]}
    item = EvidenceItem('packet', 'expert', capability, 'scope', payload, confidence=.4)
    record = compact_records((item,))[0]
    assert expand(columnar(record))['payload'] == payload
    assert expand(columnar(record))['confidence'] == .4
    assert record['evidence_id'] == 'packet'


def test_all_classification_labels_survive_spatial_array_removal():
    payload = {'findings': [{'finding': f'finding-{i}', 'score': i/100, 'spatial_support': {
        'soft_mask': encode_soft_mask(np.full((24,24), i/100)), 'mask_coordinate_system': 'original_image',
        'representation': 'activation_not_lesion'}} for i in range(18)],
        'score_semantics': 'uncalibrated_independent_sigmoid'}
    item = EvidenceItem('packet', 'expert', 'classification', 'scope', payload)
    record = compact_records((item,))[0]
    restored = expand(columnar(record))
    assert [v['score'] for v in restored['payload']['findings']] == [i/100 for i in range(18)]
    assert all('data' not in v['spatial_support']['soft_mask'] for v in restored['payload']['findings'])
    assert record['interface']['array_count'] == 18
    assert 'data' in payload['findings'][0]['spatial_support']['soft_mask']


def test_layout_budget_intersection_keeps_exact_same_evidence():
    records = compact_records([EvidenceItem(str(i), 'expert', 'classification', 's',
        {'findings': [{'label': str(j), 'score': j/100} for j in range(18)]}) for i in range(3)])
    render = [lambda x: compact_prompt('Q', x, columns=False), lambda x: compact_prompt('Q', x, columns=True)]
    limit = max(len(f(records[:1])) for f in render)+1
    measure = lambda prompt, reserve: {'fits': len(prompt) <= limit, 'input_tokens': len(prompt)}
    selected = [pack_records(records, render[i], measure, max_chars=100000, reserve_tokens=64,
                             companion_render=render[1-i])[0] for i in range(2)]
    assert selected[0] == selected[1] == records[:1]


class Verifier:
    model_id = 'external'
    modalities = frozenset({'cxr', 'mri', 'pathology'})
    def __init__(self, scores):
        self.scores, self.calls = scores, []
    def score(self, image, texts):
        self.calls.append((image, texts))
        return self.scores


@pytest.mark.parametrize('scores,decision', [([.1,.2,.3,.4], 'accept'), ([.2,.1,.4,.3], 'reject'),
    ([.1,.2,.4,.3], 'abstain'), ([.1,.1,.2,.2], 'abstain'), ([0,np.nan,0,1], 'abstain')])
def test_external_preference_has_no_calibrated_probability_claim(scores, decision):
    verifier = Verifier(scores)
    result = assess_revision('baseline', 'candidate', image='image', question='Question?', modality='cxr',
                             verifier=verifier, generalist_id='generalist')
    assert result['decision'] == decision and result['score_calls'] == 1
    assert not result['calibrated_probability'] and not result['correctness_guaranteed']
    assert len(verifier.calls[0][1]) == 4


def test_unknown_scope_and_same_checkpoint_never_auto_accept():
    verifier = Verifier([0,1,0,1])
    for modality, sources, expected in [('unregistered', (), 'no_applicable_external_verifier'),
        ('mri', ('external',), 'verifier_reuses_generator_or_evidence_checkpoint')]:
        result = assess_revision('a', 'b', image='i', question='q', modality=modality, verifier=verifier,
                                  generalist_id='generalist', source_model_ids=sources)
        assert result['decision'] == 'abstain' and result['reason'] == expected
    assert verifier.calls == []


def test_bare_answer_preserves_question_binding_without_disease_dictionary():
    for q in ['Which side?', '是否异常？', 'What is the measurement?']:
        pairs = render_pairs(q, 'No', 'Left')
        assert all(q in text for pair in pairs for text in pair)
    v = Verifier([0,1,0,1])
    result = assess_revision('same', 'same', image='i', question='q', modality='cxr', verifier=v, generalist_id='g')
    assert result['decision'] == 'unchanged' and not v.calls


def test_abstention_returns_exact_baseline_tokens_without_erasing_attempt_audit():
    base = {'text':'original', 'token_ids':[7,8], 'seconds':.5, 'finished':True, 'evidence_transport':{'presented':[]}}
    candidate = {'text':'changed', 'token_ids':[9], 'seconds':1., 'trace':[], 'evidence':[{'expert_id':'x'}]}
    saved = copy.deepcopy(candidate)
    result = arbitrate_output(base, candidate, image='i', question='q', modality='unknown', verifier=None,
                              generalist_id='g')
    assert result['text'] == 'original' and result['token_ids'] == [7,8]
    assert result['evidence'] == [] and result['candidate_evidence'] == saved['evidence']
    assert result['candidate_answer']['token_ids'] == [9] and result['seconds'] >= 1.5
    assert candidate == saved


def test_five_arms_preserve_uniform_protocol_and_isolate_layout_from_arbitration():
    config = load_experiment_yaml('configs/matched_verified_packets.yaml')
    decoder = ValueGenerationConfig(**config['capability_value']['generation'])
    arms = experiment_arms(decoder, 'verified_packets')
    assert config['prompt_contract'] == 'anchor-ce-v1' and len(arms) == 5
    assert replace(arms['compact_rows'], compact_columns=True) == arms['compact_all'] == arms['compact_verified']
    assert all(v.vector_gate == 'off' and not v.native_entry_transport and not v.semantic_spatial for v in arms.values())
    assert all(v.max_new_tokens == v.block_tokens == 64 for v in arms.values())


def test_actual_verifier_freezes_parameters_and_rejects_text_truncation():
    import torch

    from merit_feddg.answer_arbitration import FrozenImageTextVerifier
    model = torch.nn.Linear(1, 1)
    adapter = SimpleNamespace(model=model, torch=torch,
        tokenizer=SimpleNamespace(tokenizer=SimpleNamespace(encode=lambda *a, **kw: list(range(257)))),
        _text_embeddings=lambda texts: pytest.fail('must not embed truncated text'))
    verifier = FrozenImageTextVerifier(adapter, model_id='external', modalities=['cxr'])
    assert not model.training and not any(p.requires_grad for p in model.parameters())
    result = assess_revision('old', 'new', image='not_opened', question='q', modality='cxr',
                             verifier=verifier, generalist_id='g')
    assert result['decision'] == 'abstain' and result['reason'] == 'verification_unavailable'


def test_runner_reuses_candidate_and_keeps_answer_type_out_of_runtime(tmp_path, monkeypatch):
    from PIL import Image

    from merit_feddg import (
        capability_experts,
        capability_runtime,
        capability_study,
        generalist_factory,
    )
    from merit_feddg import matched_evaluation as runner
    image = tmp_path / 'i.png'
    Image.new('RGB', (4, 4)).save(image)
    manifest = tmp_path / 'manifest.jsonl'
    manifest.write_text(json.dumps({'id':'case', 'image':str(image), 'image_sha256':'h',
                                    'question':'Q', 'answer_type':'open'}))
    config = load_experiment_yaml('configs/matched_verified_packets.yaml')
    config['experts'] = {}
    config['answer_verifiers']['biomedclip']['modalities'] = ['mixed']
    monkeypatch.setattr(runner, 'load_experiment_yaml', lambda path: config)
    monkeypatch.setattr(runner, 'model_provenance', lambda spec, artifacts: {'id':spec['id']})
    monkeypatch.setattr(generalist_factory, 'generalist_provenance', lambda *a: {'id':'g'})
    monkeypatch.setattr(generalist_factory, 'load_generalist', lambda *a: object())
    monkeypatch.setattr(capability_study, '_filter_optional_experts', lambda specs, artifacts: ({}, {}))
    monkeypatch.setattr(capability_study, '_route_records', lambda rows, *a: (rows, {}))
    monkeypatch.setattr(runner, 'NativeSession', lambda probe, image, prompt, question, arm: arm)
    monkeypatch.setattr(capability_experts, 'CapabilityPool', lambda *a, **kw:
        SimpleNamespace(reset_case=lambda: None, clear=lambda: None))
    calls = []
    class Engine:
        def __init__(self, session, pool, row, specs, arm, encoder):
            assert 'answer_type' not in row
            self.arm = arm
        def run(self, mode):
            calls.append((mode, self.arm.compact_native))
            return {'text':'candidate' if self.arm.compact_native else 'base',
                    'token_ids':[2] if self.arm.compact_native else [1], 'finished':True,
                    'seconds':.1, 'trace':[], 'evidence':[]}
    monkeypatch.setattr(capability_runtime, 'CapabilityRuntime', Engine)
    verifier = Verifier([0,1,0,1])
    verifier.modalities = frozenset({'mixed'})
    monkeypatch.setattr('merit_feddg.answer_arbitration.load_verifier', lambda *a: verifier)
    reuse = tmp_path/'generalist.json'
    reuse.write_text(json.dumps({'schema':'matched-generalist-reuse-v1',
        'prompt_contract':'anchor-ce-v1', 'generalist_id':config['generalist']['id'],
        'outputs':{'case':{'text':'base', 'token_ids':[1], 'finished':True, 'seconds':0.,
                           'trace':[], 'evidence':[]}}}))
    root = runner.run(manifest, 'config', tmp_path/'out', protocol='verified_packets',
                      reuse_generalist=reuse)
    result = json.loads((root/'compact_verified.json').read_text())['case']
    assert len(calls) == 3 and len(verifier.calls) == 1
    assert result['text'] == 'candidate' and result['answer_arbitration']['decision'] == 'accept'
    protocol = json.loads((root/'protocol.json').read_text())
    assert protocol['answer_type_used_for_generation'] and not protocol['dataset_partitioned']
    assert not protocol['baseline_regenerated']
    assert protocol['reuse_generalist']['n'] == 1


def test_shared_expert_pool_reuses_request_keyed_compatible_cache(tmp_path):
    from merit_feddg.capabilities import CapabilityRequest

    request = CapabilityRequest('case', 'image', 'question', 'cxr', 'open_vqa', 'target',
                                'group', 'generation', query='q', scope='scope')
    donor = tmp_path/'donor'
    key = fingerprint(['infer', 'expert', request.__dict__])
    value = {'expert_id':'expert', 'capability':'generation', 'reason':'ok', 'items':[{
        'evidence_id':'e', 'expert_id':'expert', 'capability':'generation', 'scope':'scope',
        'payload':{'generated_text':'observation'}, 'summary':'', 'confidence':None,
        'provenance':{}}]}
    atomic_json(donor/f'{key}.json', {'identity':'donor-id', 'output':value})
    live = SimpleNamespace(infer=lambda *a: pytest.fail('compatible cache must avoid live inference'))
    pool = SharedExpertPool(live, tmp_path/'current', 'current-id', ((donor, 'donor-id'),))
    result = pool.infer('expert', request)
    assert result.items[0].payload['generated_text'] == 'observation'
    assert pool.last_origin == 'reused_compatible_native_output'


def test_shared_fields_preserve_boolean_integer_and_float_types():
    value = [{'value': {'score': True}}, {'value': {'score': 1}}, {'value': {'score': 1.0}}]
    restored = expand(columnar(factor_shared(value)))
    assert [type(v['value']['score']) for v in restored] == [bool, int, float]


def test_native_keys_that_look_like_layout_syntax_use_untransformed_layout():
    payload = {'rows': [1, 2], 'columns': ['a'], 'custom': 'not a generated table'}
    item = EvidenceItem('id', 'expert', 'generation', 'scope', payload)
    record = compact_records((item,))[0]
    assert record['interface']['native_layout'] and record['payload'] == payload
    assert compact_prompt('Q', [record], columns=True) == compact_prompt('Q', [record], columns=False)
