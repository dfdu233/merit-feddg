"""CPU contracts and fake-backend integration; not medical efficacy tests."""
from __future__ import annotations

import base64
import json
import sys
import zlib
from argparse import Namespace
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from merit_feddg.agent_protocol import (
    atomic_json, load_incumbent, merge_shards, read_inputs, read_sources,
)
from merit_feddg.agent_regions import crop_box_pixels, decode_mask, extract_regions
from merit_feddg.agent_run import METHODS, json_value, live, load_options, prepare
from merit_feddg.agent_runtime import augment_case, image_digest
from merit_feddg.evidence_agent import (
    Action, EvidenceLedger, InvalidObservation, commit_or_keep, execute_plan, make_plan,
    validate_plan,
)
from merit_feddg.matched_evaluation import generation_prompt


def evidence():
    mask = np.zeros((8, 8), dtype='<f4')
    mask[1:4, 1:4] = 1
    raw = {'encoding': 'float32-zlib-base64', 'size': [8, 8],
           'data': base64.b64encode(zlib.compress(mask.tobytes())).decode()}
    return {'expert_id': 'segmenter', 'evidence_id': 'mask', 'capability': 'segmentation',
            'scope': 'anatomy', 'payload': {'structures': [
                {'label': 'organ', 'soft_mask': raw, 'mask_coordinate_system': 'original_image'}
            ]}, 'provenance': {'target_masks_used': False}}


def image_file(tmp_path, name='q.png', value=128):
    path = tmp_path / name
    Image.fromarray(np.full((8, 8, 3), value, dtype=np.uint8)).save(path)
    return str(path)


def fixture_run(tmp_path):
    image = image_file(tmp_path)
    row = {'id': 'q', 'image': image, 'question': 'What is visible?',
           'image_sha256': image_digest(image), 'answer_type': 'open'}
    manifest = tmp_path / 'manifest.jsonl'
    manifest.write_text(json.dumps(row) + '\n')
    cfg = {'evidence_style': 'semantic', 'vector_gate': 'off', 'semantic_spatial': False,
           'native_entry_transport': False, 'max_new_tokens': 64, 'block_tokens': 64,
           'compact_native': True, 'compact_columns': False}
    output = {'text': 'old', 'token_ids': [1], 'evidence': [evidence()],
              'generation_config': cfg, 'seconds': 1.0, 'input_modality': 'cxr', 'trace': []}
    base = tmp_path / 'base'
    config = {'generalist': {'id': 'fake'}, 'experts': {'segmenter': {'id': 'frozen-seg'}}}
    atomic_json(base / 'protocol.json', {'n': 1, 'identity': 'original',
                'shards_complete': True, 'config': config})
    atomic_json(base / 'compact_rows.json', {'q': output})
    options = tmp_path / 'agent.yaml'
    options.write_text('region_limit: 1\nmax_steps: 3\n')
    args = Namespace(config=str(options), base_run=str(base), incumbent='compact_rows',
                     manifest=str(manifest), source_manifest=None, artifacts='unused',
                     shard_index=0, shard_count=1, output=str(tmp_path / 'out'))
    return args, row, output


def test_agent_replays_unified_short_v1_base_prompt_without_answer_type_branching():
    config = {'prompt_contract': 'unified-short-v1'}
    closed = {'question': 'Is there edema?', 'answer_type': 'closed', 'task': 'open_vqa'}
    opened = {'question': 'Is there edema?', 'answer_type': 'open', 'task': 'open_vqa'}
    expected = ('Is there edema?\nAnswer the question with one concise answer based only on the '
                'image. Do not explain.')
    assert generation_prompt(closed, config) == expected
    assert generation_prompt(opened, config) == expected


def test_persisted_model_provenance_compares_after_json_container_normalization():
    memory = {'file_stats': [('config.json', 10, 20)]}
    persisted = json.loads(json.dumps(memory))
    assert memory != persisted
    assert json_value(memory) == persisted


def test_ledger_copies_lineage_and_local_invalidation():
    ledger = EvidenceLedger('case')
    payload = {'nested': [1]}
    ledger.add('a', 'native_observation', payload, source='same-checkpoint')
    ledger.add('b', 'native_observation', {}, source='other-checkpoint')
    ledger.add('r', 'region', {}, parents=('a',))
    ledger.add('c', 'crop', {}, parents=('r',))
    ledger.add('o', 'observation', {}, parents=('c',))
    payload['nested'].append(2)
    ledger.get('a')['payload']['nested'].append(3)
    assert ledger.get('a')['payload']['nested'] == [1]
    assert ledger.get('o')['roots'] == ['same-checkpoint']
    assert ledger.invalidate('r', 'bad coordinates') == ['c', 'o', 'r']
    assert ledger.active('a') and ledger.active('b') and not ledger.active('o')
    with pytest.raises(ValueError):
        ledger.add('new', 'observation', {}, parents=('o',))


@pytest.mark.parametrize('kwargs', [
    {'node_id': 'a', 'kind': 'x', 'payload': {'x': float('nan')}, 'source': 'model'},
    {'node_id': 'a', 'kind': 'x', 'payload': {}, 'parents': ('missing',)},
    {'node_id': 'a', 'kind': 'x', 'payload': {}},
])
def test_bad_ledger_inputs(kwargs):
    with pytest.raises(ValueError):
        EvidenceLedger('case').add(**kwargs)


def test_plan_types_cycles_unknown_actions_and_budget():
    ledger = EvidenceLedger('c')
    ledger.add('r', 'region', {}, source='s')
    actions = make_plan(['r'])
    validate_plan(actions, ledger)
    for invalid in [actions[::-1], (Action('x', 'exec', ('r',), 'crop', 'bad'),),
                    (Action('x', 'inspect', ('r',), 'observation', 'bad'),)]:
        with pytest.raises(ValueError):
            validate_plan(invalid, ledger)
    report = execute_plan(ledger, actions, {'crop': lambda *a: {}}, max_steps=1)
    assert report['attempted'] == ['crop:0'] and report['budget_exhausted']
    assert report['pending'] == ['inspect:0']


def test_planner_cannot_select_unready_or_arbitrary_action():
    ledger = EvidenceLedger('c')
    ledger.add('r', 'region', {}, source='s')
    with pytest.raises(ValueError, match='unavailable action'):
        execute_plan(ledger, make_plan(['r']), {}, max_steps=3,
                     choose=lambda *a: 'inspect:0')


def test_missing_failed_and_fatal_tools_not_negative_diagnoses():
    ledger = EvidenceLedger('c')
    ledger.add('r', 'region', {}, source='s')
    report = execute_plan(ledger, make_plan(['r']), {}, max_steps=4)
    assert report['failed'] == ['crop:0'] and report['pending'] == ['inspect:0']
    assert report['trace'][0]['event'] == 'unavailable'
    with pytest.raises(RuntimeError, match='CUDA'):
        execute_plan(ledger, make_plan(['r']),
                     {'crop': lambda *a: (_ for _ in ()).throw(RuntimeError('CUDA OOM'))},
                     max_steps=2)


def test_invalid_dependency_revokes_only_descendants():
    ledger = EvidenceLedger('c')
    ledger.add('r', 'region', {}, source='s')
    report = execute_plan(ledger, make_plan(['r']), {
        'crop': lambda *a: {},
        'inspect': lambda *a: (_ for _ in ()).throw(InvalidObservation('crop:0', 'changed')),
    }, max_steps=3)
    assert report['trace'][-1]['event'] == 'invalidated'
    assert ledger.active('r') and not ledger.active('crop:0')


@pytest.mark.parametrize('decision', ['unknown', 'reject'])
def test_keep_incumbent_exactly(decision):
    original = {'text': 'old useful answer', 'token_ids': [1, 2], 'evidence': [{'a': 1}]}
    result = commit_or_keep(original, {'text': 'new'}, decision=decision, reason='not established')
    assert all(result[k] == v for k, v in original.items())
    result['evidence'][0]['a'] = 2
    assert original['evidence'][0]['a'] == 1
    assert not result['agent_commit']['accepted']


def test_invalid_accept_and_nonfinite():
    old = {'text': 'old'}
    assert commit_or_keep(old, {'text': 'new'}, decision='accept', reason='invalid',
                          dependencies_valid=False)['text'] == 'old'
    with pytest.raises(ValueError):
        commit_or_keep(old, None, decision='maybe', reason='')


def test_soft_and_rle_masks_and_transform():
    item = evidence()
    mask = item['payload']['structures'][0]['soft_mask']
    assert decode_mask(mask).sum() == 9
    assert decode_mask({'encoding': 'rle-row-major-zero-first', 'size': [2, 2],
                        'counts': [1, 2, 1]}).tolist() == [[0, 1], [1, 0]]
    regions, _ = extract_regions([item], (8, 8))
    assert regions[0]['box'] == [.125, .125, .5, .5]
    item['payload']['structures'][0]['mask_coordinate_system'] = 'model_grid_of_center_crop'
    item['payload']['image_transform'] = {'original_size_hw': [16, 32], 'model_size_hw': [8, 8],
                                         'crop_box_xyxy_normalized': [.25, 0, .75, 1]}
    regions, _ = extract_regions([item], (32, 16))
    assert regions[0]['box'] == [.3125, .125, .5, .5]


@pytest.mark.parametrize('change', ['empty', 'nan', 'bytes', 'coordinates', 'omitted'])
def test_invalid_masks_never_invent_region(change):
    item = evidence()
    entry = item['payload']['structures'][0]
    raw = entry['soft_mask']
    if change in {'empty', 'nan'}:
        array = np.zeros((8, 8), dtype='<f4')
        if change == 'nan':
            array[0, 0] = np.nan
        raw['data'] = base64.b64encode(zlib.compress(array.tobytes())).decode()
    elif change == 'bytes':
        raw['data'] = base64.b64encode(zlib.compress(b'invalid')).decode()
    elif change == 'coordinates':
        entry['mask_coordinate_system'] = 'mystery'
    else:
        raw.pop('data')
    regions, audit = extract_regions([item], (8, 8))
    assert regions == [] and len(audit) == 1


def test_reference_mask_forbidden_cam_not_promoted():
    item = evidence()
    item['provenance']['target_masks_used'] = True
    with pytest.raises(ValueError, match='target'):
        extract_regions([item], (8, 8))
    item['capability'] = 'classification'
    assert extract_regions([item], (8, 8))[0] == []


def test_opposite_is_same_size_not_flip_and_center_logged():
    a = crop_box_pixels([.1, .2, .4, .7], (100, 80), padding=0)
    b = crop_box_pixels([.1, .2, .4, .7], (100, 80), padding=0, mode='opposite_control')
    assert (a[2]-a[0], a[3]-a[1]) == (b[2]-b[0], b[3]-b[1])
    assert a != b
    assert crop_box_pixels([0, 0, 1, 1], (8, 8), mode='opposite_control') == (0, 0, 8, 8)


@pytest.mark.parametrize('mode', ['static', 'agent', 'opposite_control', 'full_image_control'])
def test_real_crop_workflow_with_fake_frozen_model(tmp_path, mode):
    calls, syntheses = [], []
    path = image_file(tmp_path)
    def generate(image, prompt, max_tokens, allowed):
        calls.append((image, allowed))
        return {'text': allowed[1] if allowed else 'visible local feature', 'output_tokens': 4}
    def synthesize(extras):
        syntheses.append(extras)
        return {'text': 'candidate', 'token_ids': [3]}
    work = augment_case(case_id='c', image=path, question='What is visible?', evidence=[evidence()],
                        source_models={'segmenter': 'checkpoint'}, output_dir=str(tmp_path / mode),
                        generate=generate, synthesize=synthesize, mode=mode)
    assert work['candidate']['text'] == 'candidate'
    assert work['execution']['failed'] == []
    assert syntheses[0][0]['provenance']['source_roots'] == ['checkpoint']
    assert syntheses[0][0]['payload']['same_generalist_derived'] is True
    assert not work['medical_verification_implemented']


def test_planner_stop_and_no_masks_have_no_candidate(tmp_path):
    common = dict(case_id='c', image=image_file(tmp_path), question='Q',
                  source_models={'segmenter': 's'}, output_dir=str(tmp_path / 'w'),
                  generate=lambda *a: {'text': 'STOP'}, synthesize=lambda *a: pytest.fail('called'))
    assert augment_case(**common, evidence=[evidence()], mode='agent')['candidate'] is None
    assert augment_case(**common, evidence=[])['region_count'] == 0


def test_retrieval_is_analogy_not_observation(tmp_path):
    work = augment_case(case_id='c', image=image_file(tmp_path), question='Q', evidence=[evidence()],
                        source_models={'segmenter': 's'}, output_dir=str(tmp_path / 'w'),
                        generate=lambda *a: {'text': 'feature'}, synthesize=lambda *a: None,
                        retrieve=lambda path: {'source_only': True})
    extras = work['extra_evidence']
    assert len(extras) == 2
    assert extras[1]['scope'] == 'source_analogy'
    assert extras[1]['payload']['query_patient_fact'] is False


def test_manifest_labels_duplicate_and_hash_mismatch(tmp_path):
    args, row, _ = fixture_run(tmp_path)
    assert len(read_inputs(args.manifest)) == 1
    for changed in [{**row, 'answer': 'secret'}, {**row, 'image_sha256': 'bad'}]:
        Path(args.manifest).write_text(json.dumps(changed) + '\n')
        with pytest.raises(ValueError):
            read_inputs(args.manifest)
    Path(args.manifest).write_text((json.dumps(row) + '\n') * 2)
    with pytest.raises(ValueError):
        read_inputs(args.manifest)


def test_manifest_accepts_valid_official_provenance_only(tmp_path):
    args, row, _ = fixture_run(tmp_path)
    official = {**row, 'official_index': 17, 'official_split': 'train'}
    Path(args.manifest).write_text(json.dumps(official) + '\n')
    loaded = read_inputs(args.manifest)
    assert loaded[0]['official_index'] == 17
    assert loaded[0]['official_split'] == 'train'
    for changed in [
        {**official, 'official_index': -1},
        {**official, 'official_index': True},
        {**official, 'official_split': ''},
    ]:
        Path(args.manifest).write_text(json.dumps(changed) + '\n')
        with pytest.raises(ValueError):
            read_inputs(args.manifest)


def test_source_leakage_and_valid_corpus(tmp_path):
    args, row, _ = fixture_run(tmp_path)
    queries = read_inputs(args.manifest)
    source = {'id': 's', 'image': row['image'], 'image_sha256': row['image_sha256'],
              'domain': 'source-domain', 'group_id': 'source-group', 'question': 'Q',
              'modality': 'cxr', 'role': 'source', 'split': 'train', 'reference': 'source only'}
    path = tmp_path / 'sources.jsonl'
    path.write_text(json.dumps(source) + '\n')
    with pytest.raises(ValueError, match='overlap'):
        read_sources(path, queries)
    source['image'] = image_file(tmp_path, 's.png', 33)
    source['image_sha256'] = image_digest(source['image'])
    path.write_text(json.dumps(source) + '\n')
    records, refs, audit = read_sources(path, queries)
    assert len(records) == 1 and refs == {'s': 'source only'} and audit['enabled']
    source['split'] = 'test'
    path.write_text(json.dumps(source) + '\n')
    with pytest.raises(ValueError, match='train/external'):
        read_sources(path, queries)


def test_full_manifest_baseline_and_check_only_prepare(tmp_path):
    args, _, original = fixture_run(tmp_path)
    prepared = prepare(args)
    assert prepared[3]['q'] == original
    assert len(prepared[-1]) == 64 and prepared[-2]['enabled'] is False
    protocol = Path(args.base_run) / 'protocol.json'
    data = json.loads(protocol.read_text())
    data['shards_complete'] = False
    atomic_json(protocol, data)
    with pytest.raises(ValueError, match='finalized'):
        load_incumbent(args.base_run, args.incumbent, read_inputs(args.manifest))


def test_config_rejects_unknown_training_options(tmp_path):
    path = tmp_path / 'x.yaml'
    path.write_text('fit_gate: true\n')
    with pytest.raises(ValueError):
        load_options(path)


def test_atomic_merge_rejects_partial_wrong_assignment(tmp_path):
    methods, ids = ['a'], ['0', '1', '2']
    for index in range(2):
        atomic_json(tmp_path / 'shards' / str(index) / 'results.json', {
            'identity': 'i', 'shard_count': 2, 'shard_index': index, 'complete': True,
            'outputs': {'a': {key: {'text': key} for key in ids[index::2]}},
        })
    assert list(merge_shards(tmp_path, 'i', ids, methods, 2)['a']) == ['0', '2', '1']
    assert list(json.loads((tmp_path / 'a.json').read_text())) == ids
    with pytest.raises(ValueError):
        merge_shards(tmp_path, 'wrong', ids, methods, 2)
    (tmp_path / 'shards/1/results.json').unlink()
    with pytest.raises(ValueError, match='missing shard'):
        merge_shards(tmp_path, 'i', ids, methods, 2)


def test_live_adapter_with_fake_legacy_backend(tmp_path, monkeypatch):
    """Check live control flow and API binding without claiming a real GPU test."""
    args, _, _ = fixture_run(tmp_path)
    prepared = prepare(args)
    @dataclass
    class Item:
        expert_id: str
        evidence_id: str
        capability: str
        scope: str
        payload: dict
        provenance: dict = field(default_factory=dict)
        summary: str = ''
        confidence: float | None = None
    class Session:
        def __init__(self, *a):
            self.last_transport = {}
        def context(self, state):
            self.last_transport = {'presented': [
                {'expert_id': i.expert_id, 'evidence_id': i.evidence_id} for i in state.items]}
    class Runtime:
        def __init__(self, session, *a):
            self.session = session
        def complete(self, state):
            self.session.context(state)
            return {'text': 'old' if len(state.items) == 1 else 'new',
                    'token_ids': [1] if len(state.items) == 1 else [2],
                    'evidence_transport': self.session.last_transport, 'seconds': .1}
    probe = SimpleNamespace(context_token_budget=lambda *a: {'fits': True},
                            generate_with_usage=lambda image, text, max_new_tokens, allowed_texts:
                            {'text': allowed_texts[1] if allowed_texts else 'observed', 'output_tokens': 1})
    fixtures = {
        'capabilities': {'CapabilityRequest': SimpleNamespace, 'EvidenceItem': Item},
        'capability_experts': {'CapabilityPool': object},
        'capability_runtime': {'CapabilityRuntime': Runtime, 'NativeSession': Session,
                               'NativeState': SimpleNamespace, 'ValueGenerationConfig': SimpleNamespace},
        'generalist_factory': {'generalist_provenance': lambda *a: {'model': 'fake'},
                               'load_generalist': lambda *a: probe},
        'matched_evaluation': {'generation_prompt': lambda row, config: row['question']},
        'open_study': {'model_provenance': lambda *a: {'model': 'fake'}},
        'open_data': {'INFERENCE_FIELDS': {
            'id', 'image', 'question', 'modality', 'capability', 'task', 'domain',
            'domain_kind', 'role', 'group_id', 'image_sha256'}},
    }
    for name, values in fixtures.items():
        module = ModuleType('merit_feddg.' + name)
        module.__dict__.update(values)
        monkeypatch.setitem(sys.modules, module.__name__, module)
    import contextlib
    torch = ModuleType('torch')
    torch.inference_mode = contextlib.nullcontext
    monkeypatch.setitem(sys.modules, 'torch', torch)
    root = tmp_path / 'run'
    root.mkdir()
    live(args, prepared, root)
    result = json.loads((root / 'shards/0/results.json').read_text())['outputs']
    assert set(result) == set(METHODS)
    assert result['graph_noop']['q']['text'] == result['incumbent']['q']['text'] == 'old'
    assert result['agent_candidate']['q']['text'] == 'new'
    assert result['agent_committed']['q']['text'] == 'old'
    # Completed case resume never re-runs the fake model.
    probe.generate_with_usage = lambda *a, **k: pytest.fail('cache missed')
    live(args, prepared, root)


def test_diagnostic_scorer_does_not_convert_nonbinary_to_no():
    from merit_feddg.agent_evaluate import diagnostic_score, cluster_bootstrap
    assert diagnostic_score('left', 'left', 'closed') == 1
    assert diagnostic_score('No', 'left', 'closed') == 0
    assert diagnostic_score('No, none visible.', 'no', 'closed') == 1
    assert diagnostic_score('4th', '4th ventricle', 'open') == .5
    with pytest.raises(ValueError):
        diagnostic_score('text', 'text', 'report')
    ci = cluster_bootstrap([1, -1, 1], ['a', 'a', 'b'], repetitions=20)
    assert ci['clusters'] == 2 and ci['low'] <= ci['high']


def test_offline_evaluator_requires_complete_ids(tmp_path):
    from merit_feddg.agent_evaluate import evaluate
    args, _, old = fixture_run(tmp_path)
    root = tmp_path / 'evaluation'
    atomic_json(root / 'protocol.json', {'identity': 'i', 'shards_complete': True,
                                       'methods': ['incumbent', 'candidate']})
    atomic_json(root / 'incumbent.json', {'q': old})
    atomic_json(root / 'candidate.json', {'q': {**old, 'text': 'expected'}})
    refs = tmp_path / 'references.json'
    atomic_json(refs, {'q': ['expected']})
    report, _, _, _ = evaluate(root, args.manifest, refs, repetitions=10)
    assert report['methods']['candidate']['delta_to_incumbent'] == 1
    atomic_json(root / 'candidate.json', {})
    with pytest.raises(ValueError, match='IDs'):
        evaluate(root, args.manifest, refs)
