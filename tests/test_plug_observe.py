"""CPU contract and integration tests; no model weights, targets or network."""
import copy
import json

import numpy as np
import pytest
from PIL import Image

from merit_feddg.plug_observe import (
    ANSWER_SUFFIX, available_actions, native_dict, observation, pack_observations,
    payload_view, run_case, validate_specs,
)
from merit_feddg.evidence_agent import ToolUnavailable


def item(expert='specialist', capability='classification', payload=None):
    return {'evidence_id': expert + ':case', 'expert_id': expert, 'capability': capability,
            'scope': capability + '_scope', 'payload': payload or {'finding': 'not present', 'score': .2},
            'summary': 'Native observation; not a calibrated diagnosis', 'confidence': None,
            'provenance': {'revision': 'frozen', 'log_path': '/private/log', 'sha256': 'x' * 64}}


def segment():
    # Rows: 0000 / 0110 / 0110 / 0000, encoded row-major.
    return item('segmenter', 'segmentation', {'structures': [{
        'label': 'test structure', 'mask_coordinate_system': 'original_image',
        'mask': {'encoding': 'rle-row-major-zero-first', 'size': [4, 4], 'counts': [5, 2, 2, 2, 5]},
    }]})


def spec(cap='classification', name='specialist', **extras):
    return {name: {'id': 'released/' + name, 'modalities': ['ct'], 'tasks': ['open_vqa'],
                   'capabilities': [cap], 'scope': cap + '_scope',
                   'requires_region': cap == 'segmentation',
                   'description': 'A frozen expert providing native observations.', **extras}}


@pytest.fixture
def case(tmp_path):
    path = tmp_path / 'image.png'
    Image.fromarray(np.arange(48, dtype=np.uint8).reshape(4, 4, 3)).save(path)
    return {'id': 'test', 'image': str(path), 'question': 'What is visible?',
            'modality': 'ct', 'task': 'open_vqa', 'domain': 'query', 'group_id': 'patient1',
            'image_sha256': 'unused_by_core'}


class Backend:
    def __init__(self, choices=None):
        self.calls = []
        self.choices = iter(choices or ['STOP'])
        self.limit = 1_000_000
        self.items = []

    def measure(self, path, text, reserve):
        return {'fits': len(text) + reserve <= self.limit, 'input_tokens': len(text),
                'remaining_tokens': self.limit - len(text) - reserve}

    def generate(self, path, text, tokens, allowed):
        self.calls.append({'path': path, 'text': text, 'allowed': allowed})
        result = next(self.choices) if allowed else 'New visual observation.'
        return {'text': result, 'token_ids': [10, 11], 'output_tokens': 2}

    def invoke(self, action, case):
        self.calls.append({'expert': action.expert, 'region': action.region, 'case': case})
        return self.items or [item(action.expert, action.capability)]


def run(case, tmp_path, backend=None, **kwargs):
    backend = backend or Backend()
    incumbent = {'text': 'Existing answer.', 'token_ids': [7], 'seconds': .1,
                 'evidence': [item('legacy')], 'evidence_transport': {'preserved': True}}
    return run_case(case=case, specs=kwargs.pop('specs', {}),
                    seed_items=kwargs.pop('seed_items', []), incumbent=incumbent,
                    generate=backend.generate, measure=backend.measure, invoke=backend.invoke,
                    output_dir=tmp_path / 'crops', **kwargs)


def test_native_view_preserves_values_and_moves_audit():
    raw = item(payload={'disease': {'positive': False, 'unit': 'mm', 'value': 0.0},
                        'scores': [0.0, 1.0, None], 'negation': 'no effusion'})
    original = copy.deepcopy(raw)
    out = observation(raw, 'case')
    assert out['artifact'] == original and raw == original
    assert out['content']['payload'] == raw['payload']
    assert '/private/log' not in json.dumps(out['content'])
    assert 'revision' not in out['content']
    out['artifact']['payload']['negation'] = 'changed'
    assert raw == original


@pytest.mark.parametrize('cap', sorted(['classification', 'segmentation', 'detection', 'retrieval', 'generation']))
def test_every_native_capability_keeps_its_own_payload(cap):
    raw = item(capability=cap, payload={'arbitrary': {'not_known_to_core': ['no', 0, None]}})
    out = observation(raw, 'case')
    assert out['content']['payload'] == raw['payload']
    assert out['content']['capability'] == cap
    if cap == 'retrieval':
        assert out['content']['authority'] == 'source_analogy_only'


def test_array_bytes_not_sent_and_no_top_k():
    payload = {'scores': list(range(100)), 'mask': {'encoding': 'rle-row-major-zero-first',
               'size': [4, 4], 'counts': [16]}, 'custom': {'unit': 'mm'}}
    out = payload_view(payload)
    assert out['scores'] == payload['scores']
    assert 'counts' not in out['mask'] and out['mask']['dense_values_visible'] is False
    assert out['custom'] == payload['custom'] and payload['mask']['counts'] == [16]


@pytest.mark.parametrize('key', ['target_answers_used', 'target_masks_used', 'target_mask_used'])
def test_target_annotations_rejected(key):
    raw = item()
    raw['provenance'][key] = True
    with pytest.raises(ValueError, match='target'):
        native_dict(raw)


@pytest.mark.parametrize('bad', [float('nan'), float('inf')])
def test_invalid_numerics_rejected(bad):
    with pytest.raises(ValueError):
        observation(item(payload={'score': bad}), 'case')


def test_big_new_packet_does_not_veto_small_new_packet():
    base = observation(item('base'), 'case')
    big = observation(item('big', payload={'text': 'x' * 8000}), 'case')
    small = observation(item('small'), 'case')
    def measure(text, reserve):
        return {'fits': len(text) + reserve < 4000}
    selected, audit = pack_observations('q', [base], [big, small], measure, 64)
    assert selected == [base, small]
    assert audit['omitted'] == [{'ref': big['ref'], 'reason': 'token_budget', 'context': {'fits': False}}]
    assert audit['added_refs'] == [small['ref']]


def test_block_protected_overflow_before_generation():
    def measure(*_):
        return {'fits': False}
    selected, audit = pack_observations('q', [observation(item(), 'case')], [], measure, 64)
    assert selected is None and audit['reason'] == 'protected_context_exceeds_budget'
    assert audit['model_generation_started'] is False


def test_child_requires_presented_parent():
    parent = observation(item('big', payload={'text': 'x' * 8000}), 'case')
    child = observation(item('child'), 'case', parent=parent['ref'])
    selected, audit = pack_observations('q', [], [parent, child],
                                       lambda s, _: {'fits': len(s) < 3000}, 64)
    assert not selected
    assert audit['omitted'][-1]['reason'] == 'parent_not_presented'


def test_atomic_region_action_not_crop_then_inspect(case, tmp_path):
    b = Backend(['A0'])
    out = run(case, tmp_path, b, seed_items=[segment()], mode='agent', max_calls=1)
    roles = [c['role'] for c in out['agent_workflow']['calls']]
    assert roles == ['planner', 'region_reader', 'answer']
    assert out['agent_workflow']['candidate'] is not None
    planner = b.calls[0]['text']
    assert 'test structure' in planner and 'coordinate_system' in planner
    assert 'crop:0' not in planner
    reader = b.calls[1]
    assert case['question'] not in reader['text']
    assert Image.open(reader['path']).size[0] < 5
    assert b.calls[-1]['path'] == case['image']
    assert out['agent_workflow']['observations'][-1]['content']['parent'] is not None


def test_retrieval_does_not_require_segmentation(case, tmp_path):
    b = Backend()
    b.items = [item('retriever', 'retrieval', {'source_reference': 'source answer only'})]
    out = run(case, tmp_path, b, specs=spec('retrieval', 'retriever'), max_calls=1)
    assert b.calls[0]['expert'] == 'retriever' and b.calls[0]['region'] is None
    assert out['agent_workflow']['candidate'] is not None
    assert 'source_analogy_only' in b.calls[-1]['text']


def test_new_expert_name_no_core_change(case, tmp_path):
    b = Backend()
    out = run(case, tmp_path, b, specs=spec(name='brand_new_unseen_expert'), max_calls=1)
    assert out['agent_workflow']['candidate'] is not None
    assert b.calls[0]['expert'] == 'brand_new_unseen_expert'


def test_incompatible_expert_not_invoked(case, tmp_path):
    b = Backend()
    out = run(case, tmp_path, b, specs=spec(modalities=['fundus']))
    assert not b.calls and out['text'] == 'Existing answer.'
    assert out['agent_workflow']['delivery']['reason'] == 'no_new_observation'


def test_existing_evidence_not_requeried(case):
    obs = [observation(item(), case['id'])]
    actions, _ = available_actions(spec(), case, obs, set(), (4, 4), 2)
    assert actions == []


def test_prompted_expert_waits_for_real_region(case):
    actions, _ = available_actions(spec('segmentation'), case, [], set(), (4, 4), 2)
    assert not actions
    obs = [observation(segment(), case['id'])]
    actions, _ = available_actions(spec('segmentation'), case, obs, set(), (4, 4), 2)
    expert = [a for a in actions if a.operation == 'expert'][0]
    assert expert.region is not None and expert.parent == obs[0]['ref']


def test_seed_artifact_unchanged(case, tmp_path):
    raw = segment()
    original = copy.deepcopy(raw)
    run(case, tmp_path, seed_items=[raw], max_calls=1)
    assert original == raw


def test_stop_keeps_exact_incumbent_not_fake_candidate(case, tmp_path):
    b = Backend(['STOP'])
    out = run(case, tmp_path, b, seed_items=[segment()], mode='agent')
    assert out['text'] == 'Existing answer.' and out['token_ids'] == [7]
    assert out['evidence'] == [item('legacy')]
    assert out['agent_workflow']['candidate'] is None
    assert out['agent_workflow']['fallback'] == 'incumbent'


def test_invalid_action_is_not_silently_stopped(case, tmp_path):
    with pytest.raises(ValueError, match='unregistered'):
        run(case, tmp_path, Backend(['execute_shell']), seed_items=[segment()], mode='agent')


def test_oom_is_not_absence_or_success(case, tmp_path):
    b = Backend()
    def fail(*args):
        raise RuntimeError('CUDA out of memory')
    b.generate = fail
    with pytest.raises(RuntimeError, match='memory'):
        run(case, tmp_path, b, seed_items=[segment()])


def test_empty_expert_is_explicit_unknown(case, tmp_path):
    b = Backend()
    def unavailable(*args):
        raise ToolUnavailable('not applicable')
    b.invoke = unavailable
    out = run(case, tmp_path, b, specs=spec(), max_calls=1)
    assert out['agent_workflow']['events'][-1]['status'] == 'unavailable'
    assert out['text'] == 'Existing answer.'


def test_plugin_scope_mismatch_stops(case, tmp_path):
    b = Backend()
    b.items = [item('impostor')]
    with pytest.raises(ValueError, match='identity'):
        run(case, tmp_path, b, specs=spec(), max_calls=1)


@pytest.mark.parametrize('field', ['answer', 'reference', 'label', 'answer_type', 'target_mask'])
def test_target_metadata_cannot_cross_runtime(case, tmp_path, field):
    case[field] = 'secret'
    with pytest.raises(ValueError, match='label-free'):
        run(case, tmp_path)


def test_read_arm_generates_without_acquisition(case, tmp_path):
    b = Backend()
    out = run(case, tmp_path, b, seed_items=[item()], mode='read')
    assert [c['role'] for c in out['agent_workflow']['calls']] == ['answer']
    assert out['agent_workflow']['candidate'] is not None
    assert b.calls[0]['allowed'] is None
    assert b.calls[0]['text'].endswith(ANSWER_SUFFIX)


def test_whole_image_control_records_actual_observed_scope(case, tmp_path):
    out = run(case, tmp_path, seed_items=[segment()], max_calls=1, observation_view='whole')
    support = out['agent_workflow']['observations'][-1]['content']['support']
    assert support['box_xyxy_pixels'] == [0, 0, 4, 4] and support['view'] == 'whole'
    assert support['patient_orientation'] == 'unknown'


def test_character_limit_not_used_as_context_proxy(case, tmp_path):
    b = Backend()
    b.items = [item(payload={'narrative': 'no ' * 15000})]
    out = run(case, tmp_path, b, specs=spec(), max_calls=1)
    assert out['agent_workflow']['candidate'] is not None


def test_same_prediction_can_be_a_real_candidate(case, tmp_path):
    b = Backend()
    def generate(*_):
        return {'text': 'Existing answer.', 'token_ids': [7]}
    b.generate = generate
    out = run(case, tmp_path, b, specs=spec(), max_calls=1)
    assert out['text'] == 'Existing answer.'
    assert out['agent_workflow']['candidate'] is not None
    assert out['agent_workflow']['delivery']['model_generation_started'] is True


def test_payload_support_does_not_conflate_scores_with_accuracy():
    out = observation(item(payload={'confidence': .999, 'score_semantics': 'uncalibrated'}), 'case')
    assert out['content']['confidence_is_calibrated'] is False
    assert out['content']['payload']['confidence'] == .999


@pytest.mark.parametrize('specs', [{'e': {}}, spec(modalities=['*']), spec(capabilities=['arbitrary_code'])])
def test_bad_plugin_contracts_rejected(specs):
    with pytest.raises(ValueError):
        validate_specs(specs)


def test_cross_case_dependency_cannot_be_a_free_extra():
    parent = observation(item('p'), 'another_case')
    child = observation(item('c'), 'current_case', parent=parent['ref'])
    selected, audit = pack_observations('q', [], [child], lambda *_: {'fits': True}, 64)
    assert not selected and audit['omitted'][0]['reason'] == 'parent_not_presented'


def test_cross_case_protected_context_rejected():
    a = observation(item('a'), 'case_a')
    b = observation(item('b'), 'case_b')
    with pytest.raises(ValueError, match='different cases'):
        pack_observations('q', [a], [b], lambda *_: {'fits': True}, 64)


def test_no_scope_or_negation_removed_to_make_packet_fit():
    val = observation(item('huge', payload={'text': 'no finding ' * 1000}), 'case')
    selected, audit = pack_observations('q', [], [val], lambda s, r: {'fits': len(s) < 1000}, 64)
    assert not selected and audit['omitted'][0]['reason'] == 'token_budget'
    assert val['content']['payload']['text'].endswith('no finding ')


def test_missing_context_budget_stops_planner_truthfully(case, tmp_path):
    b = Backend()
    real_measure = b.measure
    def measure(path, text, reserve):
        return ({'fits': False} if 'Choose one useful' in text
                else real_measure(path, text, reserve))
    b.measure = measure
    out = run(case, tmp_path, b, seed_items=[segment()], mode='agent')
    call = out['agent_workflow']['calls'][0]
    assert call['status'] == 'budget_rejected' and not call['model_generation_started']
    assert out['agent_workflow']['candidate'] is None


def test_only_presented_evidence_recorded_as_candidate_input(case, tmp_path):
    b = Backend()
    b.limit = 8000
    b.items = [item('specialist', payload={'text': 'x' * 10000}),
               {**item('specialist'), 'evidence_id': 'second-small'}]
    out = run(case, tmp_path, b, specs=spec(), max_calls=1)
    assert out['agent_workflow']['candidate'] is not None
    assert len(out['evidence']) == 1
    assert out['evidence'][0]['evidence_id'] == 'second-small'
    assert len(out['agent_workflow']['observations']) == 2


def test_prepare_strips_task_labels_for_all_methods(case, tmp_path):
    from types import SimpleNamespace
    from merit_feddg.agent_protocol import file_hash
    from merit_feddg.plug_run import prepare
    manifest = tmp_path / 'manifest.jsonl'
    row = {**case, 'image_sha256': file_hash(case['image']), 'answer_type': 'closed'}
    manifest.write_text(json.dumps(row) + '\n')
    base = tmp_path / 'base'
    base.mkdir()
    config = {'generalist': {'id': 'frozen/base'}, 'experts': spec()}
    (base / 'protocol.json').write_text(json.dumps({'identity': 'base-id', 'n': 1,
        'shards_complete': True, 'config': config}))
    output = {'text': 'Old answer.', 'token_ids': [1], 'evidence': [item()],
        'input_modality': 'ct', 'generation_config': {'evidence_style': 'semantic',
        'block_tokens': 64, 'max_new_tokens': 64, 'vector_gate': 'off'}}
    (base / 'compact_rows.json').write_text(json.dumps({case['id']: output}))
    opts = tmp_path / 'opts.yaml'
    opts.write_text('max_calls: 1\n')
    args = SimpleNamespace(config=str(opts), manifest=str(manifest), base_run=str(base),
        incumbent='compact_rows', shard_index=0, shard_count=1, source_manifest=None,
        expert_registry=None)
    ready = prepare(args)
    assert 'answer_type' not in ready['cases'][0]
    assert ready['cases'][0]['modality'] == 'ct'
    output['trace'] = [{'request': {'question': 'different question'}}]
    (base / 'compact_rows.json').write_text(json.dumps({case['id']: output}))
    with pytest.raises(ValueError, match='question'):
        prepare(args)


def test_prepare_rejects_invalid_budgets_before_inference(tmp_path):
    from types import SimpleNamespace
    from merit_feddg.plug_run import prepare
    cfg = tmp_path / 'bad.yaml'
    cfg.write_text('max_calls: -1\n')
    with pytest.raises(ValueError, match='max_calls'):
        prepare(SimpleNamespace(config=str(cfg)))


def test_live_runner_callback_wiring(case, tmp_path, monkeypatch):
    """End-to-end orchestration using explicit fake models, not GPU inference."""
    import sys
    import types
    from dataclasses import dataclass, field
    torch = pytest.importorskip('torch')
    from merit_feddg.plug_run import live, DEFAULTS, METHODS

    @dataclass
    class Evidence:
        evidence_id: str
        expert_id: str
        capability: str
        scope: str
        payload: dict
        summary: str = ''
        confidence: float | None = None
        provenance: dict = field(default_factory=dict)

    @dataclass
    class Config:
        max_new_tokens: int = 64
        block_tokens: int = 64

    @dataclass
    class State:
        items: tuple = ()

    class Probe:
        model = torch.nn.Linear(1, 1)
        def context_token_budget(self, image, text, reserve):
            return {'fits': True, 'input_tokens': len(text), 'remaining_tokens': 100000}
        def generate_with_usage(self, image, text, max_new_tokens, allowed_texts=None):
            return {'text': 'STOP' if allowed_texts else 'new answer', 'token_ids': [1, 2]}

    probe = Probe()
    class Session:
        def __init__(self, probe, image, prompt, question, generation):
            assert 'Please answer Yes or No' not in prompt
            self.last_transport = {}
        def propose(self, state, length):
            self.last_transport = {'presented': [{'expert_id': i.expert_id,
                'evidence_id': i.evidence_id} for i in state.items]}
            return types.SimpleNamespace(tokens=(7,), finished=True)
        def decode(self, tokens):
            return 'fresh free baseline'

    class Pool:
        def __init__(self, *args, **kwargs):
            pass
        def reset_case(self):
            pass
        def clear(self):
            pass

    replacements = {
        'capabilities': {'CapabilityRequest': object, 'EvidenceItem': Evidence,
                         'validate_result': lambda result, *args: result},
        'capability_experts': {'CapabilityPool': Pool},
        'capability_runtime': {'NativeSession': Session, 'NativeState': State,
                               'ValueGenerationConfig': Config},
        'generalist_factory': {'generalist_provenance': lambda *a: {'id': 'fake'},
                               'load_generalist': lambda *a: probe},
        'open_study': {'model_provenance': lambda *a: {'id': 'fake'}},
        'evidence_need': {'presentation_items': lambda items, *args: items},
    }
    for name, attrs in replacements.items():
        module = types.ModuleType('merit_feddg.' + name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, 'merit_feddg.' + name, module)
    historical = {'evidence': [segment()], 'generation_config': {}, 'seconds': .2,
                  'text': 'old CE/OE answer', 'token_ids': [5]}
    data = {'config': {'generalist': {'id': 'fake'}}, 'options': {**DEFAULTS, 'max_calls': 1},
            'specs': {}, 'sources': [], 'references': {}, 'cases': [case],
            'base': {case['id']: historical}, 'identity': 'synthetic-test'}
    args = types.SimpleNamespace(artifacts=str(tmp_path), shard_index=0, shard_count=1, canary_cases=0)
    assert live(args, data, tmp_path)
    saved = json.loads((tmp_path / 'shards/0/results.json').read_text())
    assert saved['complete'] and set(saved['outputs']) == set(METHODS)
    assert saved['outputs']['incumbent'][case['id']]['text'] == 'fresh free baseline'
    fixed = saved['outputs']['plug_static'][case['id']]
    assert fixed['agent_workflow']['delivery']['stage'] == 'generated'
    assert fixed['agent_workflow']['candidate'] is not None
    assert saved['outputs']['plug_agent'][case['id']]['text'] == 'fresh free baseline'
    assert historical['text'] == 'old CE/OE answer'
