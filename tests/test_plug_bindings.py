"""Support-preserving plugin contracts, lossless views and actual callback paths.

All clinical/model outputs below are synthetic; these tests make NO efficacy claim.
"""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from merit_feddg.plug_observe import (
    available_actions, answer_prompt, model_records, observation, pack_observations,
    planner_prompt, plugin_regions, run_case, validate_observation,
)
from merit_feddg.plug_run import prepare


def raw(name='e', capability='classification', **changes):
    return {'expert_id': name, 'evidence_id': name + ':0', 'capability': capability,
            'scope': 'native_scope', 'payload': {'score': .2, 'negation': 'not present'},
            'summary': 'uncalibrated observation', 'confidence': None,
            'provenance': {'revision': 'v1'}, **changes}


def detector(name='detector', **change):
    box = {'label': 'native structure', 'box': [.2, .2, .7, .8],
           'coordinate_system': 'original_image_normalized_xyxy', **change}
    return raw(name, 'detection', payload={'detections': [box]})


def merge_dict(left, right):
    out = copy.deepcopy(left)
    for key, value in right.items():
        out[key] = merge_dict(out[key], value) if (
            key in out and isinstance(out[key], dict) and isinstance(value, dict)) else value
    return out


def expand(value):
    if isinstance(value, dict):
        if set(value) == {'shared_fields', 'entries'}:
            return [merge_dict(expand(value['shared_fields']), expand(row))
                    for row in value['entries']]
        return {key: expand(val) for key, val in value.items()}
    return [expand(v) for v in value] if isinstance(value, list) else value


def strict_json(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


@pytest.mark.parametrize('field,new', [('scope', 'other_scope'), ('capability', 'generation'),
                                      ('summary', 'other meaning'), ('confidence', .8),
                                      ('provenance', {'revision': 'v2'})])
def test_observation_identity_includes_semantic_and_source_fields(field, new):
    old = observation(raw(), 'case')
    changed = observation(raw(**{field: new}), 'case')
    assert old['ref'] != changed['ref']


def test_identical_numbers_on_different_regions_are_not_deduplicated():
    a = observation(raw(), 'case', support={'box': [0, 0, .5, 1]})
    b = observation(raw(), 'case', support={'box': [.5, 0, 1, 1]})
    assert a['ref'] != b['ref']
    selected, _ = pack_observations('q', [a], [b], lambda *_: {'fits': True}, 64)
    assert len(selected) == 2


@pytest.mark.parametrize('field', ['payload', 'scope', 'support', 'native_summary'])
def test_detached_content_is_rejected_before_generation(field):
    value = observation(raw(), 'case')
    value['content'][field] = 'tampered'
    with pytest.raises(ValueError, match='binding'):
        pack_observations('q', [value], [], lambda *_: {'fits': True}, 64)


def test_parent_must_be_present_in_model_view():
    value = observation(raw(), 'case', parent='missing')
    with pytest.raises(ValueError, match='parent'):
        answer_prompt('q', [value])


@pytest.mark.parametrize('support', [{}, [], '', 0])
def test_invalid_explicit_support_rejected(support):
    with pytest.raises(ValueError, match='support'):
        observation(raw(), 'case', support=support)


def test_shared_view_exactly_preserves_types_scopes_and_dependencies():
    vals = [False, 0, 0.0, None, 'no lesion']
    observations = []
    for i, value in enumerate(vals):
        r = raw(evidence_id=f'e:{i}', payload={'value': value, 'unit': 'mm', 'unknown': None})
        parent = observations[0]['ref'] if i else None
        observations.append(observation(r, 'case', parent=parent,
            support={'input': 'current_image', 'source_region': f'R{i}'}))
    before = copy.deepcopy(observations)
    plain = model_records(observations, 'plain')
    assert strict_json(expand(model_records(observations, 'shared'))) == strict_json(plain)
    assert observations == before
    assert plain[1]['parent'] == 'O0'


def test_shared_view_preserves_missing_vs_null_and_unknown_fields():
    a = observation(raw('a', payload={'unit': 'mm', 'x': {'negative': True}}), 'case')
    b = observation(raw('b', payload={'unit': 'mm', 'missing': None,
                                     'x': {'negative': True, 'custom': [3, 'no']}}), 'case')
    assert strict_json(expand(model_records([a, b], 'shared'))) == strict_json(model_records([a, b]))


@pytest.mark.parametrize('key', ['shared_fields', 'entries', 'columns', 'rows'])
def test_native_layout_keys_are_never_reinterpreted(key):
    values = [observation(raw(str(i), payload={key: [False, 1, None]}), 'case') for i in range(2)]
    assert model_records(values, 'shared') == model_records(values, 'plain')


def test_parent_binding_has_same_short_id_in_planner_and_answer():
    p = observation(raw('p'), 'case')
    c = observation(raw('c'), 'case', parent=p['ref'])
    for prompt in (answer_prompt('q', [p, c], encoding='shared'),
                   planner_prompt('q', [], [p, c], encoding='shared')):
        assert p['ref'] not in prompt and c['ref'] not in prompt
        assert 'O0' in prompt and 'parent' in prompt


def test_shared_budget_uses_actual_measurement_and_does_not_drop_protected():
    payload = {'findings': [{'name': f'x{i}', 'score': i / 100,
                            'score_semantics': 'not_a_calibrated_probability',
                            'absence_not_established': True} for i in range(40)]}
    values = [observation(raw(str(i), payload=payload), 'case') for i in range(3)]
    plain = len(answer_prompt('q', values, encoding='plain'))
    shared = len(answer_prompt('q', values, encoding='shared'))
    assert shared < plain
    limit = (plain + shared) // 2
    def measure(text, reserve):
        return {'fits': len(text) <= limit, 'input_tokens': len(text)}
    selected, audit = pack_observations('q', values, [], measure, 64, encoding='shared')
    assert selected == values and audit['reason'] == 'packed'
    selected, audit = pack_observations('q', values, [], measure, 64, encoding='plain')
    assert selected is None and audit['reason'] == 'protected_context_exceeds_budget'


@pytest.mark.parametrize('box', [[1, 0, 0, 1], [-1, 0, 1, 1], [0, 0, float('nan'), 1],
                                [True, 0, 1, 1], [0, 0, 1]])
def test_malformed_detector_geometry_is_not_a_region(box):
    regions, audit = plugin_regions([detector(box=box)], (20, 20), 2)
    assert not regions and audit


def test_detector_needs_explicit_coordinate_convention():
    regions, audit = plugin_regions([detector(coordinate_system='crop_pixels')], (20, 20), 2)
    assert not regions and 'coordinates' in audit[0]['reason']


def test_empty_detector_is_unknown_not_whole_image():
    regions, _ = plugin_regions([raw(capability='detection', payload={'detections': []})], (20, 20), 2)
    assert regions == []


def test_detector_region_bound_to_native_result_and_never_confirms_disease():
    regions, audit = plugin_regions([detector()], (20, 20), 2)
    assert not audit and len(regions) == 1
    assert regions[0]['source_index'] == 0
    assert not regions[0]['object_presence_confirmed']
    assert regions[0]['geometry_kind'] == 'predicted_box'


def test_roi_output_does_not_disable_whole_image_tool():
    value = observation(raw(), 'case', support={'input': 'current_image', 'prompt_region': [.2,.2,.4,.4]})
    specs = {'e': {'id': 'frozen/e', 'capabilities': ['classification'],
                   'modalities': ['ct'], 'scope': 'native_scope', 'description': 'native scores'}}
    actions, _ = available_actions(specs, {'modality': 'ct', 'task': 'open_vqa'}, [value], set(), (20,20), 2)
    assert len(actions) == 1 and actions[0].region is None


@pytest.mark.parametrize('encoding', ['plain', 'shared'])
def test_new_detector_plugin_flows_to_atomic_reader_then_real_candidate(tmp_path, encoding):
    path = tmp_path / 'image.png'
    Image.fromarray(np.arange(1200, dtype=np.uint8).reshape(20,20,3)).save(path)
    case = {'id': 'x', 'image': str(path), 'question': 'What is visible?',
            'modality': 'ct', 'task': 'open_vqa'}
    name = 'previously_unseen_detector'
    specs = {name: {'id': 'frozen/new', 'scope': 'native_scope', 'capabilities': ['detection'],
                    'modalities': ['ct'], 'description': 'predict named boxes'}}
    roles = []
    def generate(path, prompt, tokens, allowed):
        roles.append((path, prompt, allowed))
        return {'text': 'visible structure', 'token_ids': [2, 3]}
    def invoke(action, current):
        assert action.expert == name and current == case
        return [detector(name)]
    seed = []
    out = run_case(case=case, specs=specs, seed_items=seed,
        incumbent={'text': 'old', 'token_ids': [1]}, generate=generate,
        measure=lambda *a: {'fits': True}, invoke=invoke, output_dir=tmp_path / 'crops',
        max_calls=2, content_encoding=encoding)
    assert [v['role'] for v in out['agent_workflow']['calls']] == ['region_reader', 'answer']
    assert out['agent_workflow']['delivery']['model_generation_started']
    assert out['agent_workflow']['candidate'] is not None
    assert roles[-1][0] == str(path) and roles[0][0] != str(path)
    assert out['agent_workflow']['observations'][1]['content']['parent'] == out['agent_workflow']['observations'][0]['ref']
    assert seed == []


def test_unsupported_encoding_rejected_before_loading_models(tmp_path):
    p = tmp_path / 'bad.yaml'
    p.write_text('content_encoding: magical\n')
    with pytest.raises(ValueError, match='content_encoding'):
        prepare(SimpleNamespace(config=str(p)))


def test_observation_validates_without_mutating_artifact():
    r = raw()
    value = observation(r, 'case')
    before = copy.deepcopy(value)
    validate_observation(value)
    assert value == before and r == value['artifact']


def test_plain_model_prompt_preserves_existing_representation():
    from merit_feddg.plug_observe import RULES, ANSWER_SUFFIX
    values = [observation(raw('x'), 'case'), observation(raw('y'), 'case')]
    content = [{'observation': f'O{i}', **v['content']} for i, v in enumerate(values)]
    expected = RULES + json.dumps(content, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    expected += '\nQuestion: q' + ANSWER_SUFFIX
    assert answer_prompt('q', values) == expected


def test_detector_cannot_bypass_case_level_target_annotation_guard(tmp_path):
    path = tmp_path / 'image.png'
    Image.new('RGB', (20,20)).save(path)
    d = detector()
    d['provenance']['target_mask_used'] = True
    with pytest.raises(ValueError, match='target'):
        run_case(case={'id':'x','image':str(path),'question':'q','modality':'ct','task':'open_vqa'},
            specs={}, seed_items=[d], incumbent={}, generate=None, measure=None,
            invoke=None, output_dir=tmp_path)


def test_detector_and_segmentation_share_one_region_budget():
    seg = raw('seg', 'segmentation', payload={'structures': [
        {'label': 'structure', 'mask_coordinate_system': 'original_image',
         'mask': {'encoding':'rle-row-major-zero-first','size':[4,4],'counts':[5,2,2,2,5]}}]})
    regions, audit = plugin_regions([seg, detector()], (4,4), 1)
    assert len(regions) == 1
    assert audit[-1]['reason'] == 'region_budget'


def test_new_packet_overflow_does_not_erase_old_binding():
    p = observation(raw('p'), 'case')
    c = observation(raw('c', payload={'text': 'no '*10000}), 'case', parent=p['ref'])
    selected, audit = pack_observations('q', [p], [c],
        lambda text, r: {'fits':len(text)<4000}, 64, encoding='shared')
    assert selected == [p] and audit['omitted'][0]['reason'] == 'token_budget'
    assert audit['content_encoding'] == 'shared'
