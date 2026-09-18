from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from merit_feddg.capabilities import CapabilityRequest, validate_result
from merit_feddg.expert_coverage import load_config, select_experts, source_catalog
from merit_feddg.experts import native_coverage

CONFIG = Path(__file__).resolve().parents[1] / 'configs/expert_coverage_v1.yaml'


def select(modality, question, **kwargs):
    config = load_config(CONFIG)
    return select_experts(config, {'modality': modality, 'question': question,
                                  'input_kind': kwargs.pop('input_kind', '2d')},
                          set(config['new']) | set(config['legacy']), **kwargs)


def test_native_scope_positive_and_negative():
    assert select('fundus', 'Is glaucoma present?')['selected'] == ['flair_retina']
    assert select('oct', 'Is glaucoma present?')['selected'] == []
    assert select('dermatology', 'Is a blister present?')['selected'] == ['monet_skin']
    assert select('dermatology', 'What treatment should be used?')['selected'] == []
    assert select('ultrasound', 'Is a breast lesion present?')['selected'] == ['unimed_breast_ultrasound']
    assert 'unimed_breast_ultrasound' not in select('ultrasound', 'Is heart disease present?')['selected']


def test_unknown_is_not_incorrect_and_not_blanket_permission():
    r = select('pathology', 'unparsed question')
    assert r['selected'] == ['quilt_pathology']
    assert r['requested_attributes'] == ['observation']
    assert r['confidence'] is None
    assert 'conch_tissue' not in r['selected']


def test_volume_is_not_2d_coverage():
    r = select('ct', 'Which organ is shown?', input_kind='volume')
    assert r['selected'] == []
    assert all(x['reason'] == 'unsupported_input_kind' for x in r['audit'].values())


def test_explicit_need_does_not_claim_automatic_parser_solved():
    r = select('pathology', 'lung image', requested=['finding_identity'])
    assert r['selected'] == ['unimed_lung_histology']
    assert r['automatic_semantics'] == [] and r['explicit_request']


def test_resource_unavailable_and_minimal_cover():
    config = load_config(CONFIG)
    row = {'modality': 'cxr', 'question': 'Is a finding present?'}
    r = select_experts(config, row, {'chexagent_description'})
    assert r['selected'] == ['chexagent_description']
    assert r['audit']['cxr_findings']['reason'] == 'resource_or_native_output_unavailable'
    assert len(select('cxr', row['question'])['selected']) == 1


def test_catalog_is_upstream_literal_not_question_candidates(tmp_path):
    (tmp_path / 'constants.py').write_text("BANK={'x':['image of x','second prompt']}\n")
    result = source_catalog({'unimed_source': str(tmp_path)}, {'catalog_symbol': 'BANK'})
    assert result == [{'name': 'x', 'prompt': 'image of x'}]
    with pytest.raises(ValueError):
        source_catalog({'unimed_source': str(tmp_path)}, {'catalog_symbol': 'MISSING'})


def test_missing_weights_fail_closed(tmp_path):
    with pytest.raises(FileNotFoundError):
        native_coverage.verify_resource(tmp_path, 'monet')


def test_old_position_buffer_requires_exact_equality():
    import torch
    model = torch.nn.Module()
    model.tower = torch.nn.Module()
    model.tower.embeddings = torch.nn.Module()
    model.tower.embeddings.register_buffer('position_ids', torch.arange(4)[None], persistent=False)
    assert native_coverage.strict_state(model, {'tower.embeddings.position_ids': torch.arange(4)[None]})
    with pytest.raises(ValueError):
        native_coverage.strict_state(model, {'tower.embeddings.position_ids': torch.zeros(1, 4)})
    with pytest.raises(RuntimeError):
        native_coverage.strict_state(model, {'unrecognized_weight': torch.ones(1)})


def test_native_result_and_no_probability_claim(tmp_path):
    image = tmp_path / 'image.png'
    Image.new('RGB', (16, 16)).save(image)
    class Torch:
        pass
    class Backend:
        device = type('Device', (), {'type': 'cpu'})()
        torch = Torch()
        load_seconds = 0.0
        def __init__(self):
            self.audit = {'revision': 'test-only'}
        def scores(self, image, prompts):
            return np.array([.1, -.2]), np.array([1., -2.])
    expert = native_coverage.CatalogExpert.__new__(native_coverage.CatalogExpert)
    expert.catalog = (('a', 'a'), ('b', 'b'))
    expert.expert_id, expert.scope, expert.modalities = 'test', 'fixed', ('fundus',)
    expert.kind, expert.source_family = 'flair', 'test'
    expert.backend, expert.new_load = Backend(), False
    request = CapabilityRequest('id', str(image), 'question', 'fundus', 'open_vqa',
                                'external', 'group', 'classification', scope='fixed')
    result = validate_result(expert.infer(request), 'test', request)
    assert result.items[0].confidence is None
    assert result.items[0].payload['catalog'][1]['similarity'] == -.2
    assert result.items[0].provenance['target_candidates_used'] is False
    changed = CapabilityRequest(**{**asdict(request), 'modality': 'oct'})
    assert not expert.infer(changed).items


def test_no_disease_specific_blacklist_or_test_condition():
    # Config only declares published capability vocabularies. No dataset IDs.
    text = CONFIG.read_text()
    assert 'pathvqa-test-' not in text
    assert load_config(CONFIG)['max_calls'] == 3
