from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from PIL import Image
from test_capability_runtime import Probe

from merit_feddg.capabilities import EvidenceItem
from merit_feddg.capability_experts import encode_binary_mask
from merit_feddg.capability_runtime import NativeSession, NativeState, ValueGenerationConfig
from merit_feddg.evidence_admission import EvidenceRequest, delivery_view


@pytest.fixture
def specs():
    return yaml.safe_load(Path('configs/llava_med_capabilities.yaml').read_text())['experts']


def request(entities=('Effusion',), dimensions=('finding_presence',), question='query'):
    return EvidenceRequest(question, entities, frozenset(dimensions), 'cxr', 'open_vqa')


def findings():
    return EvidenceItem('id', 'cxr_findings', 'classification', 'cxr_findings', {
        'findings':[{'finding':'Effusion', 'score':.123}, {'finding':'Pneumonia', 'score':.456}],
        'score_semantics':'uncalibrated_independent_sigmoid',
        'unlisted_findings':'unknown', 'old_audit':'AUDIT_SENTINEL'},
        summary='Pneumonia SECRET_SUMMARY', provenance={'audit':'AUDIT_SENTINEL'})


def anatomy():
    mask = encode_binary_mask(np.ones((2, 2), dtype=np.uint8))
    return EvidenceItem('mask', 'cxr_anatomy', 'segmentation', 'thoracic_anatomy', {
        'structures':[{'anatomical_structure':name, 'mask':mask,
                       'mask_coordinate_system':'model_grid_of_center_crop'}
                      for name in ('Left Lung', 'Right Lung', 'Heart')],
        'disease_or_lesion_segmentation':False,
        'image_transform':{'crop_box_xyxy_normalized':[0, 0, 1, 1]}})


def test_field_selection_immutable_and_not_renormalized(specs):
    item = findings()
    snapshot = deepcopy(item)
    views, audit = delivery_view([item], question='query', request=request(), specs=specs, mode='enforce')
    assert item == snapshot
    assert views[0].payload['findings'] == [{'finding':'Effusion', 'score':.123}]
    assert views[0].payload['score_semantics'] == item.payload['score_semantics']
    assert not views[0].summary
    assert 'audit' not in views[0].provenance
    views[0].payload['findings'][0]['score'] = 0
    assert item == snapshot
    assert audit['decisions'][0]['would_admit']


@pytest.mark.parametrize('req', [None, request(('Tuberculosis',)),
    request(('Pneumothorax', 'Tuberculosis')), request(dimensions=('finding_presence', 'laterality')),
    request(question='different'), replace(request(), complete=False)])
def test_denied_and_incomplete_requests(specs, req):
    views, _ = delivery_view([findings()], question='query', request=req, specs=specs, mode='enforce')
    assert not views


def test_only_actual_configured_structures(specs):
    for names, dimensions, expected in [
        (('Left Lung',), ('location', 'laterality'), True),
        (('Aorta',), ('location',), False),
        (('Tumor', 'Heart'), ('location',), False),
        (('Heart',), ('measurement',), False),
        (('Heart',), ('finding_presence',), False),
    ]:
        views, _ = delivery_view([anatomy()], question='query', request=request(names, dimensions), specs=specs, mode='enforce')
        assert bool(views) == expected
        if expected:
            assert [s['anatomical_structure'] for s in views[0].payload['structures']] == ['Left Lung']
    specs['cxr_anatomy']['structures'] = ['Left Lung', 'Aorta']
    views, audit = delivery_view([anatomy()], question='query', request=request(('Aorta',), ('location',)), specs=specs, mode='enforce')
    assert not views
    assert audit['decisions'][0]['reason'] == 'requested_entry_not_returned'


@pytest.mark.parametrize('mode', ['legacy', 'audit'])
def test_non_enforcing_modes_preserve_raw(specs, mode):
    item = findings()
    views, audit = delivery_view([item], question='query', request=request(('Tuberculosis',)), specs=specs, mode=mode)
    assert views == (item,)
    if mode == 'audit':
        assert not audit['decisions'][0]['would_admit']


class Packet(list):
    def __init__(self, items):
        super().__init__(items)
        self.sources = [(i.expert_id, i.evidence_id) for i in items]
        self.rejected = []


class SpatialProbe(Probe):
    def __init__(self):
        super().__init__()
        self.tensor_bridge = SimpleNamespace(training_free=False)
        self.spatial_inputs = []

    def context_token_budget(self, image, prompt, reserve):
        return {'fits':True, 'remaining_tokens':100000, 'input_tokens':len(prompt), 'reserved_tokens':reserve}

    def tensor_packet(self, items, image):
        self.spatial_inputs.append(tuple(items))
        return Packet(items)

    def new_tensor_answer_session(self, image, prompt, items, **kwargs):
        self.spatial_inputs.append(tuple(items))
        return self.new_answer_session(image, prompt)


@pytest.mark.parametrize('channel', ['native', 'semantic', 'tensor', 'semantic_spatial'])
@pytest.mark.parametrize('legal', [True, False])
def test_all_answer_channels_use_same_view(specs, channel, legal):
    probe = SpatialProbe()
    semantic = channel.startswith('semantic')
    cfg = ValueGenerationConfig(admission_mode='enforce', evidence_style='semantic' if semantic else channel,
        token_budgeted_evidence=semantic, visual_views=0,
        semantic_spatial=channel == 'semantic_spatial')
    req = request(('Left Lung',), ('location',)) if legal else request(('Heart',), ('measurement',))
    session = NativeSession(probe, Image.new('RGB', (4, 4)), 'query', 'query', cfg,
                            authority_specs=specs, evidence_request=req)
    raw = anatomy()
    snapshot = deepcopy(raw)
    state = NativeState(items=(raw,))
    session.context(state)
    session._evidence_session(state)
    session.propose(state, 2)
    assert raw == snapshot
    assert session.last_transport['admission']['delivered_count'] == int(legal)
    for items in probe.spatial_inputs:
        assert bool(items) == legal
        if legal:
            assert [s['anatomical_structure'] for s in items[0].payload['structures']] == ['Left Lung']
    for _, prompt in probe.contexts:
        assert 'Right Lung' not in prompt
        assert 'would_admit' not in prompt


def test_cached_session_rechecks_changed_request(specs):
    probe = SpatialProbe()
    session = NativeSession(probe, 'image', 'query', 'query', ValueGenerationConfig(
        admission_mode='enforce', evidence_style='semantic', token_budgeted_evidence=True, visual_views=0),
        authority_specs=specs, evidence_request=request())
    state = NativeState(items=(findings(),))
    session.propose(state, 2)
    assert session.last_transport['presented']
    assert 'Pneumonia' not in probe.contexts[-1][1]
    assert 'AUDIT_SENTINEL' not in probe.contexts[-1][1]
    session.evidence_request = request(('Tuberculosis',))
    session.propose(state, 2)
    assert not session.last_transport['presented']


def test_visual_overlay_uses_filtered_structures(specs):
    image = Image.new('RGB', (16, 16))
    session = NativeSession(SpatialProbe(), image, 'query', 'query',
        ValueGenerationConfig(admission_mode='enforce', visual_views=1),
        authority_specs=specs, evidence_request=request(('Heart',), ('measurement',)))
    item = anatomy()
    item.payload['image_transform'].update(original_size_hw=[16, 16], model_size_hw=[2, 2])
    state = NativeState(items=(item,))
    images, prompt = session.context(state)
    assert images is image and prompt == 'query'
    assert not session.view_metadata
    session.evidence_request = request(('Left Lung',), ('location',))
    images, prompt = session.context(state)
    assert isinstance(images, list) and len(images) == 2
    assert session.view_metadata
    assert 'Right Lung' not in prompt


def test_audit_metadata_never_enters_prompt(specs):
    probe = SpatialProbe()
    cfg = ValueGenerationConfig(evidence_style='semantic', token_budgeted_evidence=True, visual_views=0)
    prompts = []
    for mode in ('legacy', 'audit'):
        session = NativeSession(probe, 'image', 'query', 'query', replace(cfg, admission_mode=mode),
                                authority_specs=specs, evidence_request=request(('Tuberculosis',)))
        _, prompt = session.context(NativeState(items=(findings(),)))
        prompts.append(prompt)
        assert 'would_admit' not in prompt and 'native_entity_not_declared' not in prompt
    assert prompts[0] == prompts[1]


def test_target_mask_guard_is_not_erased_by_delivery_view(specs):
    item = replace(anatomy(), provenance={'target_masks_used':True})
    views, audit = delivery_view([item], question='query', request=request(('Left Lung',), ('location',)),
                                 specs=specs, mode='enforce')
    assert not views
    assert audit['decisions'][0]['reason'] == 'target_annotation_provenance_forbidden'
