from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from test_capability_runtime import Probe, spec
from test_capability_runtime import setup as setup  # noqa: PLC0414
from test_semantic_spatial import measure
from test_vector_gate import Candidate, Verifier

from merit_feddg.capabilities import EvidenceItem
from merit_feddg.capability_runtime import NativeState, ValueGenerationConfig
from merit_feddg.claim_evidence import attribute_check, native_entries
from merit_feddg.claim_gate import assess_claim_support, paired_local_views
from merit_feddg.evidence_transport import pack_records
from merit_feddg.io import load_experiment_yaml
from merit_feddg.matched_evaluation import experiment_arms
from merit_feddg.semantic_evidence import semantic_prompt, semantic_records
from merit_feddg.spatial_evidence import encode_soft_mask, spatial_packet


def finding_item(n=18):
    return EvidenceItem('parent', 'xrv', 'classification', 'cxr', {
        'findings': [{'finding': f'label-{i}', 'score': i / 100} for i in range(n)],
        'score_semantics': 'uncalibrated_independent_sigmoid'}, summary='long parent summary')


def test_native_boundary_split_preserves_every_scalar_and_source_without_mutating_parent():
    parent = finding_item()
    original = asdict(parent)
    entries = native_entries((parent,))
    assert len(entries) == 18 and len({v.evidence_id for v in entries}) == 18
    assert [v.payload['findings'][0] for v in entries] == parent.payload['findings']
    assert all(v.payload['score_semantics'] == parent.payload['score_semantics'] for v in entries)
    assert all(v.provenance['native_entry']['parent_id'] == parent.evidence_id for v in entries)
    entries[0].payload['findings'][0]['score'] = .999
    assert asdict(parent) == original


def test_entries_allow_partial_delivery_when_whole_packet_cannot_fit():
    parent = finding_item()
    # Token budget deliberately lies between one entry and the entire packet.
    render = lambda records: semantic_prompt('Q', records)
    limit = len(render(semantic_records(native_entries((parent,))[:1]))) + 20
    measure_budget = lambda prompt, reserve: {'fits': len(prompt)+reserve <= limit}
    whole, _ = pack_records(semantic_records((parent,)), render, measure_budget, max_chars=20000, reserve_tokens=1)
    entries, audit = pack_records(semantic_records(native_entries((parent,))), render, measure_budget,
                                 max_chars=20000, reserve_tokens=1)
    assert not whole and len(entries) == 1 and len(audit['omitted']) == 17


@pytest.mark.parametrize('question,text,allowed', [
    ('Which side is the opacity on?', 'No', False),
    ('Where is the opacity?', 'Yes.', False),
    ('Is cardiomegaly present?', 'No', True),
    ('What abnormalities are present?', 'None', True),
    ('Which side is the opacity on?', 'left', True),
    ('Describe the image', 'cxr_description', False),
    ('Describe the image', '', False),
    ('Describe the image', 'No pleural effusion.', True),
    ('影像显示什么？', 'unstructured observation', True)])
def test_attribute_check_is_not_ce_oe_output_grammar(question, text, allowed):
    item = EvidenceItem('a', 'e', 'generation', 'cxr', {'generated_text': text})
    assert attribute_check(item, question)['passed'] == allowed
    assert not attribute_check(item, question)['correctness_established']


def test_local_controls_preserve_grid_mass_and_only_edit_selected_pixels():
    rng = np.random.default_rng(42)
    image = Image.fromarray(rng.integers(0, 255, (24, 24, 3), dtype=np.uint8))
    mask = np.zeros((24, 24), dtype=np.float32)
    mask[:6, :6] = 1
    removed, control, audit = paired_local_views(image, mask.reshape(1, -1))
    assert audit['support_mass'] == audit['control_mass'] == 36
    assert audit['control_overlap'] == 0
    assert np.array_equal(np.asarray(removed)[mask == 0], np.asarray(image)[mask == 0])
    assert not np.array_equal(np.asarray(removed), np.asarray(control))
    with pytest.raises(ValueError, match='distinct'):
        paired_local_views(image, np.ones((1, 576)))


def test_atomization_keeps_spatial_transform_and_mask_binding():
    mask = np.zeros((8, 8), dtype=np.float32)
    mask[:4, :4] = 1
    item = finding_item(1)
    item.payload['findings'][0]['spatial_support'] = {
        'soft_mask': encode_soft_mask(mask), 'mask_coordinate_system': 'original_image'}
    entries = native_entries((item,))
    before = spatial_packet((item,), (8, 8), weighting='equal')
    after = spatial_packet(entries, (8, 8), weighting='equal')
    assert len(before) == len(after) == 1
    assert np.array_equal(before.regions, after.regions)
    assert after.sources[0][1] == entries[0].evidence_id


def test_new_arms_isolate_mechanisms_without_partition_or_answer_contract():
    config = load_experiment_yaml('configs/matched_native_claims.yaml')
    decoder = ValueGenerationConfig(**config['capability_value']['generation'])
    arms = experiment_arms(decoder, 'native_claims')
    assert len(arms) == 6
    assert all(a.max_new_tokens == a.block_tokens == a.vector_gate_probe_tokens == 64 for a in arms.values())
    assert all(a.behavior_probe == 'off' and not a.uncertainty_from_probe for a in arms.values())
    assert replace(arms['entry_all'], native_entry_transport=False) == arms['semantic_all']
    assert replace(arms['entry_filtered'], claim_attribute_filter=False) == arms['entry_all']
    assert replace(arms['hybrid_all'], semantic_spatial=False) == arms['entry_filtered']
    assert replace(arms['hybrid_gate'], vector_gate='off') == arms['hybrid_all']


def test_runtime_preserves_raw_packet_and_exact_prefix(setup, monkeypatch):
    monkeypatch.setattr(Probe, 'context_token_budget', measure, raising=False)
    build, _, _ = setup
    runtime, probe, _pool = build(specs={'A': spec()}, visual_views=0, evidence_style='semantic',
        token_budgeted_evidence=True, native_entry_transport=True, claim_attribute_filter=True,
        evidence_order='acquisition', max_evidence_chars=20000)
    state = NativeState(prefix=(9,))
    result, event = runtime.execute(state, runtime.descriptors(state)[0])
    assert result.prefix == (9,) and len(result.items) == 1
    assert event['native_evidence'][0]['evidence_id'] == 'A:native'
    assert result.items[0].evidence_id == 'A:native/native/catalog/0'
    assert not probe.proposals and not probe.controls


def test_local_gate_scores_paired_removals_without_nli_or_original_image_veto():
    image = Image.new('RGB', (24, 24))
    regions = np.zeros((1, 576), dtype=np.float32)
    regions[0, :24] = 1
    class Packet:
        rejected = ()
        def __len__(self): return 1
    packet = Packet()
    packet.regions = regions
    verifiers = iter((Verifier((.1, .6, .3)), Verifier((.1, .8, .1))))
    probe = SimpleNamespace(tensor_packet=lambda *a: packet, new_answer_session=lambda *a: next(verifiers))
    session = SimpleNamespace(probe=probe, image=image, prompt='Q', config=SimpleNamespace(
        vector_gate_probe_tokens=2, vector_gate_min_gain=1e-6, max_new_tokens=8),
        _evidence_session=lambda state: Candidate((2,) if state.items else (1,)))
    result = assess_claim_support(session, NativeState(prefix=(9,)), finding_item(1))
    assert result['accepted'] and result['local_removal_gain'] > 0
    assert result['control_removal_margin'] < 0
    assert 'image_gain' not in result and 'semantic_change' not in result
    assert result['candidate_generation_calls'] == 2 and not result['correctness_guaranteed']


def test_rectangular_image_uses_unpadded_roi_coordinates():
    rng = np.random.default_rng(7)
    image = Image.fromarray(rng.integers(0, 255, (12, 24, 3), dtype=np.uint8))
    mask = np.zeros((24, 24), dtype=np.float32)
    # Original image occupies square rows 6:18. Only its first two rows affected.
    mask[6:8, :4] = 1
    removed, _, audit = paired_local_views(image, mask.reshape(1, -1))
    assert audit['support_mass'] == audit['control_mass'] == 8
    assert np.array_equal(np.asarray(removed)[2:], np.asarray(image)[2:])
    assert not np.array_equal(np.asarray(removed)[:2, :4], np.asarray(image)[:2, :4])


def test_per_case_budget_keeps_semantics_without_spatial_bypass(setup, monkeypatch):
    from merit_feddg.capabilities import CapabilityResult
    from merit_feddg.capability_runtime import CapabilityRuntime, NativeSession
    monkeypatch.setattr(Probe, 'context_token_budget', measure, raising=False)
    build, row, image = setup
    _, probe, pool = build()
    probe.tensor_bridge = SimpleNamespace(training_free=True)
    probe.tensor_packet = lambda items, image: spatial_packet(items, (32, 24), weighting='equal')
    probe.new_tensor_answer_session = lambda *args, **kw: Candidate((1,))
    config = ValueGenerationConfig(max_new_tokens=8, block_tokens=8, visual_views=0,
        evidence_style='semantic', token_budgeted_evidence=True, native_entry_transport=True,
        claim_attribute_filter=True, evidence_order='acquisition', semantic_spatial=True,
        vector_gate='claim_support', claim_gate_max_checks=1, max_evidence_chars=20000)
    item = finding_item(3)
    mask = np.zeros((24, 32), dtype=np.float32)
    mask[:6, :6] = 1
    for entry in item.payload['findings']:
        entry['spatial_support'] = {'soft_mask': encode_soft_mask(mask), 'mask_coordinate_system': 'original_image'}
    item = replace(item, expert_id='A', scope='native-classification')
    pool.infer = lambda expert, request: CapabilityResult('A', 'classification', (item,))
    calls = []
    def assess(session, state, entry):
        calls.append(entry)
        return {'accepted': True, 'reason': 'test_measured', 'visual_status': 'measured', 'verifier_queries': 4}
    monkeypatch.setattr('merit_feddg.claim_gate.assess_claim_support', assess)
    session = NativeSession(probe, image, 'Q', row['question'], config)
    runtime = CapabilityRuntime(session, pool, row, {'A': spec()}, config)
    initial = NativeState(prefix=(9,))
    state, event = runtime.execute(initial, runtime.descriptors(initial)[0])
    assert state.prefix == initial.prefix and len(state.items) == 3
    assert len(calls) == runtime.claim_checks_used == 1
    assert len(probe.tensor_packet(state.items, image)) == 1
    assert len(event['vector_gate']['entries']) == 3
    assert event['vector_gate']['verifier_queries'] == 4
    assert len(event['native_evidence'][0]['payload']['findings']) == 3
    assert not probe.controls and not probe.proposals


def test_semantic_evidence_without_geometry_is_explicitly_unverified():
    probe = SimpleNamespace(tensor_packet=lambda items, image: spatial_packet(items, (8, 8)))
    session = SimpleNamespace(probe=probe, image=Image.new('RGB', (8, 8)))
    result = assess_claim_support(session, NativeState(), finding_item(1))
    assert result['accepted'] and result['visual_status'] == 'unknown'
    assert result['reason'] == 'semantic_only_unverified' and not result['correctness_guaranteed']


def test_native_protocol_uses_answer_type_only_at_frozen_prompt_boundary(tmp_path):
    import json

    from merit_feddg.matched_evaluation import generation_prompt, load_manifest
    row = {'image': 'i', 'question': 'Q', 'image_sha256': 'h'}
    manifest = tmp_path / 'manifest.jsonl'
    rows = [{**row, 'id': 'closed', 'answer_type': 'closed'},
            {**row, 'id': 'open', 'answer_type': 'open'}]
    manifest.write_text(''.join(json.dumps(value)+'\n' for value in rows))
    loaded = load_manifest(manifest)
    config = {'prompt_contract': 'anchor-ce-v1'}
    assert generation_prompt(loaded[0], config) == 'Q Please answer Yes or No.'
    assert generation_prompt(loaded[1], config) == 'Q\nGive only the short answer. Do not explain.'


def test_packet_confidence_is_not_relabelled_as_entry_correctness():
    parent = replace(finding_item(2), confidence=.8)
    entries = native_entries((parent,))
    assert all(entry.confidence is None for entry in entries)
    assert all(entry.provenance['native_entry']['parent_confidence'] == .8 for entry in entries)
    assert parent.confidence == .8
