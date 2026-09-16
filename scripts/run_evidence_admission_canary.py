"""Replay the unchanged eight TRAIN-image probes; explicit admission execution only."""
import argparse
import gc
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch
import yaml
from run_native_contract_canary import QUESTIONS
from run_uncertainty_train import GPUS, gpu_check, load_generalist, resources, sha

from merit_feddg.capabilities import CapabilityRequest, EvidenceItem, tool_descriptors
from merit_feddg.capability_contracts import assess_authority
from merit_feddg.capability_experts import CapabilityPool
from merit_feddg.capability_runtime import NativeSession, NativeState, ValueGenerationConfig
from merit_feddg.evidence_admission import EvidenceRequest
from merit_feddg.matched_evaluation import generation_prompt
from merit_feddg.open_data import pixel_digest
from merit_feddg.open_study import atomic_json, fingerprint

OLD = Path('/home/dbw/merit-feddg-contract-validation/runs/native-contract-canary-v1')
# Authored request structures, not predictions by a new parser. No answers.
REQUESTS = (
    (('Effusion',), ('finding_presence',)),
    (('Effusion',), ('finding_presence', 'laterality')),
    (('Tuberculosis',), ('finding_presence',)),
    (('Pneumothorax', 'Tuberculosis'), ('finding_presence',)),
    (('Left Lung',), ('location', 'laterality')),
    (('Aorta',), ('location',)),
    (('Tumor', 'Heart'), ('location',)),
    (('Heart',), ('measurement',)),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    root = args.output
    protocol, rows, incumbent = resources()
    prior = json.loads((OLD/'frozen.json').read_text())
    assert [list(v) for v in QUESTIONS] == prior['questions']
    row = next(r for r in rows if r['id'] == prior['train_id'])
    assert pixel_digest(row['image']) == prior['image_sha256']
    specs = yaml.safe_load(Path('configs/llava_med_capabilities.yaml').read_text())['experts']
    specs = {k:specs[k] for k in prior['specs']}
    assert specs == prior['specs']
    assert {k:sha(s['checkpoint_path']) for k,s in specs.items()} == prior['checkpoints']
    original = {k:tuple(EvidenceItem(**v) for v in items) for k,items in
                json.loads((OLD/'native-evidence.json').read_text()).items()}
    requests = [EvidenceRequest(q, entities, frozenset(dims), 'cxr', 'open_vqa')
                for (_, q, _), (entities, dims) in zip(QUESTIONS, REQUESTS, strict=True)]
    freeze = {'base_commit':'08f5d998637036b2527b0e434f0990845a573f20',
              'old_identity':fingerprint(prior), 'old_frozen_sha256':sha(OLD/'frozen.json'),
              'old_packets_sha256':sha(OLD/'native-evidence.json'),
              'old_cases_sha256':{str(i):sha(OLD/'cases'/f'{i}.json') for i in range(8)},
              'questions':QUESTIONS, 'requests':[r.to_json() for r in requests],
              'specs':specs, 'checkpoints':prior['checkpoints'], 'image_sha256':prior['image_sha256'],
              'generation':incumbent[row['id']]['generation_config'],
              'modes':['legacy', 'audit', 'enforce'], 'paths':['normal', 'cached', 'inherited'],
              'code':{p:sha(p) for p in [__file__, 'merit_feddg/evidence_admission.py',
                       'merit_feddg/capability_runtime.py', 'merit_feddg/capability_contracts.py']},
              'no_reference_answers':True, 'no_medical_accuracy_claim':True}
    freeze = json.loads(json.dumps(freeze))
    root.mkdir(parents=True, exist_ok=True)
    if (root/'frozen.json').exists():
        assert json.loads((root/'frozen.json').read_text()) == freeze
    else:
        atomic_json(root/'frozen.json', freeze)
    if (root/'result.json').exists():
        raise RuntimeError('do not overwrite completed results')
    identity = fingerprint(freeze)
    print('PREFLIGHT', identity, flush=True)
    if args.check_only:
        return
    gpu_check(GPUS[0])
    pool = CapabilityPool(specs, 'artifacts')
    fresh, times = {}, {}
    for k, question, _ in (QUESTIONS[0], QUESTIONS[4]):
        spec = specs[k]
        req = CapabilityRequest(row['id'], row['image'], question, 'cxr', 'open_vqa',
                                'train', row['image_sha256'], spec['capabilities'][0], scope=spec['scope'])
        started = time.perf_counter()
        fresh[k] = pool.infer(k, req).items
        assert fresh[k]
        times[k] = time.perf_counter()-started
    atomic_json(root/'fresh-packets.json', {k:[asdict(i) for i in v] for k,v in fresh.items()})
    cached = {k:tuple(EvidenceItem(**i) for i in v) for k,v in
              json.loads((root/'fresh-packets.json').read_text()).items()}
    del pool
    gc.collect()
    torch.cuda.empty_cache()
    started = time.perf_counter()
    probe = load_generalist(protocol['config']['generalist'], 'artifacts')
    probe.model.eval().requires_grad_(False)
    load_seconds = time.perf_counter()-started
    cfg = ValueGenerationConfig(**freeze['generation'])
    results = []
    for index, ((k, question, _), req) in enumerate(zip(QUESTIONS, requests, strict=True)):
        query = {**row, 'question':question, 'modality':'cxr', 'task':'open_vqa'}
        allowed = k in {d['expert'] for d in tool_descriptors(specs, query)}
        automatic = assess_authority(question, specs[k], specs[k]['capabilities'][0], expert_id=k)
        prompt = generation_prompt(query, protocol['config'])
        arms = {}
        def generate(mode, items, prompt=prompt, question=question, req=req):
            started = time.perf_counter()
            snapshot = fingerprint([asdict(i) for i in items])
            session = NativeSession(probe, row['image'], prompt, question,
                                    replace(cfg, admission_mode=mode),
                                    authority_specs=specs, evidence_request=req)
            with torch.inference_mode():
                block = session.propose(NativeState(items=items), 64)
            assert snapshot == fingerprint([asdict(i) for i in items])
            # Real native masks also go through the production spatial-packet compiler.
            # This checks packet delivery, not a new spatial-guidance generation arm.
            from PIL import Image

            from merit_feddg.spatial_evidence import spatial_packet
            delivered = session.delivery_items(items)
            packet = spatial_packet(delivered, Image.open(row['image']).size, weighting='equal')
            return {'text':block.text, 'token_ids':list(block.tokens),
                    'transport':session.last_transport, 'admission':session.last_admission,
                    'delivered_items':[asdict(i) for i in delivered],
                    'spatial_records':len(packet), 'spatial_rejected':list(packet.rejected),
                    'seconds':time.perf_counter()-started}
        arms['base'] = generate('legacy', ())
        for mode in freeze['modes']:
            for path, items in [('normal', fresh[k] if allowed else ()),
                                ('cached', cached[k]), ('inherited', original[k])]:
                arms[mode+'_'+path] = generate(mode, items)
        old = json.loads((OLD/'cases'/f'{index}.json').read_text())['arms']
        parity = {a:arms[a]['token_ids'] == old[b]['token_ids'] for a,b in
                  [('base','base'), ('legacy_normal','routed'), ('legacy_inherited','inherited_packet')]}
        if not all(parity.values()):
            atomic_json(root/'parity-failure.json', {'index':index, 'parity':parity})
            raise RuntimeError('historical token parity failed; do not continue')
        for path in freeze['paths']:
            assert arms['legacy_'+path]['token_ids'] == arms['audit_'+path]['token_ids']
            actual = arms['enforce_'+path]['delivered_items']
            expected = index in (0, 4)
            assert bool(actual) == expected, 'positive coverage or forbidden delivery failure'
            assert bool(arms['enforce_'+path]['transport']['presented']) == expected
            if expected:
                field, label = ('findings','finding') if index == 0 else ('structures','anatomical_structure')
                assert [e[label] for e in actual[0]['payload'][field]] == list(req.entities)
            else:
                assert not arms['enforce_'+path]['spatial_records']
                assert arms['enforce_'+path]['token_ids'] == arms['base']['token_ids']
        for mode in freeze['modes']:
            assert arms[mode+'_cached']['token_ids'] == arms[mode+'_inherited']['token_ids']
        record = {'index':index, 'identity':identity, 'explicit_request':req.to_json(),
                  'automatic_parser':automatic, 'original_route_allowed':allowed,
                  'historical_parity':parity, 'arms':arms}
        atomic_json(root/'cases'/f'{index}.json', record)
        results.append(record)
        print('DONE', index, 'legal', index in (0, 4), 'all path checks passed', flush=True)
    summary = {'identity':identity, 'probes':8, 'train_images':1, 'complete':True,
               'medical_accuracy_evaluated':False, 'historical_token_parity':24,
               'audit_legacy_token_parity':24, 'enforce_positive_deliveries':6,
               'enforce_forbidden_deliveries':0, 'enforce_forbidden_checks':18,
               'cached_inherited_token_parity':24,
               'cost':{'expert_calls':2, 'expert_load_inference_seconds':times,
                       'actor_load_seconds':load_seconds, 'actor_calls':80,
                       'generation_and_delivery_seconds':sum(a['seconds'] for r in results for a in r['arms'].values())},
               'limitations':['Explicit requests authored; automatic parser unchanged',
                   'One real TRAIN image, no accuracy or clinical benefit claim',
                   'Undeclared/non-XRV adapters fail closed in enforce; legacy remains default',
                   'Spatial packet compiler checked; no new spatial-guidance efficacy arm']}
    atomic_json(root/'result.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
