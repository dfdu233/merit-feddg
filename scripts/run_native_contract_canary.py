"""TRAIN-image controlled contract canary; no reference answers or accuracy claim."""
import argparse
import collections
import copy
import gc
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
import yaml

from run_uncertainty_train import (
    GPUS, ValueGenerationConfig, context, gpu_check, load_generalist,
    resources, sha, visible,
)
from merit_feddg.capabilities import CapabilityRequest, tool_descriptors
from merit_feddg.capability_contracts import assess_authority
from merit_feddg.capability_experts import CapabilityPool
from merit_feddg.open_data import pixel_digest
from merit_feddg.open_study import atomic_json, fingerprint

# Frozen behavioral contrasts, not purported dataset annotations.
QUESTIONS = (
    ('cxr_findings', 'Is there a pleural effusion?', True),
    ('cxr_findings', 'Is there a left pleural effusion?', False),
    ('cxr_findings', 'Is there tuberculosis?', False),
    ('cxr_findings', 'Is there pneumothorax or tuberculosis?', False),
    ('cxr_anatomy', 'Where is the left lung located?', True),
    ('cxr_anatomy', 'Where is the aorta located?', True),
    ('cxr_anatomy', 'Where is the tumor relative to the heart?', False),
    ('cxr_anatomy', 'How large is the heart in centimeters?', False),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    root = args.output
    protocol, rows, old = resources()
    config = yaml.safe_load(Path('configs/llava_med_capabilities.yaml').read_text())
    specs = {k:config['experts'][k] for k in ('cxr_findings', 'cxr_anatomy')}
    legacy = copy.deepcopy(specs)
    for spec in legacy.values():
        spec.pop('authority_contract')
    source_root = Path('/home/dbw/merit-feddg-uncertainty/runs/uncertainty-train-v1')
    selection = json.loads((source_root/'frozen.json').read_text())['selected'][0]
    row = next(r for r in rows if r['id'] == selection['id'])
    assert pixel_digest(row['image']) == row['image_sha256']
    freeze = {'base_commit':'d7d37cf25589b332a5f41a2a29150537951f1cd5',
              'image_sha256':row['image_sha256'], 'train_id':row['id'],
              'questions':QUESTIONS, 'specs':specs,
              'code':{p:sha(p) for p in [__file__, 'merit_feddg/capability_contracts.py',
                      'merit_feddg/capabilities.py', 'merit_feddg/capability_runtime.py']},
              'checkpoints':{k:sha(s['checkpoint_path']) for k,s in specs.items()},
              'generation':old[row['id']]['generation_config'],
              'scope':'controlled TRAIN-image boundary experiment, no task accuracy or references'}
    freeze = json.loads(json.dumps(freeze))
    root.mkdir(parents=True, exist_ok=True)
    if (root/'frozen.json').exists():
        assert json.loads((root/'frozen.json').read_text()) == freeze
    else:
        atomic_json(root/'frozen.json', freeze)
    assert not (root/'result.json').exists(), 'do not overwrite results'
    identity = fingerprint(freeze)
    # Real, unmodified full TRAIN questions; only routing, no model or evaluator calls.
    counts = {k:collections.Counter() for k in specs}
    for original in rows:
        query = {**original, 'modality':old[original['id']]['input_modality'], 'task':'open_vqa'}
        before = {d['expert'] for d in tool_descriptors(legacy, query)}
        after = {d['expert'] for d in tool_descriptors(specs, query)}
        for k in specs:
            if k in before:
                counts[k]['legacy_eligible'] += 1
                counts[k]['retained' if k in after else 'blocked'] += 1
                status = assess_authority(query['question'], specs[k], specs[k]['capabilities'][0])['status']
                counts[k][status] += 1
    atomic_json(root/'routing.json', {k:dict(v) for k,v in counts.items()})
    print('PREFLIGHT', identity, {k:dict(v) for k,v in counts.items()}, flush=True)
    if args.check_only:
        return
    gpu_check(GPUS[0])
    pool = CapabilityPool(specs, 'artifacts')
    packets, expert_times = {}, {}
    for k, question, _ in (QUESTIONS[0], QUESTIONS[4]):
        spec = specs[k]
        request = CapabilityRequest(row['id'], row['image'], question, 'cxr', 'open_vqa',
                                    'train', row['image_sha256'], spec['capabilities'][0], scope=spec['scope'])
        start = time.perf_counter()
        result = pool.infer(k, request)
        assert result.items, 'real expert must return evidence'
        packets[k] = result.items
        expert_times[k] = time.perf_counter()-start
    atomic_json(root/'native-evidence.json', {k:[asdict(v) for v in items] for k,items in packets.items()})
    del pool
    gc.collect()
    torch.cuda.empty_cache()
    start = time.perf_counter()
    probe = load_generalist(protocol['config']['generalist'], 'artifacts')
    probe.model.eval().requires_grad_(False)
    actor_load = time.perf_counter()-start
    generation = ValueGenerationConfig(**freeze['generation'])
    records = []
    for index, (k, question, expected) in enumerate(QUESTIONS):
        query = {**row, 'question':question, 'modality':'cxr', 'task':'open_vqa'}
        audit = assess_authority(question, specs[k], specs[k]['capabilities'][0], expert_id=k)
        allowed = k in {d['expert'] for d in tool_descriptors(specs, query)}
        arms = {}
        for arm, items in [('base', ()), ('routed', packets[k] if allowed else ()),
                           ('inherited_packet', packets[k])]:
            start = time.perf_counter()
            session, transport = context(probe, query, protocol, generation, items)
            with torch.inference_mode():
                block = session.propose((), count=1, length=64)[0]
            arms[arm] = {'text':block.text, 'token_ids':list(block.tokens),
                         'transport':transport, 'seconds':time.perf_counter()-start}
        record = {'index':index, 'expert':k, 'question':question, 'authority':audit,
                  'expected_global_allowed':expected, 'route_allowed':allowed, 'arms':arms,
                  'blocked_route_token_parity':not allowed and arms['base']['token_ids']==arms['routed']['token_ids'],
                  'inherited_presented':bool(visible(arms['inherited_packet']['transport'])),
                  'inherited_changed_tokens':arms['base']['token_ids']!=arms['inherited_packet']['token_ids']}
        records.append(record)
        atomic_json(root/'cases'/f'{index}.json', record)
        print('DONE', index, k, audit['status'], 'allowed', allowed,
              'expected', expected, 'inherited_presented', record['inherited_presented'], flush=True)
    summary = {'identity':identity, 'n_controlled_questions':len(records), 'n_train_images':1,
               'no_accuracy_claim':True, 'routing':{k:dict(v) for k,v in counts.items()},
               'unexpected_route_allow':sum(r['route_allowed'] and not r['expected_global_allowed'] for r in records),
               'blocked_routes':sum(not r['route_allowed'] for r in records),
               'blocked_route_parity':sum(r['blocked_route_token_parity'] for r in records),
               'authority_denied_inherited_presented':sum(not r['authority']['global_transport_allowed'] and r['inherited_presented'] for r in records),
               'authority_denied_inherited_changed':sum(not r['authority']['global_transport_allowed'] and r['inherited_changed_tokens'] for r in records),
               'cost':{'expert_calls':2, 'expert_load_and_inference_seconds':expert_times,
                       'actor_load_seconds':actor_load, 'actor_calls':24,
                       'generation_seconds':sum(a['seconds'] for r in records for a in r['arms'].values())}}
    atomic_json(root/'result.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
