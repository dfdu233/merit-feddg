"""Append specialists to unchanged native all_evidence MERIT; never replace old tools."""
import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORMAL = Path('/home/dbw/merit-feddg-benchmark-fc93dca')
PRIOR = Path('/home/dbw/merit-feddg-expert-coverage/runs/pathvqa-capability-pool-v1')
NATIVE = Path('/home/dbw/merit-feddg-quilt-delivery-repair/runs/native-quilt-full-1024-context8192-v1')
sys.path.insert(0, str(ROOT))
from merit_feddg.pathology_pilot import digest, file_sha, read_json, write_new


def prepare(out):
    from collections import Counter

    from merit_feddg.expert_coverage import new_specs
    previous = read_json(PRIOR / 'protocol.json')
    specs = new_specs(previous['config'])
    # Capability-card eligibility only for ADDED tools. No coverage optimization,
    # priority ranking, or filtering of any existing expert or evidence.
    data = {'ids': previous['ids'], 'manifest': previous['manifest'],
            'manifest_sha256': previous['manifest_sha256'], 'formal_core': previous['formal_core'],
            'config': previous['config'], 'new_specs': specs, 'resources': previous['resources'],
            'scorer_at_prepare': {str(p.relative_to('/home/dbw/ANCHOR')): file_sha(p)
                                  for p in Path('/home/dbw/ANCHOR/anchor').rglob('*.py')},
            'old_scorer': previous['scorer'], 'mode': 'all_evidence',
            'source': {name: file_sha(ROOT / name) for name in (
                'scripts/run_native_expanded.py', 'merit_feddg/experts/native_coverage.py',
                'merit_feddg/expert_coverage.py', 'configs/expert_coverage_v1.yaml')},
            'baseline_protocols': {str(i): read_json(NATIVE / f'shard{i}/protocol.json') for i in (0, 1)}}
    data['identity'] = digest(data)
    out.mkdir(parents=True, exist_ok=False)
    write_new(out / 'protocol.json', dict(data, shards_complete=False))
    counts = Counter()
    for k in data['ids']:
        p = read_json(PRIOR / 'plans' / (k + '.json'))
        if file_sha(p['incumbent_path']) != p['incumbent_sha256']:
            raise ValueError('cached native MERIT changed')
        added = [n for n in specs if p['decision']['audit'][n]['reason'] == 'eligible']
        write_new(out / 'plans' / (k + '.json'), {
            'identity': data['identity'], 'row': p['row'], 'generation_config': p['generation_config'],
            'incumbent_path': p['incumbent_path'], 'incumbent_sha256': p['incumbent_sha256'], 'added': added,
            'new_eligibility': {n: p['decision']['audit'][n]['reason'] for n in specs}})
        counts.update(added)
    write_new(out / 'prepared.json', {'identity': data['identity'], 'n': len(data['ids']),
                                     'eligible_added': counts})
    print('PREPARED', counts, flush=True)


def verify(out):
    p = read_json(out / 'protocol.json')
    for name, sha in p['source'].items():
        if file_sha(ROOT / name) != sha:
            raise ValueError('implementation identity changed')
    for name, sha in p['formal_core'].items():
        if file_sha(name) != sha:
            raise ValueError('formal MERIT core changed')
    if file_sha(p['manifest']) != p['manifest_sha256']:
        raise ValueError('manifest changed')
    return p


def prefetch(out, lane, uuid):
    import torch
    from run_quilt_worker import assert_single_gpu

    from merit_feddg.capabilities import CapabilityRequest, validate_result
    from merit_feddg.capability_experts import CapabilityPool
    from merit_feddg.evidence_need import evidence_need
    p = verify(out)
    assert_single_gpu(torch, uuid)
    torch.set_num_threads(4)
    specs = json.loads(json.dumps(p['new_specs']))
    for s in specs.values():
        s['factory_kwargs']['device'] = 'cuda:0'
    pool = CapabilityPool(specs, str(ROOT / 'artifacts'))
    for k in p['ids'][lane::2]:
        plan = read_json(out / 'plans' / (k + '.json'))
        row = plan['row']
        if not plan['added']:
            continue
        if file_sha(row['image']) != row['image_sha256']:
            raise ValueError('image identity changed')
        for n in plan['added']:
            target = out / 'prefetch' / k / (n + '.json')
            if target.exists():
                if read_json(target)['identity'] != p['identity']:
                    raise ValueError('prefetch resume mismatch')
                continue
            desc = {'capability': 'classification', 'scope': specs[n]['scope']}
            query = evidence_need(row['question'], desc).query
            if plan['generation_config']['request_style'] != 'need':
                query = row['question']
            req = CapabilityRequest(k, row['image'], row['question'], row['modality'],
                                    'open_vqa', 'official-test', row['image_sha256'],
                                    'classification', scope=desc['scope'], query=query, generated_prefix='')
            result = validate_result(pool.infer(n, req), n, req)
            write_new(target, {'identity': p['identity'], 'request': asdict(req),
                               'result': asdict(result), 'cost': pool._model(specs[n]).last_cost})
            print('PREFETCH', lane, k, n, len(result.items), flush=True)
    write_new(out / f'prefetch-complete-{lane}.json', {'identity': p['identity'], 'complete': True})


def actor(out, lane, uuid, canary):
    import torch
    from run_quilt_worker import assert_single_gpu
    from transformers import set_seed
    for name in list(sys.modules):
        if name == 'merit_feddg' or name.startswith('merit_feddg.'):
            del sys.modules[name]
    sys.path.insert(0, str(FORMAL))
    from merit_feddg.capabilities import CapabilityResult, EvidenceItem
    from merit_feddg.capability_runtime import (
        CapabilityRuntime,
        NativeSession,
        ValueGenerationConfig,
    )
    from merit_feddg.generalist_factory import load_generalist
    p = verify(out)
    assert_single_gpu(torch, uuid)
    torch.set_num_threads(4)
    old = read_json('/home/dbw/merit-feddg-pathology-quilt/runs/pathology-quilt-formal-budgeted-v2/frozen.json')
    original_config = read_json(old['protocol'])['config']
    generalist = original_config['generalist']
    probe = load_generalist(generalist, str(ROOT / 'artifacts'))
    probe.model.eval().requires_grad_(False)
    probe.benchmark_single_block_context = True
    probe.benchmark_evidence_reserve_tokens = 64
    original_limit = probe.model.config.tokenizer_model_max_length
    keys = p['ids'][lane::2]
    if canary:
        keys = [k for k in keys if read_json(out / 'plans' / (k + '.json'))['added']][:1]
        if not keys:
            raise ValueError('no eligible added-expert canary')
    for i, k in enumerate(keys, 1):
        target = out / ('canary' if canary else 'cases') / (k + '.json')
        if target.exists():
            if read_json(target)['identity'] != p['identity']:
                raise ValueError('resume mismatch')
            continue
        plan = read_json(out / 'plans' / (k + '.json'))
        if file_sha(plan['incumbent_path']) != plan['incumbent_sha256']:
            raise ValueError('native baseline changed')
        cached = read_json(plan['incumbent_path'])
        baseline = cached['outputs']['merit_quilt']
        if not plan['added']:
            write_new(target, {'id': k, 'identity': p['identity'], 'output': baseline,
                               'status': 'exact_native_incumbent_reuse', 'added': []})
            continue
        source = plan['row']
        if file_sha(source['image']) != source['image_sha256']:
            raise ValueError('image identity mismatch')
        row = {name: source[name] for name in ('id', 'image', 'question', 'modality')}
        row.update(task='open_vqa', domain='official-test', group_id=source['image_sha256'],
                   image_sha256=source['image_sha256'], capability='classification',
                   role='target', domain_kind='official_dataset_split')
        cfg = ValueGenerationConfig(**plan['generation_config'])
        saved_specs = p['baseline_protocols'][str(lane)]['specs']
        # JSON manifests sort keys; reconstruct the original registration order,
        # not the alphabetical order of the serialized augmented registry.
        base_specs = {n: saved_specs[n] for n in original_config['experts'] if n in saved_specs}
        base_specs['quilt_pathology'] = saved_specs['quilt_pathology']
        if cached.get('reused_outside_quilt_scope'):
            base_specs.pop('quilt_pathology', None)
        specs = dict(base_specs)
        specs.update({n: p['new_specs'][n] for n in plan['added']})
        old_tools = [t for t in baseline['trace'] if t.get('event') == 'tool']

        class Replay:
            last_origin = 'exact_native_cache'

            def __init__(self, added, case_id, traces):
                self.added, self.case_id, self.traces = added, case_id, traces

            def infer(self, name, req):
                if name in self.added:
                    value = read_json(out / 'prefetch' / self.case_id / (name + '.json'))
                    if value['identity'] != p['identity'] or value['request'] != asdict(req):
                        raise RuntimeError('new native request mismatch')
                    raw = value['result']['items']
                else:
                    matches = [t for t in self.traces if t['expert'] == name and t['request'] == asdict(req)]
                    if len(matches) != 1 or not matches[0]['executed']:
                        raise RuntimeError('old expert request/execution mismatch')
                    raw = matches[0]['native_evidence']
                return CapabilityResult(name, req.capability, tuple(EvidenceItem(**v) for v in raw))

        probe.model.config.tokenizer_model_max_length = (
            original_limit if cached.get('reused_outside_quilt_scope') else 8192)

        replay = Replay(plan['added'], k, old_tools)

        def run(current_specs, row=row, source=source, cfg=cfg, replay=replay):
            set_seed(42)
            session = NativeSession(probe, row['image'], source['benchmark_prompt'], row['question'], cfg)
            with torch.inference_mode():
                return CapabilityRuntime(session, replay, row, current_specs, cfg, None).run('all_evidence')

        parity = None
        if canary:
            unchanged = run(base_specs)
            parity = unchanged['token_ids'] == baseline['token_ids'] and unchanged['text'] == baseline['text']
            if not parity:
                raise RuntimeError('native baseline parity failure')
        output = run(specs)
        tools = [t for t in output['trace'] if t.get('event') == 'tool']
        retained = [t for t in tools if t['expert'] not in plan['added']]
        if [(t['expert'], t['request'], t['executed'], t['adopted']) for t in retained] != [
                (t['expert'], t['request'], t['executed'], t['adopted']) for t in old_tools]:
            raise RuntimeError('old tool sequence or adoption changed')
        old_items = {(v['expert_id'], v['evidence_id']): v for v in baseline['evidence']}
        new_items = {(v['expert_id'], v['evidence_id']): v for v in output['evidence']}
        if any(new_items.get(key) != val for key, val in old_items.items()):
            raise RuntimeError('old evidence changed or removed')
        old_transport = [t['evidence_transport'] for t in baseline['trace'] if t.get('event') == 'decode'][-1]
        transport = [t['evidence_transport'] for t in output['trace'] if t.get('event') == 'decode'][-1]
        visible = {(v['expert_id'], v['evidence_id']) for v in transport['presented']}
        if not {(v['expert_id'], v['evidence_id']) for v in old_transport['presented']} <= visible:
            raise RuntimeError('new evidence evicted old delivery')
        if not output['text'].strip() or len(output['token_ids']) >= cfg.max_new_tokens:
            raise RuntimeError('empty/capped final output')
        fresh = [t for t in tools if t['expert'] in plan['added']]
        delivered = [n for n in plan['added'] if any(expert == n for expert, _ in visible)]
        write_new(target, {'id': k, 'identity': p['identity'], 'output': output,
                           'status': 'expanded_native_merit', 'added': plan['added'],
                           'new_calls': len(fresh), 'new_delivered': delivered,
                           'baseline_token_parity': parity, 'old_evidence_preserved': True})
        if canary and not delivered:
            raise RuntimeError('canary has no delivered new evidence; stop before full run')
        print('NATIVE_EXPANDED', lane, i, len(keys), k, 'added', delivered, flush=True)
    write_new(out / f'{"canary" if canary else "actor"}-complete-{lane}.json',
              {'identity': p['identity'], 'complete': True, 'n': len(keys)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=['prepare', 'prefetch', 'actor'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--shard-index', type=int, choices=[0, 1], default=0)
    parser.add_argument('--gpu-uuid')
    parser.add_argument('--canary', action='store_true')
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare(args.output)
    elif args.stage == 'prefetch':
        prefetch(args.output, args.shard_index, args.gpu_uuid)
    else:
        actor(args.output, args.shard_index, args.gpu_uuid, args.canary)
