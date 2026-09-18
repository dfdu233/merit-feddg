"""Single frozen expanded-pool arm on the original full PathVQA manifest.

prepare/prefetch run in the existing research env; actor uses the original formal
LLaVA-Med env/core. Cache reuse requires matching requests and image identities.
No labels, reference answers, learned thresholds or generated-code execution.
"""
import argparse
import gzip
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from merit_feddg.pathology_pilot import digest, file_sha, read_json, write_new

FORMAL = Path('/home/dbw/merit-feddg-benchmark-fc93dca')
OLD = Path('/home/dbw/merit-feddg-pathology-quilt/runs/pathology-quilt-formal-budgeted-v2')
NATIVE = Path('/home/dbw/merit-feddg-quilt-delivery-repair/runs/native-quilt-full-1024-context8192-v1')


def sources():
    files = ['configs/expert_coverage_v1.yaml', 'merit_feddg/expert_coverage.py',
             'merit_feddg/experts/native_coverage.py', 'scripts/run_coverage_pathvqa.py']
    return {name: file_sha(ROOT / name) for name in files}


def original(key, frozen):
    from merit_feddg.open_study import fingerprint
    root = Path(frozen['protocol']).parent
    path = root / 'case-cache/compact_rows' / (fingerprint(key) + '.json')
    if not path.exists():
        path = Path(str(path) + '.gz')
    if file_sha(path) != frozen['native_cache_hashes'][str(path)]:
        raise ValueError('incumbent native cache changed')
    with gzip.open(path, 'rt') if path.suffix == '.gz' else path.open() as stream:
        cache = json.load(stream)
    if cache['identity'] != root.name:
        raise ValueError('baseline identity mismatch')
    return cache['output'], path


def prepare(out, config_path):
    from collections import Counter

    from merit_feddg.expert_coverage import load_config, new_specs, select_experts
    config = load_config(config_path)
    frozen = read_json(OLD / 'frozen.json')
    rows = [json.loads(line) for line in Path(frozen['manifest']).read_text().splitlines()]
    if len(rows) != 6719 or [r['id'] for r in rows] != frozen['full_ids']:
        raise ValueError('full original manifest required')
    native_protocols = {}
    for shard in range(2):
        source = NATIVE / f'shard{shard}'
        summary, protocol = read_json(source / 'summary.json'), read_json(source / 'protocol.json')
        if not summary['complete'] or summary['identity'] != protocol['identity']:
            raise ValueError('native incumbent incomplete')
        if set(protocol['cases']) != set(frozen['full_ids'][shard::2]):
            raise ValueError('native incumbent IDs mismatch')
        native_protocols[str(shard)] = file_sha(source / 'protocol.json')
    specs = new_specs(config)
    for kind in {c['kind'] for c in config['new'].values()}:
        from merit_feddg.experts.native_coverage import verify_resource
        verify_resource(Path(config['model_root']) / kind, kind)
    resources = {d.name: read_json(d / 'resource.json')
                 for d in Path(config['model_root']).iterdir() if (d / 'resource.json').exists()}
    identity_data = {'config': config, 'resources': resources,
                     'source_sha256': sources(), 'manifest': frozen['manifest'],
                     'manifest_sha256': file_sha(frozen['manifest']), 'original_identity': digest(frozen),
                     'native_protocol_sha256': native_protocols, 'ids': frozen['full_ids'],
                     'formal_core': frozen['formal_core'], 'scorer': frozen['scorer']}
    identity = digest(identity_data)
    counts, new_counts = Counter(), Counter()
    out.mkdir(parents=True, exist_ok=False)
    write_new(out / 'protocol.json', {'identity': identity, **identity_data,
                                    'shards_complete': False, 'n': len(rows),
                                    'fallback': 'reuse native MERIT+Quilt incumbent when no compatible capability'})
    for index, row in enumerate(rows):
        if any(k in row for k in ('answer', 'answers', 'label', 'labels', 'reference', 'references')):
            raise ValueError('generation manifest contains labels')
        before, old_path = original(row['id'], frozen)
        native_case = NATIVE / f'shard{index % 2}' / 'cases' / (row['id'] + '.json')
        current = read_json(native_case)
        available, origins = set(specs), {}
        for event in before['trace']:
            if event.get('event') == 'tool' and event.get('executed') and event.get('native_evidence'):
                request = event['request']
                if any(request[k] != row[v] for k, v in
                       [('sample_id', 'id'), ('image', 'image'), ('question', 'question')]):
                    raise ValueError('inherited expert request mismatch')
                available.add(event['expert'])
                origins[event['expert']] = {'items': event['native_evidence'], 'request': request,
                                           'source': str(old_path), 'source_sha256': file_sha(old_path)}
        quilt = NATIVE / f'shard{index % 2}' / 'prefetch' / (row['id'] + '.json')
        if quilt.exists():
            q = read_json(quilt)
            if any(q['request'][k] != row[v] for k, v in
                   [('sample_id', 'id'), ('image', 'image'), ('question', 'question')]):
                raise ValueError('Quilt native request mismatch')
            if q['result']['items']:
                available.add('quilt_pathology')
                origins['quilt_pathology'] = {'items': q['result']['items'], 'request': q['request'],
                                            'source': str(quilt), 'source_sha256': file_sha(quilt)}
        route_row = {**row, 'modality': before['input_modality'], 'input_kind': '2d'}
        decision = select_experts(config, route_row, available)
        selected = decision['selected']
        fresh = [n for n in selected if n in specs]
        inherited = {n: origins[n] for n in selected if n in origins}
        incumbent = current['outputs']['merit_quilt']
        write_new(out / 'plans' / (row['id'] + '.json'), {
            'identity': identity, 'row': route_row, 'decision': decision, 'new_experts': fresh,
            'inherited': inherited, 'incumbent_path': str(native_case),
            'incumbent_sha256': file_sha(native_case), 'generation_config': before['generation_config'],
            'incumbent_nonempty': bool(incumbent['text'].strip())})
        counts.update(selected)
        new_counts.update(fresh)
    write_new(out / 'selection-summary.json', {'selected': counts, 'new_selected': new_counts,
                                              'n': len(rows), 'identity': identity})
    print('PREPARED', identity, counts, 'NEW', new_counts, flush=True)


def verify(out):
    p = read_json(out / 'protocol.json')
    if p['source_sha256'] != sources() or p['manifest_sha256'] != file_sha(p['manifest']):
        raise ValueError('frozen implementation/config/manifest mismatch')
    for filename, expected in p['formal_core'].items():
        if file_sha(filename) != expected:
            raise ValueError('formal core changed')
    return p


def prefetch(out, lane, uuid):
    import gc

    import torch
    from run_quilt_worker import assert_single_gpu

    from merit_feddg.capabilities import CapabilityRequest, validate_result
    from merit_feddg.capability_experts import CapabilityPool
    from merit_feddg.expert_coverage import new_specs
    p = verify(out)
    assert_single_gpu(torch, uuid)
    torch.set_num_threads(4)
    specs = new_specs(p['config'], 'cuda:0')
    # Group by checkpoint to avoid repeated loads; process remains backgrounded.
    for kind in ('unimed', 'flair', 'monet'):
        subset = {n: s for n, s in specs.items() if s['factory_kwargs']['kind'] == kind}
        pool = CapabilityPool(subset, str(ROOT / 'artifacts'))
        for key in p['ids'][lane::2]:
            plan = read_json(out / 'plans' / (key + '.json'))
            for name in plan['new_experts']:
                if name not in subset:
                    continue
                target = out / 'new-evidence' / key / (name + '.json')
                if target.exists():
                    if read_json(target)['identity'] != p['identity']:
                        raise ValueError('prefetch resume mismatch')
                    continue
                row = plan['row']
                if file_sha(row['image']) != row['image_sha256']:
                    raise ValueError('image identity mismatch')
                request = CapabilityRequest(key, row['image'], row['question'], row['modality'],
                                            'open_vqa', 'official-test', row['image_sha256'],
                                            'classification', scope=specs[name]['scope'])
                result = validate_result(pool.infer(name, request), name, request)
                model = pool._model(specs[name])
                write_new(target, {'identity': p['identity'], 'request': asdict(request),
                                   'result': asdict(result), 'cost': model.last_cost, 'gpu_uuid': uuid})
                print('NEW_EXPERT', lane, key, name, len(result.items), flush=True)
        del pool
        if 'model' in locals():
            del model
        gc.collect()
        torch.cuda.empty_cache()
    write_new(out / f'prefetch-complete-{lane}.json', {'identity': p['identity'], 'complete': True})


def actor(out, lane, uuid, canary):
    import time

    import torch
    from run_quilt_worker import assert_single_gpu
    from transformers import set_seed
    # No research runtime imports have occurred in this process before here.
    for name in list(sys.modules):
        if name == 'merit_feddg' or name.startswith('merit_feddg.'):
            del sys.modules[name]
    sys.path.insert(0, str(FORMAL))
    from merit_feddg.capabilities import EvidenceItem
    from merit_feddg.capability_runtime import NativeSession, NativeState, ValueGenerationConfig
    from merit_feddg.generalist_factory import load_generalist
    p = verify(out)
    assert_single_gpu(torch, uuid)
    torch.set_num_threads(4)
    frozen = read_json(OLD / 'frozen.json')
    base_protocol = read_json(frozen['protocol'])
    probe = load_generalist(base_protocol['config']['generalist'], str(ROOT / 'artifacts'))
    probe.model.eval().requires_grad_(False)
    probe.benchmark_single_block_context = True
    probe.benchmark_evidence_reserve_tokens = 64
    original_limit = probe.model.config.tokenizer_model_max_length
    keys = p['ids'][lane::2]
    if canary:
        # Fixed first scheduled case with a new expert; not selected by any score.
        keys = [k for k in keys if read_json(out / 'plans' / (k + '.json'))['new_experts']][:1]
        if not keys:
            raise ValueError('no real new-expert canary in this lane')
    for i, key in enumerate(keys, 1):
        target = out / ('canary' if canary else 'cases') / (key + '.json')
        if target.exists():
            if read_json(target)['identity'] != p['identity']:
                raise ValueError('case resume mismatch')
            continue
        plan = read_json(out / 'plans' / (key + '.json'))
        row = plan['row']
        if file_sha(row['image']) != row['image_sha256']:
            raise ValueError('image bytes changed')
        if file_sha(plan['incumbent_path']) != plan['incumbent_sha256']:
            raise ValueError('native incumbent changed')
        incumbent = read_json(plan['incumbent_path'])['outputs']['merit_quilt']
        cfg = ValueGenerationConfig(**plan['generation_config'])
        def context(items, row=row, cfg=cfg):
            session = NativeSession(probe, row['image'], row['benchmark_prompt'], row['question'], cfg)
            images, prompt = session.context(NativeState(items=tuple(items)))
            return images, prompt, session.last_transport
        def generate(images, prompt):
            set_seed(42)
            start = time.perf_counter()
            with torch.inference_mode():
                block = probe.new_answer_session(images, prompt).propose((), count=1, length=1024)[0]
            if not block.text.strip() or len(block.tokens) >= 1024:
                raise RuntimeError('empty/capped output: engineering failure, no fallback')
            return {'text': block.text.strip(), 'token_ids': list(block.tokens),
                    'seconds': time.perf_counter() - start}
        parity = None
        if canary:
            before, _ = original(key, frozen)
            probe.model.config.tokenizer_model_max_length = original_limit
            images, prompt, _ = context([EvidenceItem(**x) for x in before['evidence']])
            parity = generate(images, prompt)['token_ids'] == before['token_ids']
            if not parity:
                raise RuntimeError('formal baseline token parity failure')
        items, costs = [], []
        for name in plan['decision']['selected']:
            if name in plan['inherited']:
                data = plan['inherited'][name]
                if file_sha(data['source']) != data['source_sha256']:
                    raise ValueError('inherited evidence changed')
                items.extend(EvidenceItem(**x) for x in data['items'])
            else:
                data = read_json(out / 'new-evidence' / key / (name + '.json'))
                if data['identity'] != p['identity']:
                    raise ValueError('new evidence identity mismatch')
                items.extend(EvidenceItem(**x) for x in data['result']['items'])
                costs.append(data['cost'])
        probe.model.config.tokenizer_model_max_length = p['config']['input_limit']
        if not items:
            result, transport, status = incumbent, {}, 'no_compatible_evidence_incumbent_reuse'
        else:
            images, prompt, transport = context(items)
            visible = {(v['expert_id'], v['evidence_id']) for v in transport['presented']}
            expected = {(x.expert_id, x.evidence_id) for x in items}
            if visible != expected:
                raise RuntimeError('selected evidence omitted: stop, do not call it successful routing')
            result, status = generate(images, prompt), 'real_candidate'
        write_new(target, {'id': key, 'identity': p['identity'], 'output': result,
                           'status': status, 'decision': plan['decision'], 'transport': transport,
                           'expert_costs': costs, 'baseline_token_parity': parity,
                           'input_modality': row['modality'], 'gpu_uuid': uuid,
                           'incumbent_seconds': incumbent.get('seconds', 0),
                           'generation_config': asdict(cfg), 'input_limit': p['config']['input_limit']})
        print('COVERAGE_ACTOR', lane, i, len(keys), key, status, flush=True)
    write_new(out / (f'canary-complete-{lane}.json' if canary else f'actor-complete-{lane}.json'),
              {'identity': p['identity'], 'n': len(keys), 'complete': True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['prepare', 'prefetch', 'actor'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', default=str(ROOT / 'configs/expert_coverage_v1.yaml'))
    parser.add_argument('--shard-index', type=int, choices=[0, 1], default=0)
    parser.add_argument('--gpu-uuid')
    parser.add_argument('--canary', action='store_true')
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare(args.output, args.config)
    elif args.stage == 'prefetch':
        prefetch(args.output, args.shard_index, args.gpu_uuid)
    else:
        actor(args.output, args.shard_index, args.gpu_uuid, args.canary)


if __name__ == '__main__':
    main()
