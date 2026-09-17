"""User-authorized formal PathVQA adapter; preserves the upstream TRAIN pilot.

Uses the frozen formal benchmark core in a separate actor process, never the
Quilt llava package. Original 6719 IDs, main prompts/1024 budget and cached scope.
"""
import argparse
import gzip
import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = Path('/home/dbw/merit-feddg-benchmark-fc93dca')
BASE = Path('/home/dbw/ANCHOR/corrected_runs/paper_baselines_v1/merit_common_protocol_v1')
DONOR = BASE/'pathvqa/a095b918c9a484530e90b3d1c47f7590ed5295c1ac01ac40f8c0eaba8744ac70'
GPU = 'GPU-3846413a-4238-d307-b1f3-10c2dfbe002c'
sys.path.insert(0, str(CORE))
# New dependency-light helpers, but actor classes come from the formal core.
import merit_feddg
spec = importlib.util.spec_from_file_location('merit_feddg.pathology_pilot', ROOT/'merit_feddg/pathology_pilot.py')
helpers = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = helpers
spec.loader.exec_module(helpers)
from merit_feddg.pathology_pilot import digest, file_sha, read_json, write_new, prediction_jobs, quilt_item, ARMS
from merit_feddg.open_study import fingerprint


def cached(key):
    path = DONOR/'case-cache/compact_rows'/(fingerprint(key)+'.json')
    if not path.exists():
        path = Path(str(path)+'.gz')
    with gzip.open(path, 'rt') if path.suffix == '.gz' else path.open() as f:
        value = json.load(f)
    if value['identity'] != DONOR.name:
        raise ValueError('old native cache identity mismatch')
    return value['output'], path


def core_identity():
    return {str(p): file_sha(p) for p in sorted((CORE/'merit_feddg').rglob('*.py'))}


def prepare(out):
    from run_pathology_quilt_pilot import code_identity, scorer_identity
    if out.exists():
        raise FileExistsError('fresh formal output required')
    protocol = read_json(DONOR/'protocol.json')
    rows = [json.loads(l) for l in (BASE/'pathvqa.jsonl').read_text().splitlines()]
    assert protocol['shards_complete'] and protocol['n'] == len(rows) == 6719
    assert protocol['config']['prompt_contract'] == 'anchor-final-v1'
    routes = read_json(DONOR/'routing.json')
    assert set(routes) == {r['id'] for r in rows}
    selected, cache_hashes, images, counts = [], {}, {}, {}
    for row in rows:
        assert not helpers.LABEL_KEYS.intersection(row)
        before, path = cached(row['id'])
        assert before['generation_config'] == protocol['arm_configs']['compact_rows']
        cache_hashes[str(path)] = file_sha(path)
        if row['image'] not in images:
            images[row['image']] = file_sha(row['image'])
        assert images[row['image']] == row['image_sha256']
        modality = before['input_modality']
        assert modality == routes[row['id']]['modality']
        counts[modality] = counts.get(modality, 0)+1
        if modality == 'pathology':
            selected.append({k: row[k] for k in ('id', 'image', 'image_sha256', 'question')} | {
                'group_id': row['image_sha256'], 'image_file_sha256': row['image_sha256'],
                'modality': 'pathology', 'task': 'open_vqa', 'prompt': row['benchmark_prompt']})
    selected.sort(key=lambda r: digest([r['id'], r['image_sha256']]))
    if len({r['image_sha256'] for r in selected}) < 2:
        raise ValueError('not enough eligible different images')
    donors = {}
    for i, row in enumerate(selected):
        for j in range(1, len(selected)):
            other = selected[(i+j) % len(selected)]
            if other['image_sha256'] != row['image_sha256']:
                donors[row['id']] = other['id']
                break
    canary = [selected[0]['id'], donors[selected[0]['id']]]
    frozen = {'schema': 'pathology-quilt-formal-v1', 'split': 'test',
        'manifest': str(BASE/'pathvqa.jsonl'), 'manifest_sha256': file_sha(BASE/'pathvqa.jsonl'),
        'protocol': str(DONOR/'protocol.json'), 'protocol_sha256': file_sha(DONOR/'protocol.json'),
        'rows': selected, 'full_ids': [r['id'] for r in rows], 'coverage': counts,
        'donors': donors, 'donor_policy': 'next hash-ordered distinct image; same question',
        'canary_ids': canary, 'max_new_tokens': 1024, 'quilt_precision': 'fp16',
        'arms': ARMS, 'seed': 42, 'source_code': code_identity(), 'formal_core': core_identity(),
        'formal_adapter_sha256': file_sha(__file__), 'native_cache_hashes': cache_hashes,
        'scorer': scorer_identity('/home/dbw/ANCHOR'), 'no_test_parameter_selection': True,
        'scope': 'all 6719 IDs; Quilt only for existing microscopy routes; otherwise explicit incumbent reuse'}
    write_new(out/'frozen.json', frozen)
    print('PREPARED', digest(frozen), 'eligible', len(selected), 'full', len(rows), flush=True)


def verify(frozen):
    from run_pathology_quilt_pilot import code_identity
    assert frozen['formal_adapter_sha256'] == file_sha(__file__)
    assert frozen['source_code'] == code_identity()
    assert frozen['formal_core'] == core_identity()
    for name in ('manifest', 'protocol'):
        assert file_sha(frozen[name]) == frozen[name+'_sha256']


def actor(out, canary):
    import torch
    from transformers import set_seed
    from run_quilt_worker import assert_single_gpu
    from merit_feddg.capabilities import EvidenceItem
    from merit_feddg.capability_runtime import NativeSession, NativeState, ValueGenerationConfig
    from merit_feddg.generalist_factory import load_generalist
    frozen = read_json(out/'frozen.json')
    verify(frozen)
    assert_single_gpu(torch, GPU)
    prior = read_json(out/('quilt_canary.json' if canary else 'quilt_predictions.json'))
    assert prior['identity'] == digest(frozen) and prior['complete']
    predictions = prior['predictions']
    jobs = {(j['target_id'], j['variant']): j for j in prediction_jobs(frozen)}
    protocol = read_json(frozen['protocol'])
    torch.set_num_threads(4)
    started = time.perf_counter()
    probe = load_generalist(protocol['config']['generalist'], str(ROOT/'artifacts'))
    probe.model.eval().requires_grad_(False)
    probe.benchmark_single_block_context = True
    probe.benchmark_evidence_reserve_tokens = 64
    write_new(out/('actor_load_'+str(time.time_ns())+'.json'), {'seconds': time.perf_counter()-started, 'gpu_uuid': GPU})
    selected = {r['id']: r for r in frozen['rows']}
    allrows = {r['id']: r for r in map(json.loads, Path(frozen['manifest']).read_text().splitlines())}
    def visible(t):
        return {(x['expert_id'], x['evidence_id']) for x in t.get('presented', [])}
    def answer(images, prompt, limit):
        start = time.perf_counter()
        set_seed(42)
        with torch.inference_mode():
            block = probe.new_answer_session(images, prompt).propose((), count=1, length=limit)[0]
        if not block.text.strip() or len(block.tokens) >= limit:
            raise RuntimeError('empty/capped formal actor output; no fallback')
        return {'text': block.text, 'token_ids': list(block.tokens), 'seconds': time.perf_counter()-start,
                'prompt_sha256': fingerprint(prompt)}
    for key in frozen['canary_ids'] if canary else frozen['full_ids']:
        target = out/'cases'/(key+'.json')
        if target.exists():
            assert read_json(target)['identity'] == digest(frozen)
            continue
        start = time.perf_counter()
        before, path = cached(key)
        assert file_sha(path) == frozen['native_cache_hashes'][str(path)]
        incumbent = {k: before[k] for k in ('text', 'token_ids', 'seconds')}
        if key not in selected:
            write_new(target, {'identity': digest(frozen), 'id': key, 'status': 'unavailable_modality',
                'arms': {a: incumbent for a in ARMS}, 'new_actor_calls': 0, 'wall_seconds': 0})
            continue
        row = allrows[key]
        assert file_sha(row['image']) == row['image_sha256']
        cfg = ValueGenerationConfig(**before['generation_config'])
        assert cfg.max_new_tokens == cfg.block_tokens == 1024
        def context(items):
            s = NativeSession(probe, row['image'], row['benchmark_prompt'], row['question'], cfg)
            im, prompt = s.context(NativeState(items=tuple(items)))
            return im, prompt, s.last_transport
        raw = tuple(EvidenceItem(**e) for e in before['evidence'])
        im, prompt, transport = context(raw)
        historical = [x['evidence_transport'] for x in before['trace'] if 'evidence_transport' in x][-1]
        assert transport['prompt_sha256'] == historical['prompt_sha256'], 'historical prompt mismatch'
        assert transport['evidence_sha256'] == historical['evidence_sha256'], 'historical evidence mismatch'
        allowed = visible(transport)
        kept = tuple(e for e in raw if (e.expert_id, e.evidence_id) in allowed)
        _, retained_prompt, retained_transport = context(kept)
        assert retained_prompt == prompt and visible(retained_transport) == allowed
        checks, calls = {}, 0
        if key in frozen['canary_ids']:
            reproduced = answer(im, prompt, 1024)
            checks['compact_token_parity'] = reproduced['token_ids'] == incumbent['token_ids']
            assert checks['compact_token_parity'], 'incumbent token mismatch'
            calls += 1
        arms, transports = {'compact': incumbent}, {'compact': transport}
        no_conch = tuple(e for e in kept if e.expert_id != 'conch_tissue')
        im, prompt, t = context(no_conch)
        assert visible(t) == {(e.expert_id,e.evidence_id) for e in no_conch}
        arms['without_conch'] = answer(im, prompt, 1024) if no_conch != kept else incumbent
        calls += no_conch != kept
        transports['without_conch'] = t
        for variant, arm in (('matched','compact_quilt'),('wrong_image','compact_wrong_image')):
            job = jobs[key, variant]
            record = predictions[job['key']]
            assert record['job'] == job
            item = quilt_item(record)
            im, prompt, t = context(kept+(item,))
            assert visible(t) == allowed | {(item.expert_id,item.evidence_id)}, 'evidence omission/displacement'
            assert record['text'] in prompt or json.dumps(record['text'],ensure_ascii=False)[1:-1] in prompt, 'expert text truncated'
            arms[arm] = answer(im, prompt, 1024)
            calls += 1
            transports[arm] = t
            if variant == 'matched':
                arms['quilt_alone'] = {k: record[k] for k in ('text','token_ids','seconds')}
        write_new(target, {'identity': digest(frozen), 'id': key, 'status': 'real_candidate',
            'arms': arms, 'transport': transports, 'checks': checks, 'new_actor_calls': calls,
            'wall_seconds': time.perf_counter()-start, 'peak_allocated_bytes': torch.cuda.max_memory_allocated()})
        print('ACTOR', key, calls, flush=True)
    marker = 'actor_canary.json' if canary else 'complete.json'
    if not canary:
        assert {p.stem for p in (out/'cases').glob('*.json')} == set(frozen['full_ids'])
    write_new(out/marker, {'identity': digest(frozen), 'complete': True, 'full_dataset_complete': not canary})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=('prepare','actor'), required=True)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--canary', action='store_true')
    args = p.parse_args()
    prepare(args.output) if args.stage == 'prepare' else actor(args.output,args.canary)


if __name__ == '__main__':
    main()
