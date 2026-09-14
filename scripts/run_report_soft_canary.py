"""Two scheduling-stop MIMIC reports, unchanged spatial soft operator, no labels."""
import argparse
import json
import os
import time
from pathlib import Path

import torch
from run_soft_guidance_full import full_case, sha

from merit_feddg.generalist_factory import generalist_provenance, load_generalist
from merit_feddg.open_study import atomic_json, fingerprint

BASE = Path('/home/dbw/merit-feddg/runs/mimic-partial-matched-verified-packets-anchor/'
            '4a764a53e98ef755bb1a16e32e650033a8e55b06db27aae3f795e300273d5d0c')
MANIFEST = Path('/home/dbw/merit-feddg/runs/mimic-partial-official/manifest.jsonl')


def inputs():
    rows = [json.loads(s) for s in MANIFEST.read_text().splitlines()]
    ids = {r['id'] for r in rows}
    if len(rows) != 694 or len(ids) != 694 or any(r['answer_type'] != 'report' for r in rows):
        raise RuntimeError('existing full 694-report manifest required')
    if any(set(r) & {'answer', 'answers', 'reference', 'label', 'report', 'gt_ans'} for r in rows):
        raise RuntimeError('generation manifest contains reference fields')
    protocols = [json.loads(p.read_text()) for p in sorted(BASE.glob('shards/*/protocol.json'))]
    if len(protocols) != 2 or {p['shard_index'] for p in protocols} != {0, 1}:
        raise RuntimeError('both historical shard protocols required')
    if any(p['identity'] != BASE.name or p['n'] != 694 or p['shard_count'] != 2
           or p['config'] != protocols[0]['config'] for p in protocols):
        raise RuntimeError('historical shard configuration mismatch')
    files = {p.stem for p in (BASE / 'case-cache/compact_rows').glob('*.json')}
    if files != {fingerprint(k) for k in ids}:
        raise RuntimeError('compact cache IDs do not cover the exact report manifest')
    return rows, protocols[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    rows, protocol = inputs()
    spec = protocol['config']['generalist']
    frozen = {'schema': 'mimic-report-spatial-soft-canary-v1', 'alpha': .5,
        'base': str(BASE), 'manifest': str(MANIFEST), 'manifest_sha256': sha(MANIFEST),
        'full_manifest_n': len(rows), 'scheduling_stop': 2,
        'source_root_completion_missing': not (BASE / 'protocol.json').exists(),
        'model': generalist_provenance(spec, 'artifacts'),
        'code': {str(p): sha(p) for p in sorted(Path('merit_feddg').rglob('*.py'))},
        'runner_sha256': sha(__file__),
        'operator_sha256': sha(Path(__file__).with_name('run_soft_guidance_full.py')),
        'runtime': {'torch': torch.__version__, 'cuda': torch.version.cuda, 'threads': 4}}
    frozen = json.loads(json.dumps(frozen))
    root = args.output / fingerprint(frozen)
    os.umask(0o002)
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'frozen.json').exists():
        if json.loads((root / 'frozen.json').read_text()) != frozen:
            raise RuntimeError('conflicting frozen configuration')
    else:
        atomic_json(root / 'frozen.json', frozen)
    print('ROOT', root.resolve(), 'CHECK PASSED; NOT FULL BASELINE CERTIFICATION', flush=True)
    if args.check_only:
        return
    import fcntl
    with (root / '.worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(4)
        torch.ones(1, device='cuda:0').sum().item()
        start = time.perf_counter()
        probe = load_generalist(spec, 'artifacts')
        probe.model.eval().requires_grad_(False)
        atomic_json(root / f'load-{time.time_ns()}.json', {'seconds': time.perf_counter() - start,
            'device': str(torch.cuda.get_device_properties(0)),
            'timing_caveat': 'concurrent full VQA/SLAKE worker; not isolated latency'})
        generalists = json.loads((BASE / 'generalist.json').read_text())
        if set(generalists) != {r['id'] for r in rows}:
            raise RuntimeError('historical generalist ID mismatch')
        for row in rows[:2]:
            target = root / (fingerprint(row['id']) + '.json')
            if target.exists():
                raise RuntimeError('canary output exists; inspect before resuming')
            entry = json.loads((BASE / 'case-cache/compact_rows' / (fingerprint(row['id']) + '.json')).read_text())
            if entry['identity'] != BASE.name:
                raise RuntimeError('source cache identity mismatch')
            result = full_case(probe, row, entry['output'], generalists[row['id']], protocol, .5)
            atomic_json(target, {'identity': root.name, **result})
            print('DONE REPORT', row['id'], 'applicable', result['applicable'],
                  'seconds', result['wall_seconds'], flush=True)
        atomic_json(root / 'canary-complete.json', {'n': 2, 'full_manifest_n': 694,
            'full_dataset_complete': False, 'identity': root.name})
        print('CANARY COMPLETE; no full report score; no automatic expansion', flush=True)


if __name__ == '__main__':
    main()
