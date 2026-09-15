"""Scheduling-only continuation of a frozen full run; never changes full_case."""
import argparse
import fcntl
import json
import os
import socket
import subprocess
import time
from pathlib import Path

import torch
from run_soft_guidance_full import ARMS, full_case, sha

from merit_feddg.generalist_factory import generalist_provenance, load_generalist
from merit_feddg.open_study import atomic_json, fingerprint


def validate(root):
    frozen = json.loads((root / 'frozen.json').read_text())
    if root.name != fingerprint(frozen):
        raise RuntimeError('frozen identity mismatch')
    for path, digest in frozen['source'].items():
        if sha(path) != digest:
            raise RuntimeError('frozen implementation changed: ' + path)
    if sha(Path(__file__).with_name('run_soft_guidance_full.py')) != frozen['runner_sha256']:
        raise RuntimeError('original runner changed')
    runtime = {'torch': torch.__version__, 'cuda': torch.version.cuda, 'threads': 4}
    if runtime != frozen['runtime']:
        raise RuntimeError('runtime mismatch')
    for data in frozen['datasets'].values():
        if sha(data['manifest']) != data['manifest_sha256']:
            raise RuntimeError('manifest changed')
    spec = frozen['datasets']['vqarad']['protocol']['config']['generalist']
    actual = json.loads(json.dumps(generalist_provenance(spec, 'artifacts')))
    if actual != frozen['model']:
        raise RuntimeError('model/source identity mismatch')
    return frozen, spec


def assigned(index, shard):
    return index % 2 == shard


def valid_record(path, identity, case_id):
    record = json.loads(path.read_text())
    if record['identity'] != identity or record['id'] != case_id or set(record['arms']) != set(ARMS):
        raise RuntimeError('invalid completed record')
    if any(not a['text'] or not a['token_ids'] for a in record['arms'].values()):
        raise RuntimeError('empty completed arm')


def merge(root, frozen):
    with (root / '.merge.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for data_name, data in frozen['datasets'].items():
            expected = {r['id'] for r in data['rows']}
            actual = {p.stem for p in (root / data_name).glob('*.json')}
            if actual - expected:
                raise RuntimeError('unexpected case IDs')
            if actual != expected:
                print('MERGE PENDING', data_name, len(actual), len(expected), flush=True)
                return False
            for case_id in expected:
                valid_record(root / data_name / (case_id + '.json'), root.name, case_id)
        for name, data in frozen['datasets'].items():
            marker = root / (name + '-complete.json')
            if not marker.exists():
                atomic_json(marker, {'n': len(data['rows']), 'identity': root.name})
        if not (root / 'complete.json').exists():
            atomic_json(root / 'complete.json', {'identity': root.name,
                'datasets': {k: len(v['rows']) for k, v in frozen['datasets'].items()},
                'full_dataset_complete': True})
        print('FULL SHARD MERGE COMPLETE', flush=True)
        return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--shard-index', type=int, choices=(0, 1), required=True)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--max-cases', type=int, default=0)
    args = parser.parse_args()
    if args.max_cases < 0:
        raise ValueError('negative scheduling limit')
    root = args.run
    frozen, spec = validate(root)
    print('FROZEN CHECK PASSED', root.name, flush=True)
    if args.check_only:
        return
    os.umask(0o002)
    # Shared lock excludes the old single-worker EX lock; shard locks exclude duplicates.
    with (root / '.worker.lock').open('r') as legacy, (root / f'.shard-{args.shard_index}.lock').open('a') as shard:
        fcntl.flock(legacy, fcntl.LOCK_SH | fcntl.LOCK_NB)
        fcntl.flock(shard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(4)
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable')
        torch.ones(1, device='cuda:0').sum().item()
        started = time.perf_counter()
        probe = load_generalist(spec, 'artifacts')
        probe.model.eval().requires_grad_(False)
        atomic_json(root / f'load-shard-{args.shard_index}-{time.time_ns()}.json', {
            'seconds': time.perf_counter() - started, 'hostname': socket.gethostname(),
            'device': str(torch.cuda.get_device_properties(0)), 'shard_index': args.shard_index,
            'scheduler_sha256': sha(__file__), 'identity': root.name})
        completed = 0
        for dataset, data in frozen['datasets'].items():
            dest = root / dataset
            dest.mkdir(exist_ok=True)
            base = Path(data['base'])
            generalists = json.loads((base / 'generalist.json').read_text())
            if set(generalists) != {r['id'] for r in data['rows']}:
                raise RuntimeError('generalist IDs mismatch')
            for index, row in enumerate(data['rows']):
                if not assigned(index, args.shard_index):
                    continue
                target = dest / (row['id'] + '.json')
                if target.exists():
                    valid_record(target, root.name, row['id'])
                    continue
                if args.max_cases and completed >= args.max_cases:
                    print('SCHEDULING STOP', flush=True)
                    return
                entry = json.loads((base / 'case-cache/compact_rows' / (fingerprint(row['id']) + '.json')).read_text())
                if entry['identity'] != data['protocol']['identity']:
                    raise RuntimeError('expert identity mismatch')
                try:
                    result = full_case(probe, row, entry['output'], generalists[row['id']], data['protocol'], frozen['alpha'])
                except Exception as exc:
                    atomic_json(root / f'failure-{row["id"]}-{time.time_ns()}.json',
                        {'id': row['id'], 'error_type': type(exc).__name__, 'error': str(exc)})
                    raise
                atomic_json(target, {'identity': root.name, 'dataset': dataset,
                    'worker': {'shard_index': args.shard_index, 'hostname': socket.gethostname()}, **result})
                completed += 1
                print('DONE', dataset, row['id'], 'seconds', round(result['wall_seconds'], 3), flush=True)
        if merge(root, frozen):
            # Only the final successful worker scores; use the unchanged frozen evaluator.
            with (root / '.score.lock').open('a') as score_lock:
                fcntl.flock(score_lock, fcntl.LOCK_EX)
                if not (root / 'evaluation.json').exists():
                    subprocess.run(['/home/dbw/merit-feddg/.venv/bin/python',
                        str(Path(__file__).with_name('evaluate_soft_guidance_full.py')),
                        '--run', str(root)], check=True)


if __name__ == '__main__':
    main()
