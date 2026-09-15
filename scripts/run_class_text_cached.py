"""Explicit, audited scheduling adapter around the unchanged frozen runner.

One process per scheduling shard. A real four-cell score audit is required.
Old rows remain untouched; every newly generated row labels its cost backend.
"""
import json
import argparse
import fcntl
import os
import subprocess
import sys
import time
from pathlib import Path

import run_class_text_guidance as runner
from run_soft_guidance_full import sha
from merit_feddg.open_study import atomic_json
from merit_feddg.persistent_scores import PersistentScores

HOST_GPU0 = 'GPU-809e1541-5fe0-e1a6-d360-d0ea647e9023'


def assigned(index, shard):
    return index % 2 == shard


def sharded_main(audit_path, provenance):
    """Scheduling-only continuation; never change the frozen generation runner."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--shard-index', type=int, choices=(0, 1), required=True)
    parser.add_argument('--gpu-uuid', choices=(runner.GPU_UUID, HOST_GPU0), required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    # Reuse the ORIGINAL full frozen identity and its unmodified preflight.
    argv = sys.argv
    try:
        sys.argv = [argv[0], '--base-run', str(args.base_run), '--output', str(args.output), '--check-only']
        runner.main()
    finally:
        sys.argv = argv
    expected_scripts = {name: sha(Path(__file__).with_name(name)) for name in
        ('run_class_text_guidance.py', 'run_channel_soft_canary.py',
         'run_soft_guidance_full.py', 'evaluate_class_text_guidance.py')}
    roots = []
    for candidate in args.output.iterdir():
        if not (candidate/'frozen.json').exists():
            continue
        identity = json.loads((candidate/'frozen.json').read_text())
        if (identity.get('scripts') == expected_scripts and
                identity.get('source_run') == str(args.base_run.resolve()) and
                candidate.name == runner.fingerprint(identity)):
            roots.append(candidate)
    if len(roots) != 1:
        raise RuntimeError('ambiguous output root')
    root = roots[0]
    frozen = json.loads((root/'frozen.json').read_text())
    if frozen['source_run'] != str(args.base_run.resolve()):
        raise RuntimeError('source mismatch')
    source, spec = runner.validate(args.base_run)
    audit = json.loads(audit_path.read_text())
    if args.gpu_uuid == HOST_GPU0 and (audit.get('gpu_uuid') != HOST_GPU0 or
            Path(audit.get('reference_run') or '').resolve() != root.resolve()):
        raise RuntimeError('host GPU0 requires its own cross-device token parity audit')
    if args.check_only:
        return
    os.umask(0o002)
    with (root/'.worker.lock').open('r') as legacy, (root/f'.shard-{args.shard_index}.lock').open('a') as shard:
        fcntl.flock(legacy, fcntl.LOCK_SH | fcntl.LOCK_NB)
        fcntl.flock(shard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        uuid = subprocess.check_output(['nvidia-smi', '-i', '0', '--query-gpu=uuid',
            '--format=csv,noheader'], text=True, timeout=10).strip()
        if uuid != args.gpu_uuid or os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
            raise RuntimeError('unauthorized device mapping')
        provenance.update(actual_gpu_uuid=uuid, shard_index=args.shard_index, shard_count=2,
                          frozen_gpu_uuid_is_original_launch_device=True)
        runner.torch.set_num_threads(4)
        runner.torch.ones(1, device='cuda:0').sum().item()
        start = time.perf_counter()
        probe = runner.load_generalist(spec, 'artifacts')
        probe.model.eval().requires_grad_(False)
        atomic_json(root/f'load-shard-{args.shard_index}-{time.time_ns()}.json',
                    {'seconds': time.perf_counter()-start, **provenance})
        for dataset, data in source['datasets'].items():
            for channel in runner.CHANNELS:
                dest = root/dataset/channel
                dest.mkdir(parents=True, exist_ok=True)
                for index, row in enumerate(data['rows']):
                    if not assigned(index, args.shard_index):
                        continue
                    path = dest/(row['id']+'.json')
                    if path.exists():
                        cached = json.loads(path.read_text())
                        if cached['identity'] != root.name or cached['id'] != row['id']:
                            raise RuntimeError('invalid resumed record')
                        continue
                    entry = json.loads((Path(data['base'])/'case-cache/compact_rows'/
                        (runner.fingerprint(row['id'])+'.json')).read_text())
                    if entry['identity'] != data['protocol']['identity']:
                        raise RuntimeError('expert cache changed')
                    old = entry['output']
                    if channel not in {e['capability'] for e in old['evidence']}:
                        incumbent = json.loads((args.base_run/dataset/(row['id']+'.json')).read_text())['arms']['compact_matched']
                        result = {'id': row['id'], 'channel': channel, 'status': 'unavailable',
                            'reason': 'no_cached_channel', 'current_incumbent': incumbent,
                            'incumbent_reused': True, 'wall_seconds': 0.}
                    else:
                        with runner.torch.inference_mode():
                            result = runner.guided_case(probe, row, old, data['protocol'], channel)
                    atomic_json(path, {'identity': root.name, 'dataset': dataset, **result})
                    print('DONE', dataset, channel, row['id'], result['status'], flush=True)
        # Both workers attempt merge; only exact FULL sets admit offline scoring.
        with (root/'.merge.lock').open('a') as merge:
            fcntl.flock(merge, fcntl.LOCK_EX)
            for dataset, data in source['datasets'].items():
                for channel in runner.CHANNELS:
                    files = {p.stem: p for p in (root/dataset/channel).glob('*.json')}
                    expected = {r['id'] for r in data['rows']}
                    if set(files)-expected:
                        raise RuntimeError('unexpected case IDs')
                    if set(files) != expected:
                        print('SHARD COMPLETE; full merge pending', flush=True)
                        return
                    for key, path in files.items():
                        record = json.loads(path.read_text())
                        if record['identity'] != root.name or record['id'] != key:
                            raise RuntimeError('merge record identity mismatch')
            if not (root/'complete.json').exists():
                atomic_json(root/'complete.json', {'identity': root.name, 'full_dataset_complete': True})
            if not (root/'evaluation.json').exists():
                subprocess.run([sys.executable, str(Path(__file__).with_name('evaluate_class_text_guidance.py')),
                                '--run', str(root)], check=True)


def main():
    if '--help' in sys.argv or '-h' in sys.argv:
        print(__doc__ + '\nAdditional required option: --cache-audit PATH')
        runner.main()
        return
    if '--cache-audit' not in sys.argv:
        raise RuntimeError('--cache-audit is required')
    index = sys.argv.index('--cache-audit')
    audit_path = Path(sys.argv[index+1])
    del sys.argv[index:index+2]
    audit = json.loads(audit_path.read_text())
    if audit.get('status') != 'exact_parity_passed' or len(audit.get('cases', [])) != 4:
        raise RuntimeError('four real dataset/channel audit cells required')
    if not audit['checks'] or not all(c['exact'] and c['max_abs_delta'] == 0 for c in audit['checks']):
        raise RuntimeError('exact score parity required')
    admission_path = audit_path.with_suffix('.admission.json')
    admission = json.loads(admission_path.read_text())
    scorer_path = Path(__file__).parents[1]/'merit_feddg/persistent_scores.py'
    if admission != {'audit_sha256': sha(audit_path), 'scorer_sha256': sha(scorer_path)}:
        raise RuntimeError('audit/scorer admission identity mismatch')
    provenance = {'name': 'persistent-production-kv-v1', **admission,
        'wrapper_sha256': sha(__file__), 'scope': 'semantic classification/generation only',
        'parity_scope': 'four fixed real dataset/channel cells; complete score vectors',
        'vision_reuse': 'retained per-branch multimodal prefill KV; no cross-case cache'}
    original_load, original_case = runner.load_generalist, runner.guided_case
    sessions = []

    def load(*args, **kwargs):
        probe = original_load(*args, **kwargs)
        factory = probe.new_answer_session
        def new_session(*a, **k):
            session = PersistentScores(factory(*a, **k))
            sessions.append(session)
            return session
        probe.new_answer_session = new_session
        return probe

    def case(*args, **kwargs):
        sessions.clear()
        try:
            result = original_case(*args, **kwargs)
            result['persistent_scoring_cost'] = {
                'multimodal_prefills': sum(s.prefills for s in sessions),
                'incremental_forwards': sum(s.incremental_calls for s in sessions),
                'scope': 'soft/CAD scoring only; ordinary controls and inherited expert costs remain separate'}
        finally:
            sessions.clear()
        result['execution_backend'] = provenance
        if result['status'] == 'real_candidate':
            result['soft']['cache_policy'] = 'persistent production-prefill; independent branch KV'
            result['cad']['implementation'] = 'same-prefix persistent production KV'
        return result

    runner.load_generalist, runner.guided_case = load, case
    atomic_json(audit_path.parent/f'cache-launch-{time.time_ns()}.json', provenance)
    print('CACHE BACKEND', json.dumps(provenance), flush=True)
    if '--shard-index' in sys.argv:
        sharded_main(audit_path, provenance)
    else:
        runner.main()


if __name__ == '__main__':
    main()
