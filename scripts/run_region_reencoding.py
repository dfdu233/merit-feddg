"""Bounded, opt-in TRAIN experiment. Does not resume or alter stopped CRES runs."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
from pathlib import Path
from time import perf_counter

from merit_feddg.region_reencoding import ARMS, reencoding_case


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def train_rows(path, loader):
    """Inspect raw split/label metadata BEFORE the legacy loader normalizes it."""
    raw = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not raw or any(row.get('official_split') != 'train' for row in raw):
        raise ValueError('unchanged official TRAIN manifest required; test use is not enabled')
    forbidden = ('answer', 'answers', 'label', 'reference', 'references', 'mask', 'target_mask', 'report')
    if any(any(k in row for k in forbidden) for row in raw):
        raise ValueError('manifest contains target evidence')
    rows = loader(path)
    if any(row.get('task') != 'open_vqa' for row in rows):
        raise ValueError('this 64-token experiment is VQA-only, not report generation')
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--anchor-root', type=Path, help='Existing scorer path; hashes pinned before model allocation')
    p.add_argument('--artifacts', type=Path, default=Path('artifacts'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu-uuid', help='Explicitly authorized device, verified before loading')
    p.add_argument('--max-cases', type=int, default=2, help='New completed cases per invocation; positive bounded budget')
    p.add_argument('--max-seconds', type=float, default=600., help='Cooperative stop between cases; not a hard interrupt')
    p.add_argument('--check-only', action='store_true')
    args = p.parse_args()
    import math
    if args.max_cases < 1 or not math.isfinite(args.max_seconds) or args.max_seconds <= 0:
        raise ValueError('positive case/time budgets required; no implicit full run')
    # These are read-only helpers from the frozen CRES runner, not its main loop.
    from run_control_evidence import check_route, image_identity, routing_origin
    from merit_feddg.control_study import validate_outputs
    from merit_feddg.matched_evaluation import load_manifest
    from merit_feddg.open_study import atomic_json, fingerprint

    rows = train_rows(args.manifest, load_manifest)
    protocol = json.loads((args.source_run/'protocol.json').read_text())
    routes = json.loads((args.source_run/'routing.json').read_text())
    if (protocol.get('n') != len(rows) or protocol.get('shards_complete') is not True
            or set(routes) != {r['id'] for r in rows} or len(set(r['id'] for r in rows)) != len(rows)):
        raise ValueError('complete, unique, same-manifest source run required')
    origin = routing_origin(protocol)
    caches, images = {}, {}
    for row in rows:
        check_route(row, routes[row['id']], origin)
        images[row['image']] = image_identity(row)
        path = args.source_run/'case-cache'/'compact_rows'/(fingerprint(row['id'])+'.json')
        cache = json.loads(path.read_text())
        if cache.get('identity') != protocol['identity']:
            raise ValueError('source cache identity mismatch')
        caches[row['id']] = sha(path)
    spec = {**protocol['config']['generalist'], 'training_free_spatial': True,
            'deterministic_image_padding': True}
    if spec.get('backend') != 'llava_med' or spec.get('tensor_bridge_checkpoint'):
        raise ValueError('first adapter only supports frozen LLaVA-Med; no learned bridge')
    if args.check_only:
        print(json.dumps({'preflight': True, 'n': len(rows), 'model_loaded': False,
            'arms': ARMS, 'references_loaded': False, 'source': str(args.source_run.resolve()),
            'hypothesis': 'native crop detail helps beyond low-detail and displaced controls'}))
        return
    import torch
    from merit_feddg.generalist_factory import generalist_provenance, load_generalist
    if args.anchor_root is None:
        raise ValueError('freeze existing scorer via --anchor-root before generation')
    scorer_names = ('anchor/corrected_sgta/evaluate_medheval_answers.py', 'anchor/medeval/evaluate_mixed_vqa_table.py')
    scorer_hashes = {n: sha(args.anchor_root/n) for n in scorer_names}
    if not args.gpu_uuid or os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
        raise ValueError('explicit authorized UUID and visible local device 0 required')
    actual = subprocess.check_output(['nvidia-smi', '-i', '0', '--query-gpu=uuid',
                '--format=csv,noheader'], timeout=10, text=True).strip()
    if actual != args.gpu_uuid:
        raise RuntimeError('device identity mismatch')
    os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'
    torch.set_num_threads(4)
    source_root = Path(__file__).resolve().parents[1]
    frozen = {'schema': 'region-reencoding-v1', 'base_commit': '02c9397852e876b2a6aaba2f1a2f9f8c60d8b2e6',
        'options': {'prompt_contract': 'uniform', 'mask_cutoff': .5, 'region_selection': 'first_usable_stored_mask',
                    'fusion': '(original + mask * aligned_local) / (1 + mask)', 'max_new_tokens': 64},
        'arms': list(ARMS), 'rows': rows, 'manifest': str(args.manifest.resolve()), 'manifest_sha256': sha(args.manifest),
        'source_run': str(args.source_run.resolve()), 'source_protocol_sha256': sha(args.source_run/'protocol.json'),
        'source_caches': caches, 'image_file_sha256': images, 'generalist_spec': spec,
        'model': generalist_provenance(spec, args.artifacts),
        'code': {str(f.relative_to(source_root)): sha(f) for f in sorted((source_root/'merit_feddg').rglob('*.py'))},
        'runner_sha256': sha(__file__), 'base_runner_sha256': sha(Path(__file__).with_name('run_control_evidence.py')),
        'evaluator_sha256': sha(Path(__file__).with_name('evaluate_region_reencoding.py')),
        'runtime': {'torch': torch.__version__, 'cuda': torch.version.cuda, 'threads': 4},
        'training': False, 'labels_in_generation': False, 'scorer_hashes': scorer_hashes}
    frozen = json.loads(json.dumps(frozen))
    identity = fingerprint(frozen)
    root = args.output/identity
    root.mkdir(parents=True, exist_ok=True)
    with (root/'.worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root/'frozen.json').exists() and json.loads((root/'frozen.json').read_text()) != frozen:
            raise RuntimeError('frozen identity mismatch')
        atomic_json(root/'frozen.json', frozen)
        pin = root/'scorer-frozen.json'
        if pin.exists() and json.loads(pin.read_text()) != scorer_hashes:
            raise ValueError('scorer pin changed')
        atomic_json(pin, scorer_hashes)
        print('ROOT', root.resolve(), flush=True)
        start = perf_counter()
        probe = load_generalist(spec, args.artifacts)
        probe.model.eval().requires_grad_(False)
        atomic_json(root/f'startup-{len(list(root.glob("startup-*.json"))):03d}.json',
            {'seconds': perf_counter()-start, 'gpu_uuid': actual,
             'max_cases': args.max_cases, 'max_seconds': args.max_seconds})
        dest = root/'cases'
        dest.mkdir(exist_ok=True)
        executed, loop_start = 0, perf_counter()
        for row in rows:
            target = dest/(fingerprint(row['id'])+'.json')
            if target.exists():
                cached = json.loads(target.read_text())
                if cached.get('identity') != identity or cached.get('id') != row['id']:
                    raise ValueError('resumed row identity mismatch')
                validate_outputs(cached, ARMS)
                continue
            if executed >= args.max_cases or perf_counter()-loop_start >= args.max_seconds:
                atomic_json(root/f'stop-{len(list(root.glob("stop-*.json"))):03d}.json',
                    {'new_cases': executed, 'full_manifest_complete': False,
                     'reason': 'explicit_case_or_cooperative_time_budget'})
                print('STOP: bounded prefix, not a complete result', flush=True)
                return
            path = args.source_run/'case-cache'/'compact_rows'/(fingerprint(row['id'])+'.json')
            if sha(path) != caches[row['id']] or sha(row['image']) != images[row['image']]:
                raise RuntimeError('inputs changed after preflight')
            try:
                with torch.inference_mode():
                    result = reencoding_case(probe, row, json.loads(path.read_text())['output'], protocol)
            except Exception as exc:
                atomic_json(root/('failure-'+fingerprint(row['id'])+'.json'),
                    {'id': row['id'], 'type': type(exc).__name__, 'message': str(exc)})
                raise
            atomic_json(target, {'identity': identity, 'gpu_uuid': actual, **result})
            executed += 1
            print('DONE', row['id'], result['region_audit']['reason'], flush=True)
        if {f.stem for f in dest.glob('*.json')} != {fingerprint(r['id']) for r in rows}:
            raise ValueError('extra or missing case IDs')
        atomic_json(root/'complete.json', {'identity': identity, 'n': len(rows), 'full_manifest_complete': True})


if __name__ == '__main__':
    main()
