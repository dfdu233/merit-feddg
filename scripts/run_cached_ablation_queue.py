"""Run full-protocol Huatuo ablations on answer-blind, fully cached case subsets.

Outputs stay explicitly partial until the full official manifest is covered.
The queue waits for one already-running SLAKE worker, then works through small
to large VQA datasets using only exact native-cache hits.
"""

import argparse
import gzip
import json
import os
import pathlib
import subprocess
import sys
import time


ROOT = pathlib.Path('/home/dbw/merit-feddg-huatuo-spatial')
PYTHON = '/home/dbw/.runtime/miniconda3/envs/huatuo/bin/python'
MANIFESTS = {
    'pathvqa': 'runs/llava-test-v1/pathvqa-manifest.jsonl',
    'mmmu': 'runs/llava-test-v1/mmmu-manifest.jsonl',
    'pmcvqa': 'runs/llava-test-v1/pmcvqa-manifest.jsonl',
    'omnimedvqa': 'runs/llava-test-v1/omnimedvqa-manifest.jsonl',
}
CACHES = {
    'pathvqa': 'runs/llava-test-v1/pathvqa-expert-cache/b2fb387f19bd079203a85db91599dd4c0b08489d11a1b63a60c560a361f03084',
    'mmmu': 'runs/llava-test-v1/mmmu-expert-cache/bbf4b8aa15a659224747e723bdc36338905ee02f4a59ec56fe44162df2915dbb',
    'pmcvqa': 'runs/llava-test-v1/pmcvqa-expert-cache/3d798e8e253a45479101eea46aaec4ffce11efb9e53fecbf472a9f5ff572227e',
    'omnimedvqa': 'runs/llava-test-v1/omnimedvqa-expert-cache/5b93d68625e2e8ffaac18d33aefdb56315bd30b459e6f602dfff6ac29dc016c7',
}
ARMS = ('generalist', 'joint_all', 'isolated_mean', 'isolated_geomedian', 'bard')


def run_case_subset(gpu, dataset, ids, canary=False):
    manifest = MANIFESTS[dataset]
    output = (f'runs/researchstudio-{dataset}-huatuo-cachefill-canary-gpu{gpu}'
              if canary else f'runs/researchstudio-{dataset}-huatuo-cachefill-v1')
    log = pathlib.Path(f'{output}.log') if canary else pathlib.Path(
        f'runs/researchstudio-{dataset}-huatuo-cachefill-gpu{gpu}.log')
    command = [PYTHON, '-u', 'scripts/run_adaptive_bard_canary.py',
               '--cache', CACHES[dataset], '--config', 'runs/huatuo-test-v1/config.yaml',
               '--manifest', manifest, '--output', output, '--methods', *ARMS,
               '--matched-joint-evidence', '--skip-stress', '--cached-receiver',
               '--compress-artifacts', '--reference-native-evidence',
               '--shard-index', str(gpu), '--shard-count', '2',
               '--start-index', '0', '--end-index', '999999', '--max-forwards',
               '200000000']
    for sample_id in ids:
        command.extend(('--case-id', sample_id))
    print(('CANARY' if canary else 'RUN'), dataset, 'GPU', gpu,
          'cases', len(ids), 'log', log, flush=True)
    with (ROOT / log).open('a', encoding='utf-8') as stream:
        return subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT).returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, choices=(0, 1), required=True)
    parser.add_argument('--predecessor', type=int, required=True)
    args = parser.parse_args()
    if args.predecessor > 1:
        proc = pathlib.Path(f'/proc/{args.predecessor}/cmdline')
        if proc.exists():
            command = proc.read_bytes().replace(b'\0', b' ').decode(errors='replace')
            if 'run_adaptive_bard_canary.py' not in command or 'researchstudio-slake' not in command:
                raise SystemExit('Predecessor PID is not the expected SLAKE worker')
            print('WAIT_FOR_SLAKE', args.predecessor, 'GPU', args.gpu, flush=True)
            subprocess.run(['tail', f'--pid={args.predecessor}', '-f', '/dev/null'], check=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    time.sleep(5)

    for dataset in ('pathvqa', 'mmmu', 'pmcvqa', 'omnimedvqa'):
        ready_path = ROOT / f'runs/researchstudio-ready-{dataset}.json'
        deadline = time.time() + 1800
        while not ready_path.exists() and time.time() < deadline:
            print('WAIT_READY_LIST', dataset, flush=True)
            time.sleep(10)
        if not ready_path.exists():
            print('SKIP_NO_READY_LIST', dataset, flush=True)
            continue
        ready = json.loads(ready_path.read_text())
        if ready['denominator'] != sum(1 for _ in (ROOT / MANIFESTS[dataset]).open()):
            print('SKIP_MANIFEST_SIZE_MISMATCH', dataset, flush=True)
            continue
        ids = [item['id'] for item in ready['ready'] if item['index'] % 2 == args.gpu]
        if not ids:
            print('SKIP_NO_CACHED_CASES', dataset, flush=True)
            continue
        free = int(subprocess.check_output(
            ['df', '--output=avail', '-B1', str(ROOT / 'runs')], text=True).splitlines()[-1])
        if free < 1073741824:
            print('STOP_LOW_DISK', dataset, free, flush=True)
            return 4
        canary_id = ids[0]
        result = run_case_subset(args.gpu, dataset, [canary_id], canary=True)
        if result:
            print('CANARY_FAILED_CONTINUE_NEXT_DATASET', dataset, result, flush=True)
            continue
        canary_dir = ROOT / f'runs/researchstudio-{dataset}-huatuo-cachefill-canary-gpu{args.gpu}' / canary_id
        try:
            for arm in ARMS:
                with gzip.open(canary_dir / f'{arm}.json.gz', 'rt', encoding='utf-8') as stream:
                    if not json.load(stream).get('text', '').strip():
                        raise ValueError(f'empty output: {arm}')
            if not (canary_dir / 'provenance.json.gz').is_file():
                raise ValueError('missing provenance')
        except Exception as error:
            print('CANARY_INVALID_CONTINUE_NEXT_DATASET', dataset, repr(error), flush=True)
            continue
        result = run_case_subset(args.gpu, dataset, ids, canary=False)
        print('DATASET_QUEUE_EXIT', dataset, 'GPU', args.gpu, 'code', result,
              'selected', len(ids), 'official_n', ready['denominator'], flush=True)
    print('READY_CACHE_QUEUE_EXIT', 'GPU', args.gpu, flush=True)


if __name__ == '__main__':
    main()
