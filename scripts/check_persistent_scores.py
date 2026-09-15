"""Real-data exact score/token audit; writes to a new local output only."""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from run_class_text_guidance import GPU_UUID, guided_case
from run_soft_guidance_shard import validate
from merit_feddg.generalist_factory import load_generalist
from merit_feddg.open_study import atomic_json, fingerprint
from merit_feddg.persistent_scores import PersistentScores


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu-uuid', default=GPU_UUID,
                   choices=(GPU_UUID, 'GPU-809e1541-5fe0-e1a6-d360-d0ea647e9023'))
    p.add_argument('--reference-run', type=Path)
    args = p.parse_args()
    if args.output.exists():
        raise RuntimeError('audit output already exists')
    source, spec = validate(args.base_run)
    uuid = subprocess.check_output(['nvidia-smi', '-i', '0', '--query-gpu=uuid',
        '--format=csv,noheader'], text=True, timeout=10).strip()
    if uuid != args.gpu_uuid or os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
        raise RuntimeError('unauthorized device')
    torch.set_num_threads(4)
    probe = load_generalist(spec, 'artifacts')
    probe.model.eval().requires_grad_(False)
    factory = probe.new_answer_session
    checks = []

    class Audited(PersistentScores):
        def next_scores(self, prefix):
            torch.cuda.synchronize()
            start = time.perf_counter()
            fast = super().next_scores(prefix)
            torch.cuda.synchronize()
            fast_time = time.perf_counter()-start
            start = time.perf_counter()
            reference = self.session.next_scores(prefix)
            torch.cuda.synchronize()
            equal = np.array_equal(fast, reference)
            checks.append(dict(prefix_length=len(prefix), exact=equal,
                max_abs_delta=float(np.max(np.abs(fast-reference))),
                fast_seconds=fast_time, replay_seconds=time.perf_counter()-start))
            if not equal:
                atomic_json(args.output, dict(status='parity_failed', checks=checks))
                raise RuntimeError('exact score parity failed; full runner unchanged')
            return fast

    probe.new_answer_session = lambda *a, **k: Audited(factory(*a, **k))
    cases = []
    for dataset, data in source['datasets'].items():
        for channel in ('classification', 'generation'):
            for row in data['rows']:
                old = json.loads((Path(data['base'])/'case-cache/compact_rows'/
                    (fingerprint(row['id'])+'.json')).read_text())['output']
                if not any(e['capability'] == channel for e in old['evidence']):
                    continue
                with torch.inference_mode():
                    result = guided_case(probe, row, old, data['protocol'], channel)
                if result['status'] != 'real_candidate':
                    raise RuntimeError('canary has no real candidate')
                if args.reference_run:
                    reference = json.loads((args.reference_run/dataset/channel/(row['id']+'.json')).read_text())
                    for arm in ('current_incumbent', 'without_channel', 'soft', 'cad'):
                        if result[arm]['token_ids'] != reference[arm]['token_ids']:
                            raise RuntimeError('cross-device candidate token parity failed: '+arm)
                cases.append(dict(dataset=dataset, channel=channel, id=row['id'],
                    blend_tokens=len(result['soft']['token_ids']), cad_tokens=len(result['cad']['token_ids'])))
                print('PASS', dataset, channel, len(checks), flush=True)
                break
    atomic_json(args.output, dict(status='exact_parity_passed', cases=cases, checks=checks,
        gpu_uuid=uuid, reference_run=str(args.reference_run) if args.reference_run else None,
        timing_caveat='GPU shared with existing full run; timings are not isolated speed benchmarks'))


if __name__ == '__main__':
    main()
