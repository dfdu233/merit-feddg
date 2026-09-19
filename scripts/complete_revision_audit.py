"""New terminal-outcome protocol for the unchanged, earlier 24 TRAIN cases.

No silent retries, EOS suppression, evidence deletion, or imputed answer. A
budget-invalid arm is explicitly unavailable, not a successful clinical output.
"""
import argparse
import os
import subprocess
import time
from pathlib import Path

from run_anchored_revision_train import GPUS, ROOT, read, resources, sha

from merit_feddg.anchored_revision import revision_prompt
from merit_feddg.open_study import atomic_json, fingerprint

PRIOR = Path('/home/dbw/merit-feddg-anchored-revision/runs/pilot-v4')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu-uuid', required=True, choices=sorted(GPUS))
    a = p.parse_args()
    source, rows, compact, generalist, protocol = resources()
    prior = read(PRIOR / 'frozen.json')
    for key in ('input_hashes', 'model', 'selected'):
        if source[key] != prior[key]:
            raise ValueError('Historical revision inputs/model changed')
    if any(sha(ROOT / name) != value for name, value in prior['source'].items()):
        raise ValueError('Historical revision generation sources changed')
    cfg = {'schema': 'revision-terminal-audit-v1', 'prior': prior,
           'source_sha256': sha(__file__), 'input_outcome_policy':
           'Record unavailable context without inference; keep actual empty outputs; never fallback',
           'reference_policy': 'offline only', 'gpu_scheduling_changes_method': False}
    identity = fingerprint(cfg)
    a.output.mkdir(parents=True, exist_ok=True)
    frozen = a.output / 'frozen.json'
    if frozen.exists() and read(frozen) != cfg:
        raise ValueError('Preserve frozen results')
    if not frozen.exists():
        atomic_json(frozen, cfg)
    device = subprocess.check_output(['nvidia-smi', '-i', '0', '--query-gpu=uuid',
        '--format=csv,noheader'], text=True, timeout=10).strip()
    free = int(subprocess.check_output(['nvidia-smi', '-i', '0', '--query-gpu=memory.free',
        '--format=csv,noheader,nounits'], text=True, timeout=10).strip())
    if device != a.gpu_uuid or os.environ.get('CUDA_VISIBLE_DEVICES') != '0' or free < 22000:
        raise RuntimeError('GPU mapping/headroom check failed')
    import torch
    from transformers import set_seed

    from merit_feddg.capabilities import EvidenceItem
    from merit_feddg.capability_runtime import NativeSession, NativeState, ValueGenerationConfig
    from merit_feddg.generalist_factory import load_generalist
    from merit_feddg.matched_evaluation import generation_prompt
    from merit_feddg.open_data import pixel_digest
    torch.set_num_threads(4)
    start = time.perf_counter()
    probe = load_generalist(protocol['config']['generalist'], str(ROOT / 'artifacts'))
    probe.model.eval().requires_grad_(False)
    atomic_json(a.output / f'load-{time.time_ns()}.json', {'seconds': time.perf_counter()-start})
    for i, key in enumerate(prior['selected']):
        path = a.output / 'cases' / (key + '.json')
        if path.exists() and read(path).get('complete'):
            if read(path)['identity'] != identity:
                raise ValueError('Case identity mismatch')
            continue
        row, old, base = rows[key], compact[key], generalist[key]
        if pixel_digest(row['image']) != row['image_sha256']:
            raise ValueError('Image changed')
        gen = ValueGenerationConfig(**old['generation_config'])
        items = tuple(EvidenceItem(**e) for e in old['evidence'])
        prompt = generation_prompt(row, protocol['config'])
        original = NativeSession(probe, row['image'], prompt, row['question'], gen)
        original.context(NativeState(items=items))
        saved = read(PRIOR / 'cases' / (key + '.json'))['arms'] if (
            PRIOR / 'cases' / (key + '.json')).exists() else {}
        out = {'id': key, 'identity': identity, 'complete': False, 'arms': {}}
        for arm, cached in (('generalist', base), ('compact', old)):
            out['arms'][arm] = {'text': cached['text'], 'token_ids': cached['token_ids'],
                'status': 'generated' if cached['text'] else 'empty', 'reused': True,
                'new_calls': 0, 'new_seconds': 0, 'inherited_seconds': cached['seconds']}
        for arm, evidence in (('revision_no_evidence', ()), ('revision_evidence', items)):
            session = NativeSession(probe, row['image'], revision_prompt(prompt, base['text']),
                                    row['question'], gen)
            image, compiled = session.context(NativeState(items=evidence))
            transport = session.last_transport
            if arm == 'revision_evidence' and (
                transport['evidence_sha256'] != original.last_transport['evidence_sha256']
                or transport['presented'] != original.last_transport['presented']
            ):
                value = {'text': None, 'token_ids': None, 'status': 'unavailable_evidence_displacement',
                         'new_calls': 0, 'new_seconds': 0, 'reused': False}
            else:
                previous = saved.get(arm)
                failure = PRIOR / 'failures' / (key + '-' + arm + '.json')
                if previous is None and failure.exists():
                    previous = read(failure)
                if previous is not None:
                    if previous['transport']['prompt_sha256'] != transport['prompt_sha256']:
                        raise ValueError('Reused prompt changed')
                    value = {'text': previous['text'], 'token_ids': previous['token_ids'],
                             'new_calls': 0, 'new_seconds': 0, 'reused': True,
                             'inherited_seconds': previous['seconds']}
                else:
                    set_seed(42)
                    start = time.perf_counter()
                    with torch.inference_mode():
                        block = probe.new_answer_session(image, compiled).propose((), 1, gen.max_new_tokens)[0]
                    value = {'text': session.decode(block.tokens).strip(), 'token_ids': list(block.tokens),
                             'new_calls': 1, 'new_seconds': time.perf_counter()-start,
                             'finished': block.finished, 'reused': False}
                value['status'] = 'generated' if value['text'] else 'empty'
            value['transport'] = transport
            out['arms'][arm] = value
            atomic_json(path, out)
        out['complete'] = True
        atomic_json(path, out)
        print('DONE', i+1, key, {k:v['status'] for k,v in out['arms'].items()}, flush=True)
    if any(not read(a.output/'cases'/(k+'.json'))['complete'] for k in prior['selected']):
        raise RuntimeError('Missing terminal outcomes')
    atomic_json(a.output/'complete.json', {'identity':identity,'n':len(prior['selected']),
        'terminal_audit_complete':True,'all_arms_clinically_valid':False})


if __name__ == '__main__':
    main()
