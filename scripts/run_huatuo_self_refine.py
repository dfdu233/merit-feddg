"""One Self-Refine iteration, with/without native evidence, on frozen Huatuo controls.

Mechanism adapted from madaan/self-refine (NeurIPS2023), not a reproduction of
its task-specific few-shot prompts. No trained verifier, score gate or test labels.
"""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import run_huatuo_admission_probe as setup

native, checked = setup.native, setup.checked
BASE = Path('/home/dbw/merit-feddg-huatuo-admission/runs/huatuo-train-admission-v5')
FEEDBACK = (
    '\nReview the proposed answer to the original image question. Give specific, actionable '
    'feedback on correctness, relevance and consistency with the image and any supplied observations. '
    'Distinguish what is actually observed from assumptions made by the answer. '
    'The specialist observations, if present, are fallible inputs rather than a reference answer. '
    'Do not invent facts. If you find no concrete error, say that no concrete error was found. '
    'Provide feedback, not a replacement answer.\nProposed answer: '
)
REFINE = (
    '\nRevise the proposed answer using the feedback. Correct identified errors while preserving '
    'supported content. The feedback may itself be mistaken; check it against the original image '
    'and supplied information. Return only the final answer to the original question, '
    'without describing the revision process.\nProposed answer: '
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--base', type=Path, default=BASE)
    p.add_argument('--gpu-uuid')
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--shard-count', type=int, default=1)
    p.add_argument('--check-only', action='store_true')
    p.add_argument('--merge-only', action='store_true')
    a = p.parse_args()
    source = native.read(a.base / 'protocol.json')
    complete = native.read(a.base / 'complete.json')
    if source['identity'] != complete['identity']:
        raise ValueError('Native source not complete')
    for path, digest in (source['source'] | source['formal_sources'] |
                         source['base']['huatuo_canary_sources']).items():
        if native.sha(path) != digest:
            raise ValueError('Native dependency changed: ' + path)
    rows = source['rows']
    cfg = {'method': 'one-step Self-Refine adaptation', 'base': str(a.base.resolve()),
        'base_identity': source['identity'], 'ids': [r['id'] for r in rows],
        'controls': {r['id']: native.sha(a.base / 'cases' / (r['id'] + '.json')) for r in rows},
        'source_sha256': native.sha(__file__), 'feedback_prompt': FEEDBACK, 'refine_prompt': REFINE,
        'feedback_tokens': 256, 'answer_tokens': 1024, 'iterations': 1,
        'arms': ['self_refine_blind', 'self_refine_evidence'],
        'paper': 'https://papers.nips.cc/paper_files/paper/2023/hash/91edff07232fb1b55a505a9e9f6c0ff3-Abstract-Conference.html',
        'official_code': 'madaan/self-refine@9a206d41e5d2d0c241bb441f41eeadb945afaa55',
        'no_training': True, 'no_threshold': True, 'test_labels_read': False}
    identity = native.fingerprint(cfg)
    frozen = a.output / 'protocol.json'
    payload = dict(identity=identity, **cfg)
    if frozen.exists() and native.read(frozen) != payload:
        raise ValueError('Frozen experiment changed')
    if not frozen.exists():
        native.atomic_json(frozen, payload)
    print('PREFLIGHT', identity, len(rows), flush=True)
    if a.check_only:
        return
    if a.merge_only:
        paths = {p.stem: p for p in (a.output / 'cases').glob('*.json')}
        if set(paths) != set(cfg['ids']):
            raise ValueError('Exact full probe required')
        if any(not native.read(p)['complete'] or native.read(p)['identity'] != identity for p in paths.values()):
            raise ValueError('Incomplete probe cases')
        native.atomic_json(a.output / 'complete.json', {'identity': identity, 'n': len(rows)})
        return
    if not 0 <= a.shard_index < a.shard_count:
        raise ValueError('Invalid shard')
    if a.gpu_uuid not in ('GPU-3846413a-4238-d307-b1f3-10c2dfbe002c',
                          'GPU-809e1541-5fe0-e1a6-d360-d0ea647e9023'):
        raise ValueError('Unauthorized GPU')
    device = subprocess.check_output(['nvidia-smi', '-i', '0', '--query-gpu=uuid',
        '--format=csv,noheader'], text=True, timeout=10).strip()
    free = int(subprocess.check_output(['nvidia-smi', '-i', '0', '--query-gpu=memory.free',
        '--format=csv,noheader,nounits'], text=True, timeout=10).strip())
    if device != a.gpu_uuid or os.environ.get('CUDA_VISIBLE_DEVICES') != '0' or free < 24000:
        raise RuntimeError('GPU mapping/headroom check failed; leave other jobs alone')
    import torch
    from transformers import set_seed

    from merit_feddg.capabilities import EvidenceItem
    from merit_feddg.capability_runtime import NativeState, ValueGenerationConfig
    torch.set_num_threads(4)
    torch.ones(1, device='cuda:0').sum().item()
    start = time.perf_counter()
    probe = checked.HuatuoGeneralist(source['base']['generalist']['checkpoint_path'])
    native.atomic_json(a.output / f'load-{a.shard_index}.json', {'seconds': time.perf_counter()-start})
    arm = ValueGenerationConfig(**source['generation_config'])
    for i, row in enumerate(rows):
        if i % a.shard_count != a.shard_index:
            continue
        path = a.output / 'cases' / (row['id'] + '.json')
        if path.exists():
            if native.read(path)['identity'] != identity or not native.read(path)['complete']:
                raise ValueError('Preserve incomplete/incompatible case')
            continue
        if native.sha(row['image']) != row['image_sha256']:
            raise ValueError('Image changed')
        prior = native.read(a.base / 'cases' / (row['id'] + '.json'))
        items = tuple(EvidenceItem(**e) for e in prior['compact_raw']['evidence'])
        session = checked.NativeSession(probe, row['image'], row['benchmark_prompt'], row['question'], arm)
        _, compiled = session.context(NativeState(items=items))
        out = {'identity': identity, 'id': row['id'], 'complete': False, 'arms': {},
               'original_transport': session.last_transport, 'calls': []}
        # Exact replay validates real formal serialization before any adaptation.
        if i == 0:
            set_seed(42)
            start = time.perf_counter()
            replay = probe.generate_with_usage(row['image'], compiled, 1024)
            out['parity'] = {'seconds': time.perf_counter()-start,
                'passed': replay['text'] == prior['arms']['compact']['text'] and
                          replay['token_ids'] == prior['arms']['compact']['token_ids']}
            if not out['parity']['passed']:
                native.atomic_json(path, out)
                raise RuntimeError('Original Huatuo compact parity failed')
        for name in cfg['arms']:
            context = compiled if name.endswith('evidence') else row['benchmark_prompt']
            draft = prior['arms']['compact']['text']
            feedback_prompt = context + FEEDBACK + json.dumps(draft, ensure_ascii=False)
            if not probe.context_token_budget(row['image'], feedback_prompt, 256)['fits']:
                raise RuntimeError('Feedback budget unavailable; never displace evidence')
            set_seed(42)
            start = time.perf_counter()
            feedback = probe.generate_with_usage(row['image'], feedback_prompt, 256)
            out['calls'].append({'arm': name, 'stage': 'feedback', 'usage': feedback,
                                 'seconds': time.perf_counter()-start})
            native.atomic_json(a.output / 'progress' / (row['id']+'.json'), out)
            prompt = context + REFINE + json.dumps(draft, ensure_ascii=False) + '\nFeedback: ' + json.dumps(feedback['text'], ensure_ascii=False)
            if not probe.context_token_budget(row['image'], prompt, 1024)['fits']:
                raise RuntimeError('Refinement budget unavailable; never displace evidence')
            set_seed(42)
            start = time.perf_counter()
            answer = probe.generate_with_usage(row['image'], prompt, 1024)
            out['calls'].append({'arm': name, 'stage': 'refine', 'usage': answer,
                                 'seconds': time.perf_counter()-start})
            out['arms'][name] = answer
        out['complete'] = True
        native.atomic_json(path, out)
        print('DONE', row['id'], {k: len(v['token_ids']) for k,v in out['arms'].items()}, flush=True)


if __name__ == '__main__':
    main()
