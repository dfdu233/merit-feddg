"""TRAIN development: image-grounded pairwise selection, not answer rewriting.

Adapted from FastChat's two-order pairwise judge (NeurIPS 2023) and
LLaVA-Critic's image/question/two-answer input (CVPR 2025). Uses existing
Huatuo, NOT the pretrained LLaVA-Critic checkpoint; no reproduction claim.
"""
import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path


def verdict(text):
    labels = re.findall(r'\[\[([ABC])\]\]', text)
    if len(labels) != 1:
        raise ValueError('Malformed pairwise verdict; no implicit fallback')
    return labels[0]


def selection(forward, reverse):
    mapping = ({'A': 'generalist', 'B': 'compact', 'C': 'tie'},
               {'A': 'compact', 'B': 'generalist', 'C': 'tie'})
    winners = [mapping[0][forward], mapping[1][reverse]]
    if winners == ['compact', 'compact']:
        return 'compact', 'order_consistent_compact'
    if winners == ['generalist', 'generalist']:
        return 'generalist', 'order_consistent_generalist'
    return 'generalist', 'tie_or_order_inconsistent'


def comparison_prompt(question, first, second, channel):
    instruction = ('Compare two answers to the question using the original image. Judge factual '
        'correctness and relevance, not answer order, verbosity, or writing style. Neither '
        'answer is a reference. ')
    if channel == 'finite_choice':
        instruction += ('Return only A if answer A is better, B if answer B is better, or C '
                        'if tied or you cannot distinguish their correctness.\n')
    elif channel == 'free_text':
        instruction += ('Briefly explain which answer is better, or whether they '
            'are tied. End with exactly one verdict: [[A]] for A, [[B]] for B, or [[C]] for '
            'a tie.\n')
    else:
        raise ValueError('Unknown decision channel')
    return instruction + json.dumps({'question': question, 'A': first, 'B': second}, ensure_ascii=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu-uuid')
    p.add_argument('--canary-cases', type=int)
    p.add_argument('--check-only', action='store_true')
    p.add_argument('--decision-channel', choices=['free_text', 'finite_choice'], default='free_text')
    a = p.parse_args()
    import run_huatuo_admission_probe as setup
    native, checked = setup.native, setup.checked
    source = native.read(a.base/'protocol.json')
    if native.read(a.base/'complete.json')['identity'] != source['identity']:
        raise ValueError('Incomplete native controls')
    for path, sha in (source['source'] | source['formal_sources'] |
                      source['base']['huatuo_canary_sources']).items():
        if native.sha(path) != sha:
            raise ValueError('Frozen native dependency changed: '+path)
    cfg = {'method': 'two-order image-grounded Huatuo pairwise selection',
        'source_sha256': native.sha(__file__), 'base_identity': source['identity'],
        'ids': [r['id'] for r in source['rows']],
        'judge_tokens': 8 if a.decision_channel == 'finite_choice' else 512,
        'decision_channel': a.decision_channel,
        'decision_labels_are_calibrated_confidence': False,
        'control_hashes': {r['id']: native.sha(a.base/'cases'/(r['id']+'.json')) for r in source['rows']},
        'official_code': 'lm-sys/FastChat@587d5cfa1609a43d192cedb8441cac3c17db105d',
        'training': False, 'numeric_threshold': False, 'references_at_inference': False,
        'policy': 'select compact only if both orderings select compact; ties/disagreement keep baseline',
        'raw_expert_context_to_judge': False, 'new_answer_generation': False,
        'evaluation_status': 'development; these TRAIN images have already been inspected'}
    identity = native.fingerprint(cfg)
    protocol = {'identity': identity, **cfg}
    pp = a.output/'protocol.json'
    if pp.exists() and native.read(pp) != protocol:
        raise ValueError('Preserve old identity')
    native.atomic_json(pp, protocol)
    print('PREFLIGHT', identity, len(source['rows']), flush=True)
    if a.check_only:
        return
    allowed = ('GPU-3846413a-4238-d307-b1f3-10c2dfbe002c', 'GPU-809e1541-5fe0-e1a6-d360-d0ea647e9023')
    device = subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).strip()
    free = int(subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True,timeout=10).strip())
    if a.gpu_uuid not in allowed or device != a.gpu_uuid or os.environ.get('CUDA_VISIBLE_DEVICES') != '0' or free < 24000:
        raise RuntimeError('GPU mapping/headroom failed')
    import torch
    from transformers import set_seed
    torch.set_num_threads(4)
    start = time.perf_counter()
    model = checked.HuatuoGeneralist(source['base']['generalist']['checkpoint_path'])
    native.atomic_json(a.output/'load.json', {'seconds': time.perf_counter()-start})
    for row in source['rows'][:a.canary_cases]:
        path = a.output/'cases'/(row['id']+'.json')
        if path.exists():
            if native.read(path)['identity'] != identity or not native.read(path)['complete']:
                raise ValueError('Invalid existing case')
            continue
        if native.sha(row['image']) != row['image_sha256']:
            raise ValueError('Image changed')
        prior = native.read(a.base/'cases'/(row['id']+'.json'))
        answers = prior['arms']
        out = {'id': row['id'], 'identity': identity, 'complete': False,
               'calls': [], 'verdicts': [], 'arms': {}}
        if answers['generalist']['text'] == answers['compact']['text']:
            chosen, reason = 'generalist', 'identical_text_no_selection_opportunity'
        else:
            for order in [('generalist','compact'), ('compact','generalist')]:
                prompt = comparison_prompt(row['question'], answers[order[0]]['text'],
                                           answers[order[1]]['text'], a.decision_channel)
                if not model.context_token_budget(row['image'], prompt, cfg['judge_tokens'])['fits']:
                    raise RuntimeError('Judge context exceeds budget; do not truncate answers')
                set_seed(42)
                start = time.perf_counter()
                kwargs = {'allowed_texts': ('A', 'B', 'C')} if a.decision_channel == 'finite_choice' else {}
                response = model.generate_with_usage(row['image'], prompt, cfg['judge_tokens'], **kwargs)
                out['calls'].append({'order': order, 'seconds': time.perf_counter()-start, 'usage': response})
                native.atomic_json(a.output/'progress'/(row['id']+'.json'), out)
                if a.decision_channel == 'finite_choice':
                    if response['text'] not in ('A', 'B', 'C'):
                        raise ValueError('Invalid finite action; never silently KEEP')
                    out['verdicts'].append(response['text'])
                else:
                    out['verdicts'].append(verdict(response['text']))
            chosen, reason = selection(*out['verdicts'])
        out.update(selected=chosen, reason=reason, complete=True)
        out['arms']['pairwise_selected'] = dict(answers[chosen], reuse=chosen, new_answer_calls=0)
        native.atomic_json(path, out)
        print('DONE', row['id'], chosen, reason, flush=True)
    files = list((a.output/'cases').glob('*.json'))
    if {f.stem for f in files} == set(cfg['ids']) and all(native.read(f)['complete'] and native.read(f)['identity']==identity for f in files):
        native.atomic_json(a.output/'complete.json', {'identity': identity, 'n': len(files)})
    else:
        print('CANARY STOP: incomplete full schedule, no full-run score', len(files), flush=True)


if __name__ == '__main__':
    main()
