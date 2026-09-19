"""TRAIN development: image-grounded pairwise selection, not answer rewriting.

Adapted from FastChat's two-order pairwise judge (NeurIPS 2023) and
LLaVA-Critic's image/question/two-answer input (CVPR 2025). Supports existing
Huatuo or a frozen independent LLaVA-Critic judge; no reproduction claim.
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path


def verdict(text):
    # Conditional option-list echoes are not verdicts; a separate explicit
    # bracketed decision is still required. Do not choose among actual conflicts.
    text = re.sub(r'(?m)^\s*-\s*\[[ABC]\]\s+if\b[^\n]*', '', text)
    matches = re.findall(r'(?<!\[)(?:\[\[([ABC])\]\]|\[([ABC])\])(?!\])', text)
    labels = [a or b for a, b in matches]
    if len(labels) > 1 and len(set(labels)) == 1:
        terminal = re.search(r'(?im)^\s*(?:final\s+)?verdict:\s*(?:\[\[([ABC])\]\]|\[([ABC])\])[.\s]*$', text)
        if terminal and (terminal.group(1) or terminal.group(2)) == labels[0]:
            return labels[0]
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


def comparison_orders(case_id, mode):
    normal = ('generalist', 'compact')
    reverse = ('compact', 'generalist')
    if mode == 'single':
        # Label-free deterministic positioning, no second judge call.
        return [reverse if int(hashlib.sha256(case_id.encode()).hexdigest(), 16) % 2 else normal]
    if mode != 'double':
        raise ValueError('Unknown comparison order mode')
    return [normal, reverse]


def single_selection(label, order):
    if label not in ('A', 'B', 'C'):
        raise ValueError('Invalid single verdict')
    return ('generalist', 'single_tie') if label == 'C' else (
        order[0 if label == 'A' else 1], 'single_judge_choice')


def comparison_prompt(question, first, second, channel):
    if channel == 'critic_reasoned':
        # Official LLaVA-Critic pairwise layout, plus explicit machine verdict.
        # This is an adaptation: neither paper reproduction nor confidence.
        return ('Given an image and a corresponding question, please serve as an unbiased '
            'and fair judge to evaluate the quality of the answers provided by a Large '
            'Multimodal Model (LMM). Determine which answer is better and explain your '
            'reasoning with specific details. Your task is provided as follows:\n'
            f'Question: [{question}]\nThe first response: [{first}]\n'
            f'The second response: [{second}]\n'
            'Neither response is a reference answer. After your explanation, end with '
            'exactly one verdict: [[A]] if the first response is better, [[B]] if the '
            'second response is better, or [[C]] if tied or not distinguishable.\nASSISTANT:\n')
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
    p.add_argument('--decision-channel', choices=['free_text', 'finite_choice', 'critic_reasoned'], default='free_text')
    p.add_argument('--judge', choices=['huatuo', 'llava_critic'], default='huatuo')
    p.add_argument('--comparison-orders', choices=['single', 'double'], default='double')
    p.add_argument('--critic-attention', choices=['eager', 'sdpa'], default='eager')
    p.add_argument('--reuse-judgments', type=Path,
                   help='Reparse immutable saved calls from an otherwise identical run')
    p.add_argument('--image-control', choices=['original', 'cyclic_next'], default='original',
                   help='cyclic_next is a deliberately mismatched image diagnostic, never a patient prediction')
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
    cfg = {'method': 'two-order image-grounded pairwise selection',
        'source_sha256': native.sha(__file__), 'base_identity': source['identity'],
        'ids': [r['id'] for r in source['rows']],
        'judge_tokens': 8 if a.decision_channel == 'finite_choice' else 512,
        'decision_channel': a.decision_channel,
        'decision_labels_are_calibrated_confidence': False,
        'image_control': a.image_control,
        'patient_prediction': a.image_control == 'original',
        'control_hashes': {r['id']: native.sha(a.base/'cases'/(r['id']+'.json')) for r in source['rows']},
        'official_code': 'lm-sys/FastChat@587d5cfa1609a43d192cedb8441cac3c17db105d',
        'training': False, 'numeric_threshold': False, 'references_at_inference': False,
        'policy': 'select compact only if both orderings select compact; ties/disagreement keep baseline',
        'raw_expert_context_to_judge': False, 'new_answer_generation': False,
        'evaluation_status': 'development; these TRAIN images have already been inspected'}
    cfg['judge'] = a.judge
    cfg['comparison_orders'] = a.comparison_orders
    if a.comparison_orders == 'single':
        cfg['policy'] = 'one judge call, SHA256(id) parity orders candidates; tie keeps generalist'
    if a.judge == 'llava_critic':
        from llava_critic_backend import identity as critic_identity
        cfg['critic'] = critic_identity(a.critic_attention)
        cfg['critic']['decision_channel'] = a.decision_channel
    elif a.decision_channel == 'critic_reasoned':
        raise ValueError('Native critic prompt requires the independent critic backend')
    cfg['verdict_parser'] = 'consistent_explicit_terminal_bracket_v2'
    reused = {}
    if a.reuse_judgments:
        old = native.read(a.reuse_judgments/'protocol.json')
        comparable = lambda d: {k:v for k,v in d.items()
            if k not in ('identity', 'source_sha256', 'verdict_parser', 'reused_judgments')}
        if comparable(old) != comparable(cfg):
            raise ValueError('Cannot reuse different model, inputs, ordering or generation protocol')
        snapshots = {}
        for key in cfg['ids']:
            candidates = [a.reuse_judgments/folder/(key+'.json') for folder in ('cases', 'progress')]
            saved_path = next((f for f in candidates if f.exists()), None)
            if saved_path is not None:
                value = native.read(saved_path)
                if value['identity'] != old['identity'] or value['id'] != key:
                    raise ValueError('Reused judgment identity mismatch')
                reused[key] = value
                snapshots[str(saved_path.resolve())] = native.sha(saved_path)
        cfg['reused_judgments'] = {'identity': old['identity'],
            'protocol_sha256': native.sha(a.reuse_judgments/'protocol.json'), 'files': snapshots}
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
    if a.judge == 'llava_critic':
        from llava_critic_backend import LlavaCritic
        model = LlavaCritic(a.critic_attention)
    else:
        model = checked.HuatuoGeneralist(source['base']['generalist']['checkpoint_path'])
    load_record = {'seconds': time.perf_counter()-start,
                   'weight_loading': getattr(model, 'loading_info', None)}
    load_path = a.output/'load.json'
    if load_path.exists():
        load_path = a.output/('load-resume-'+str(time.time_ns())+'.json')
    native.atomic_json(load_path, load_record)
    for index, row in enumerate(source['rows'][:a.canary_cases]):
        path = a.output/'cases'/(row['id']+'.json')
        if path.exists():
            if native.read(path)['identity'] != identity or not native.read(path)['complete']:
                raise ValueError('Invalid existing case')
            continue
        if native.sha(row['image']) != row['image_sha256']:
            raise ValueError('Image changed')
        visual = row if a.image_control == 'original' else source['rows'][(index+1) % len(source['rows'])]
        if native.sha(visual['image']) != visual['image_sha256']:
            raise ValueError('Control image changed')
        if a.image_control != 'original' and visual.get('pixel_sha256', visual['image_sha256']) == row.get('pixel_sha256', row['image_sha256']):
            raise ValueError('Mismatched image control must actually differ')
        prior = native.read(a.base/'cases'/(row['id']+'.json'))
        answers = prior['arms']
        out = {'id': row['id'], 'identity': identity, 'complete': False,
               'calls': [], 'verdicts': [], 'arms': {},
               'image_control': a.image_control, 'delivered_image_id': visual['id'],
               'delivered_image_sha256': visual['image_sha256']}
        if answers['generalist']['text'] == answers['compact']['text']:
            chosen, reason = 'generalist', 'identical_text_no_selection_opportunity'
        else:
            orders = comparison_orders(row['id'], a.comparison_orders)
            cached = reused.get(row['id'])
            if cached and (cached['delivered_image_sha256'] != visual['image_sha256'] or
                           cached['delivered_image_id'] != visual['id']):
                raise ValueError('Reused call image mismatch')
            for call_index, order in enumerate(orders):
                prompt = comparison_prompt(row['question'], answers[order[0]]['text'],
                                           answers[order[1]]['text'], a.decision_channel)
                if not model.context_token_budget(visual['image'], prompt, cfg['judge_tokens'])['fits']:
                    raise RuntimeError('Judge context exceeds budget; do not truncate answers')
                set_seed(42)
                start = time.perf_counter()
                kwargs = {'allowed_texts': ('A', 'B', 'C')} if a.decision_channel == 'finite_choice' else {}
                if cached and call_index < len(cached['calls']):
                    call = dict(cached['calls'][call_index])
                    if tuple(call['order']) != tuple(order):
                        raise ValueError('Reused candidate order mismatch')
                    response = call['usage']
                    call['reused_judgment'] = True
                else:
                    response = model.generate_with_usage(visual['image'], prompt, cfg['judge_tokens'], **kwargs)
                    call = {'order': order, 'seconds': time.perf_counter()-start, 'usage': response}
                out['calls'].append(call)
                native.atomic_json(a.output/'progress'/(row['id']+'.json'), out)
                if a.decision_channel == 'finite_choice':
                    if response['text'] not in ('A', 'B', 'C'):
                        raise ValueError('Invalid finite action; never silently KEEP')
                    out['verdicts'].append(response['text'])
                else:
                    out['verdicts'].append(verdict(response['text']))
            chosen, reason = (single_selection(out['verdicts'][0], orders[0])
                if a.comparison_orders == 'single' else selection(*out['verdicts']))
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
