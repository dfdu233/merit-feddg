"""Full classification/generated-text channel comparisons, physical GPU1 only.

CAD adaptation: z = 1.5 z_with - .5 z_without (Shi et al., NAACL 2024).
The separate convex arm uses .5/.5. No fitted confidence or native class-logit
alignment is claimed. Old model, expert, renderer and scoring code is unchanged.
"""
import argparse
import fcntl
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from run_channel_soft_canary import visible
from run_soft_guidance_full import output, sha
from run_soft_guidance_shard import validate

from merit_feddg.capabilities import EvidenceItem
from merit_feddg.capability_runtime import NativeSession, NativeState, ValueGenerationConfig
from merit_feddg.generalist_factory import load_generalist
from merit_feddg.matched_evaluation import generation_prompt
from merit_feddg.open_study import atomic_json, fingerprint
from merit_feddg.soft_guidance import decode

CHANNELS = ('classification', 'generation')
ARMS = ('text', 'without_channel', 'blend', 'cad')
GPU_UUID = 'GPU-3846413a-4238-d307-b1f3-10c2dfbe002c'


def cad_scores(without, conditioned, strength=.5):
    a, b = np.asarray(without, dtype=np.float64), np.asarray(conditioned, dtype=np.float64)
    if not np.isfinite(strength) or strength < 0 or a.ndim != 1 or a.shape != b.shape:
        raise ValueError('invalid CAD inputs')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('nonfinite CAD input')
    return (1 + strength) * b - strength * a


def cad_decode(base, conditioned, max_tokens, eos_ids, strength=.5):
    start = time.perf_counter()
    if strength == 0:
        block = conditioned.propose((), count=1, length=max_tokens)[0]
        return {'text': block.text.strip(), 'token_ids': list(block.tokens),
                'seconds': time.perf_counter()-start, 'score_calls': 0}
    prefix, steps = [], []
    for _ in range(max_tokens):
        a, b = base.next_scores(tuple(prefix)), conditioned.next_scores(tuple(prefix))
        token = int(np.argmax(cad_scores(a, b, strength)))
        steps.append({'without_top': int(np.argmax(a)), 'with_top': int(np.argmax(b)),
                      'selected': token, 'max_logit_delta': float(np.max(np.abs(a-b)))})
        prefix.append(token)
        if token in eos_ids:
            break
    return {'text': base.decode(prefix).strip(), 'token_ids': prefix,
            'seconds': time.perf_counter()-start, 'score_calls': 2*len(steps),
            'steps': steps, 'implementation': 'same-prefix production replay; no KV optimization'}


def guided_case(probe, row, old, protocol, channel):
    started = time.perf_counter()
    cfg = ValueGenerationConfig(**old['generation_config'])
    if cfg.evidence_style != 'semantic' or cfg.semantic_spatial or cfg.vector_gate != 'off':
        raise RuntimeError('original ungated semantic configuration required')
    prompt = generation_prompt(row, protocol['config'])
    native = tuple(EvidenceItem(**e) for e in old['evidence'])
    original = NativeSession(probe, row['image'], prompt, row['question'], cfg)
    t = time.perf_counter()
    hard = original.propose(NativeState(items=native), cfg.max_new_tokens)
    incumbent = output(hard, time.perf_counter()-t)
    result = {'id': row['id'], 'channel': channel, 'current_incumbent': incumbent,
        'original_transport': original.last_transport, 'source_expert_calls': 0,
        'historical_token_parity': incumbent['token_ids'] == old['token_ids'],
        'native_class_to_token_logits': False, 'alpha': .5}
    delivered = visible(original.last_transport)
    selected_ids = {(e.expert_id, e.evidence_id) for e in native
                    if e.capability == channel and (e.expert_id, e.evidence_id) in delivered}
    if not selected_ids:
        return {**result, 'status': 'unavailable', 'reason': 'channel_not_presented',
                'wall_seconds': time.perf_counter()-started}
    # Removing one channel must not admit formerly omitted evidence into another.
    other = tuple(e for e in native if (e.expert_id, e.evidence_id) in delivered-selected_ids)
    context = NativeSession(probe, row['image'], prompt, row['question'], cfg)
    image, rendered = context.context(NativeState(items=other))
    if visible(context.last_transport) != delivered-selected_ids:
        raise RuntimeError('nonintervened evidence delivery changed')
    base = probe.new_answer_session(image, rendered)
    contexts = [base, original._session]
    eos = probe.tokenizer.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    result['without_channel'] = decode(base, None, alpha=0, max_tokens=cfg.max_new_tokens, eos_ids=eos_ids)
    t = time.perf_counter()
    direct = base.propose((), count=1, length=cfg.max_new_tokens)[0]
    result['zero_verification_seconds'] = time.perf_counter()-t
    if list(direct.tokens) != result['without_channel']['token_ids']:
        raise RuntimeError('zero-strength without-control parity failure')
    result['soft'] = decode(*contexts, alpha=.5, max_tokens=cfg.max_new_tokens, eos_ids=eos_ids)
    zero = cad_decode(*contexts, cfg.max_new_tokens, eos_ids, strength=0)
    if zero['token_ids'] != incumbent['token_ids']:
        raise RuntimeError('CAD zero-strength hard-control parity failure')
    result['cad_zero_verification_seconds'] = zero['seconds']
    result['cad'] = cad_decode(*contexts, cfg.max_new_tokens, eos_ids)
    result.update(wall_seconds=time.perf_counter()-started, status='real_candidate',
                  without_transport=context.last_transport, selected_evidence=sorted(selected_ids))
    if not result['cad']['text'] or not result['soft']['text']:
        raise RuntimeError('empty CAD candidate')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--canary', action='store_true', help='first available case per channel/dataset; same full manifest')
    args = parser.parse_args()
    source, spec = validate(args.base_run)
    if not json.loads((args.base_run/'complete.json').read_text())['full_dataset_complete']:
        raise RuntimeError('complete source run required')
    scripts = ('run_class_text_guidance.py', 'run_channel_soft_canary.py', 'run_soft_guidance_full.py',
               'evaluate_class_text_guidance.py')
    frozen = {'schema': 'class-text-guidance-v1', 'source_run': str(args.base_run.resolve()),
        'source_identity': args.base_run.name, 'channels': CHANNELS, 'arms': ARMS,
        'blend': .5, 'cad_strength': .5, 'gpu_uuid': GPU_UUID,
        'native_class_to_token_logits': False, 'no_test_parameter_selection': True,
        'scripts': {s: sha(Path(__file__).with_name(s)) for s in scripts},
        'scorer': json.loads((args.base_run/'scorer-frozen.json').read_text())}
    frozen = json.loads(json.dumps(frozen))
    root = args.output/fingerprint(frozen)
    root.mkdir(parents=True, exist_ok=True)
    if (root/'frozen.json').exists():
        if json.loads((root/'frozen.json').read_text()) != frozen:
            raise RuntimeError('frozen settings changed')
    else:
        atomic_json(root/'frozen.json', frozen)
    print('ROOT', root.resolve(), flush=True)
    if args.check_only:
        return
    with (root/'.worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Check exact physical device before model allocation; never probe host GPU0.
        uuid = subprocess.check_output(['nvidia-smi', '-i', '0', '--query-gpu=uuid',
                                        '--format=csv,noheader'], timeout=10, text=True).strip()
        if uuid != GPU_UUID or os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
            raise RuntimeError('unauthorized GPU mapping')
        torch.set_num_threads(4)
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable')
        torch.ones(1, device='cuda:0').sum().item()
        t = time.perf_counter()
        probe = load_generalist(spec, 'artifacts')
        probe.model.eval().requires_grad_(False)
        atomic_json(root/f'load-{time.time_ns()}.json', {'seconds': time.perf_counter()-t, 'gpu_uuid': uuid})
        for dataset, data in source['datasets'].items():
            for channel in CHANNELS:
                dest = root/dataset/channel
                dest.mkdir(parents=True, exist_ok=True)
                for row in data['rows']:
                    path = dest/(row['id']+'.json')
                    if path.exists():
                        cached = json.loads(path.read_text())
                        if cached['identity'] != root.name or cached['id'] != row['id']:
                            raise RuntimeError('invalid resumed record')
                        if args.canary and cached['status'] == 'real_candidate':
                            break
                        continue
                    entry = json.loads((Path(data['base'])/'case-cache/compact_rows'/
                                       (fingerprint(row['id'])+'.json')).read_text())
                    if entry['identity'] != data['protocol']['identity']:
                        raise RuntimeError('expert cache changed')
                    old = entry['output']
                    if channel not in {e['capability'] for e in old['evidence']}:
                        if args.canary:
                            continue
                        incumbent = json.loads((args.base_run/dataset/(row['id']+'.json')).read_text())['arms']['compact_matched']
                        result = {'id': row['id'], 'channel': channel, 'status': 'unavailable',
                            'reason': 'no_cached_channel', 'current_incumbent': incumbent,
                            'incumbent_reused': True, 'wall_seconds': 0.}
                    else:
                        with torch.inference_mode():
                            result = guided_case(probe, row, old, data['protocol'], channel)
                    atomic_json(path, {'identity': root.name, 'dataset': dataset, **result})
                    print('DONE', dataset, channel, row['id'], result['status'], flush=True)
                    if args.canary:
                        if result['status'] != 'real_candidate':
                            raise RuntimeError('first available canary channel was not delivered')
                        break
                if not args.canary and {p.stem for p in dest.glob('*.json')} != {r['id'] for r in data['rows']}:
                    raise RuntimeError('full IDs incomplete')
        if args.canary:
            atomic_json(root/'canary-complete.json', {'identity': root.name, 'full_dataset_complete': False})
            print('CANARY COMPLETE; not a full score', flush=True)
        else:
            atomic_json(root/'complete.json', {'identity': root.name, 'full_dataset_complete': True})
            subprocess.run(['/home/dbw/merit-feddg/.venv/bin/python',
                str(Path(__file__).with_name('evaluate_class_text_guidance.py')), '--run', str(root)], check=True)


if __name__ == '__main__':
    main()
