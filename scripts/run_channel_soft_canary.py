"""First available classification/text/retrieval cases; frozen .5 soft guidance.

Classification is mediated through a conditional language distribution, NOT a
native image-classifier-to-token mapping. No reference answers are loaded.
"""
import argparse
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from run_soft_guidance_full import output, sha
from run_soft_guidance_shard import validate

from merit_feddg.capabilities import EvidenceItem
from merit_feddg.capability_runtime import NativeSession, NativeState, ValueGenerationConfig
from merit_feddg.generalist_factory import load_generalist
from merit_feddg.matched_evaluation import generation_prompt
from merit_feddg.open_study import atomic_json, fingerprint
from merit_feddg.soft_guidance import decode
from merit_feddg.text_kb import TextKnowledgeExpert, build


def visible(transport):
    return {(r['expert_id'], r['evidence_id']) for r in transport['presented']}


def case(probe, row, old, protocol, channel, kb):
    start = time.perf_counter()
    cfg = ValueGenerationConfig(**old['generation_config'])
    if cfg.evidence_style != 'semantic' or cfg.semantic_spatial or cfg.vector_gate != 'off':
        raise RuntimeError('original semantic ungated control required')
    prompt = generation_prompt(row, protocol['config'])
    native = tuple(EvidenceItem(**e) for e in old['evidence'])
    original = NativeSession(probe, row['image'], prompt, row['question'], cfg)
    t = time.perf_counter()
    incumbent = output(original.propose(NativeState(items=native), cfg.max_new_tokens), time.perf_counter()-t)
    delivered = visible(original.last_transport)
    selected = tuple(e for e in native if e.capability == channel and (e.expert_id, e.evidence_id) in delivered)
    result = {'id': row['id'], 'channel': channel, 'current_incumbent': incumbent,
        'original_transport': original.last_transport, 'source_expert_calls': 0,
        'historical_token_parity': incumbent['token_ids'] == old['token_ids'],
        'native_class_to_token_logits': False, 'alpha': .5}
    if channel == 'retrieval':
        t = time.perf_counter()
        retrieved = kb.infer(SimpleNamespace(question=row['question']))
        result['retrieval_seconds'] = time.perf_counter()-t
        selected = tuple(retrieved.items)
        condition_items = native + selected
        base_items = native
    else:
        condition_items = native
        base_items = tuple(e for e in native if e not in selected)
    if not selected:
        return {**result, 'status': 'unavailable', 'reason': 'no_delivered_channel',
                'wall_seconds': time.perf_counter()-start}
    conditioned = NativeSession(probe, row['image'], prompt, row['question'], cfg)
    t = time.perf_counter()
    hard = conditioned.propose(NativeState(items=condition_items), cfg.max_new_tokens)
    result['hard_text'] = output(hard, time.perf_counter()-t)
    if channel != 'retrieval' and list(hard.tokens) != incumbent['token_ids']:
        raise RuntimeError('same-current-context token parity failure')
    shown = visible(conditioned.last_transport)
    selected_ids = {(e.expert_id, e.evidence_id) for e in selected}
    if not selected_ids <= shown or not delivered <= shown:
        return {**result, 'status': 'unavailable', 'reason': 'new_channel_omitted_or_old_evidence_displaced',
                'conditioned_transport': conditioned.last_transport,
                'wall_seconds': time.perf_counter()-start}
    context = NativeSession(probe, row['image'], prompt, row['question'], cfg)
    image, base_prompt = context.context(NativeState(items=base_items))
    base = probe.new_answer_session(image, base_prompt)
    eos = probe.tokenizer.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    result['without_channel'] = decode(base, None, alpha=0, max_tokens=cfg.max_new_tokens, eos_ids=eos_ids)
    t = time.perf_counter()
    direct = base.propose((), count=1, length=cfg.max_new_tokens)[0]
    result['zero_verification_seconds'] = time.perf_counter()-t
    if list(direct.tokens) != result['without_channel']['token_ids']:
        raise RuntimeError('zero control token parity failure')
    result['soft'] = decode(base, conditioned._session, alpha=.5, max_tokens=cfg.max_new_tokens, eos_ids=eos_ids)
    if not result['soft']['text']:
        raise RuntimeError('empty real candidate')
    result.update(status='real_candidate', conditioned_transport=conditioned.last_transport,
                  selected_evidence=[e.evidence_id for e in selected],
                  wall_seconds=time.perf_counter()-start)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--check-only', action='store_true')
    args = p.parse_args()
    frozen, spec = validate(args.base_run)
    seed = Path('assets/knowledge/imaging_seed.json')
    identity = fingerprint({'source_identity': args.base_run.name, 'runner_sha256': sha(__file__),
        'seed_sha256': sha(seed), 'alpha': .5, 'selection': 'first cached capability; first manifest row for retrieval'})
    os.umask(0o002)
    root = args.output / identity
    root.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (root / '.worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        index = root / 'knowledge.sqlite'
        if not index.exists():
            build(seed, index)
        kb = TextKnowledgeExpert(str(index))
        selected = []
        for dataset, data in frozen['datasets'].items():
            remaining = {'classification', 'generation'}
            for row in data['rows']:
                path = Path(data['base']) / 'case-cache/compact_rows' / (fingerprint(row['id']) + '.json')
                entry = json.loads(path.read_text())
                if entry['identity'] != data['protocol']['identity']:
                    raise RuntimeError('expert source mismatch')
                caps = {e['capability'] for e in entry['output']['evidence']}
                for channel in sorted(remaining & caps):
                    selected.append((dataset, channel, row, path))
                    remaining.remove(channel)
                if not remaining:
                    break
            row = data['rows'][0]
            selected.append((dataset, 'retrieval', row, Path(data['base']) / 'case-cache/compact_rows' / (fingerprint(row['id']) + '.json')))
        print('ROOT', root.resolve(), 'SELECTION', [(d, c, r['id']) for d, c, r, _ in selected], flush=True)
        if args.check_only:
            return
        torch.set_num_threads(4)
        torch.ones(1, device='cuda:0').sum().item()
        t = time.perf_counter()
        probe = load_generalist(spec, 'artifacts')
        probe.model.eval().requires_grad_(False)
        atomic_json(root / f'load-{time.time_ns()}.json', {'seconds': time.perf_counter()-t,
            'identity': identity, 'seed_sha256': sha(seed), 'runner_sha256': sha(__file__),
            'timing_caveat': 'shared GPU with full run; not isolated latency'})
        for dataset, channel, row, path in selected:
            target = root / (dataset + '-' + channel + '.json')
            if target.exists():
                raise RuntimeError('existing case output; inspect before resume')
            with torch.inference_mode():
                result = case(probe, row, json.loads(path.read_text())['output'],
                              frozen['datasets'][dataset]['protocol'], channel, kb)
            atomic_json(target, {'identity': identity, 'dataset': dataset, **result})
            print('DONE CHANNEL', dataset, channel, result['status'], flush=True)
        print('CANARY FINISHED; no full score or automatic expansion', flush=True)


if __name__ == '__main__':
    main()
