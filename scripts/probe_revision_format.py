"""Eight-case TRAIN behavior diagnostic; never loads references or scores accuracy.

Change only JSON versus plain draft serialization. Empty answers are observations
in this diagnostic, not repaired outputs or completed clinical predictions.
"""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from run_anchored_revision_train import GPUS, ROOT, read, resources, sha

from merit_feddg.anchored_revision import assert_same_delivery, revision_prompt
from merit_feddg.open_study import atomic_json, fingerprint


def plain_draft_prompt(original_prompt, draft):
    rendered = revision_prompt(original_prompt, draft)
    suffix = 'Draft answer (JSON string): ' + json.dumps(draft, ensure_ascii=False)
    if not rendered.endswith(suffix):
        raise ValueError('Original draft serialization changed')
    return rendered[:-len(suffix)] + 'Draft answer:\n' + draft


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu-uuid', required=True, choices=sorted(GPUS))
    p.add_argument('--check-only', action='store_true')
    a = p.parse_args()
    prior, rows, compact, generalist, source = resources()
    selected = prior['selected'][:8]
    cfg = {'schema': 'revision-format-behavior-v1', 'source': prior,
           'selected': selected, 'code_sha256': sha(__file__),
           'arms': ['generalist', 'compact', 'json_none', 'json_evidence',
                    'plain_none', 'plain_evidence'],
           'selection': 'first eight of prior frozen answer-blind TRAIN-only image order',
           'outcome': 'nonempty, wrapper behavior, token parity and unchanged evidence; no accuracy',
           'stop': 'runtime/parity/delivery failure; empty outputs recorded without fallback',
           'no_prompt_search': True, 'no_reference_access': True}
    identity = fingerprint(cfg)
    a.output.mkdir(parents=True, exist_ok=True)
    frozen = a.output / 'frozen.json'
    if frozen.exists() and read(frozen) != cfg:
        raise ValueError('Preserve frozen diagnostic')
    if not frozen.exists():
        atomic_json(frozen, cfg)
    print('PREFLIGHT', identity, len(selected), flush=True)
    if a.check_only:
        return
    device = subprocess.check_output(['nvidia-smi', '-i', '0', '--query-gpu=uuid',
        '--format=csv,noheader'], text=True, timeout=10).strip()
    free = int(subprocess.check_output(['nvidia-smi', '-i', '0', '--query-gpu=memory.free',
        '--format=csv,noheader,nounits'], text=True, timeout=10).strip())
    if device != a.gpu_uuid or os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
        raise ValueError('GPU mapping not authorized')
    if free < 22000:
        raise RuntimeError('Insufficient safe headroom; existing jobs not interrupted')
    import torch
    from transformers import set_seed

    from merit_feddg.capabilities import EvidenceItem
    from merit_feddg.capability_runtime import NativeSession, NativeState, ValueGenerationConfig
    from merit_feddg.generalist_factory import load_generalist
    from merit_feddg.matched_evaluation import generation_prompt
    from merit_feddg.open_data import pixel_digest
    torch.set_num_threads(4)
    torch.ones(1, device='cuda:0').sum().item()
    started = time.perf_counter()
    probe = load_generalist(source['config']['generalist'], str(ROOT / 'artifacts'))
    probe.model.eval().requires_grad_(False)
    atomic_json(a.output / f'load-{time.time_ns()}.json', {
        'identity': identity, 'seconds': time.perf_counter() - started, 'gpu_uuid': device})
    for index, key in enumerate(selected):
        path = a.output / 'cases' / (key + '.json')
        record = read(path) if path.exists() else {
            'identity': identity, 'id': key, 'arms': {}, 'complete': False}
        if record['identity'] != identity:
            raise ValueError('Case identity mismatch')
        if record['complete']:
            continue
        row, old, base = rows[key], compact[key], generalist[key]
        if pixel_digest(row['image']) != row['image_sha256']:
            raise ValueError('Image changed')
        gen = ValueGenerationConfig(**old['generation_config'])
        if gen.semantic_spatial or gen.visual_views or gen.evidence_style != 'semantic':
            raise ValueError('Unexpected historical text protocol')
        items = tuple(EvidenceItem(**e) for e in old['evidence'])
        prompt = generation_prompt(row, source['config'])
        json_prompt = revision_prompt(prompt, base['text'])
        plain_prompt = plain_draft_prompt(prompt, base['text'])
        for name, instruction, evidence in (
            ('generalist', prompt, ()), ('compact', prompt, items),
            ('json_none', json_prompt, ()), ('json_evidence', json_prompt, items),
            ('plain_none', plain_prompt, ()), ('plain_evidence', plain_prompt, items),
        ):
            if name in record['arms']:
                continue
            generation = (ValueGenerationConfig(**base['generation_config'])
                          if name == 'generalist' else gen)
            session = NativeSession(probe, row['image'], instruction, row['question'], generation)
            image, compiled = session.context(NativeState(items=evidence))
            if name.endswith('_evidence'):
                original = record['arms']['compact']['transport']
                assert_same_delivery(original, session.last_transport)
                if original['evidence_sha256'] != session.last_transport['evidence_sha256']:
                    raise RuntimeError('Delivered evidence content changed')
            set_seed(42)
            start = time.perf_counter()
            with torch.inference_mode():
                block = probe.new_answer_session(image, compiled).propose(
                    (), count=1, length=generation.max_new_tokens)[0]
            text = session.decode(block.tokens).strip()
            value = {'text': text, 'token_ids': list(block.tokens), 'finished': block.finished,
                     'nonempty': bool(text), 'seconds': time.perf_counter() - start,
                     'transport': session.last_transport, 'calls': 1,
                     'prompt_sha256': fingerprint(compiled)}
            if name in ('generalist', 'compact'):
                expected = base if name == 'generalist' else old
                value['token_parity'] = value['token_ids'] == expected['token_ids']
                value['text_parity'] = text == expected['text']
            record['arms'][name] = value
            atomic_json(path, record)  # preserve every call, even before a later failure
            if name in ('generalist', 'compact') and not (
                value['token_parity'] and value['text_parity']
            ):
                raise RuntimeError('Historical parity failed')
        record['complete'] = True
        atomic_json(path, record)
        print('DONE', index + 1, key, {n: v['nonempty'] for n, v in record['arms'].items()}, flush=True)
    records = [read(a.output / 'cases' / (k + '.json')) for k in selected]
    if any(not r['complete'] or set(r['arms']) != set(cfg['arms']) for r in records):
        raise RuntimeError('Incomplete diagnostic')
    summary = {'identity': identity, 'n': len(selected), 'complete': True,
               'medical_scoring': False, 'arms': {}, 'references_loaded': False,
               'model_load_seconds': sum(read(q)['seconds'] for q in a.output.glob('load-*.json'))}
    for name in cfg['arms']:
        values = [r['arms'][name] for r in records]
        summary['arms'][name] = {
            'nonempty': sum(v['nonempty'] for v in values),
            'not_eos_finished': sum(not v['finished'] for v in values),
            'seconds': sum(v['seconds'] for v in values), 'calls': len(values),
            'text_equal_to_generalist': sum(r['arms'][name]['text'] ==
                r['arms']['generalist']['text'] for r in records)}
    atomic_json(a.output / 'behavior.json', summary)
    print('COMPLETE', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
