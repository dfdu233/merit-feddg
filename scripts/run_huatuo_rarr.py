"""RARR-inspired agreement-gated minimal edits of a Huatuo image-only draft.

Official reference: anthonywchen/RARR@51a1a10fe5bada837a368f98cb55288ac5168c9e.
Native observations replace retrieval; they are fallible, not gold evidence.
"""
import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path

import run_huatuo_admission_probe as setup

native, checked = setup.native, setup.checked
BASE = Path('/home/dbw/merit-feddg-huatuo-admission/runs/huatuo-train-admission-v5')


def decision(text):
    matches = re.findall(r'^\s*Decision:\s*(AGREES|DISAGREES|IRRELEVANT|UNKNOWN)(?:\s*$|\.\s)', text, re.MULTILINE | re.IGNORECASE)
    if len(matches) != 1:
        raise ValueError('Malformed agreement decision: no implicit keep fallback')
    return matches[0].upper()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, default=BASE)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu-uuid')
    p.add_argument('--check-only', action='store_true')
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--shard-count', type=int, default=1)
    p.add_argument('--merge-only', action='store_true')
    p.add_argument('--editors-only', action='store_true', help='Frozen editor with matched no-evidence control; omit failed agreement gate')
    a = p.parse_args()
    source = native.read(a.base / 'protocol.json')
    if native.read(a.base / 'complete.json')['identity'] != source['identity']:
        raise ValueError('Incomplete controls')
    for path, digest in (source['source'] | source['formal_sources'] |
                         source['base']['huatuo_canary_sources']).items():
        if native.sha(path) != digest:
            raise ValueError('Frozen native dependencies changed: ' + path)
    cfg = {'method': 'RARR-inspired sequential agreement/edit; not full RARR reproduction',
        'source_sha256': native.sha(__file__), 'base': str(a.base.resolve()),
        'base_identity': source['identity'], 'ids': [r['id'] for r in source['rows']],
        'control_hashes': {r['id']: native.sha(a.base / 'cases' / (r['id']+'.json')) for r in source['rows']},
        'arms': ['anchored_blind_editor','anchored_editor'] if a.editors_only else ['anchored_editor', 'rarr_agreement'],
        'gate_tokens': 256, 'answer_tokens': 1024,
        'paper': 'https://aclanthology.org/2023.acl-long.910/',
        'official_code': 'anthonywchen/RARR@51a1a10fe5bada837a368f98cb55288ac5168c9e',
        'numeric_threshold': False, 'training': False, 'reference_at_inference': False,
        'agreement_categories': ['AGREES', 'DISAGREES', 'IRRELEVANT', 'UNKNOWN'],
        'failure_policy': 'explicit failure; never interpret parse failure as KEEP'}
    identity = native.fingerprint(cfg)
    payload = dict(identity=identity, **cfg)
    path = a.output / 'protocol.json'
    if path.exists() and native.read(path) != payload:
        raise ValueError('Cannot mix source/config identities')
    if not path.exists():
        native.atomic_json(path, payload)
    print('PREFLIGHT', identity, len(source['rows']), flush=True)
    if a.check_only:
        return
    if a.merge_only:
        files = {p.stem: p for p in (a.output/'cases').glob('*.json')}
        if set(files) != set(cfg['ids']) or any(not native.read(p)['complete'] or
                native.read(p)['identity'] != identity for p in files.values()):
            raise ValueError('Exact complete cases required')
        native.atomic_json(a.output/'complete.json', {'identity': identity, 'n': len(files)})
        return
    if not 0 <= a.shard_index < a.shard_count:
        raise ValueError('Invalid shard')
    if a.gpu_uuid not in ('GPU-3846413a-4238-d307-b1f3-10c2dfbe002c', 'GPU-809e1541-5fe0-e1a6-d360-d0ea647e9023'):
        raise ValueError('Unauthorized GPU')
    device = subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).strip()
    free = int(subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True,timeout=10).strip())
    if device != a.gpu_uuid or os.environ.get('CUDA_VISIBLE_DEVICES') != '0' or free < 24000:
        raise RuntimeError('GPU mapping/headroom failed')
    import torch
    from transformers import set_seed

    from merit_feddg.capabilities import EvidenceItem
    from merit_feddg.capability_runtime import NativeState, ValueGenerationConfig
    from merit_feddg.compact_evidence import compact_records
    torch.set_num_threads(4)
    start = time.perf_counter()
    probe = checked.HuatuoGeneralist(source['base']['generalist']['checkpoint_path'])
    native.atomic_json(a.output/f'load-{a.shard_index}.json', {'seconds': time.perf_counter()-start})
    arm = ValueGenerationConfig(**source['generation_config'])
    for index, row in enumerate(source['rows']):
        if index % a.shard_count != a.shard_index:
            continue
        path = a.output/'cases'/(row['id']+'.json')
        if path.exists():
            if native.read(path)['identity'] != identity or not native.read(path)['complete']:
                raise ValueError('Preserve incomplete case')
            continue
        if native.sha(row['image']) != row['image_sha256']:
            raise ValueError('Image changed')
        prior = native.read(a.base/'cases'/(row['id']+'.json'))
        items = tuple(EvidenceItem(**e) for e in prior['compact_raw']['evidence'])
        session = checked.NativeSession(probe,row['image'],row['benchmark_prompt'],row['question'],arm)
        _, compiled = session.context(NativeState(items=items))
        delivered = {(x['expert_id'],x['evidence_id']) for x in session.last_transport['presented']}
        items = tuple(e for e in items if (e.expert_id,e.evidence_id) in delivered)
        out = {'id':row['id'],'identity':identity,'complete':False,'arms':{},'calls':[],'decisions':[],
               'delivered_packets':len(items)}

        def generate(prompt, length, stage, row=row, out=out):
            if not probe.context_token_budget(row['image'],prompt,length)['fits']:
                raise RuntimeError('Budget unavailable; do not drop original evidence')
            set_seed(42)
            start = time.perf_counter()
            value = probe.generate_with_usage(row['image'],prompt,length)
            out['calls'].append({'stage':stage,'seconds':time.perf_counter()-start,'usage':value})
            native.atomic_json(a.output/'progress'/(row['id']+'.json'),out)
            return value

        draft = prior['arms']['generalist']
        edit_instruction = ('\nMake only necessary factual corrections to the draft answer using the original '
            'image and the supplied observations. Preserve supported content. Observations are fallible, '
            'and their native score or measurement semantics must not be strengthened into an unsupported '
            'factual claim. Do not rewrite merely for style. Return only the final answer to the original '
            'question.\nDraft answer: ')
        if a.editors_only:
            out['arms']['anchored_blind_editor'] = generate(row['benchmark_prompt']+edit_instruction+json.dumps(draft['text']),1024,'anchored_blind_editor')
        out['arms']['anchored_editor'] = generate(compiled+edit_instruction+json.dumps(draft['text']),1024,'anchored_editor')
        current = draft
        for item in (() if a.editors_only else items):
            spec = source['registry'][item.expert_id]
            definition = {k:spec[k] for k in ('description','scope','capabilities','modalities') if k in spec}
            data = {'question':row['question'],'draft_answer':current['text'],
                    'specialist_definition':definition,'observation':compact_records((item,))[0]}
            prompt = ('Check the factual agreement between the draft answer and one specialist observation '
                'for the original image question. The observation is not a gold reference; use its native '
                'meaning and the image, without converting scores or measurements to unsupported facts. '
                'Does the observation actually disagree with the draft, agree, or fail to address it? '
                'Use UNKNOWN when the relationship cannot be established. Give a brief reason, then '
                'end with exactly one line: Decision: AGREES, Decision: DISAGREES, '
                'Decision: IRRELEVANT, or Decision: UNKNOWN.\n' + json.dumps(data,ensure_ascii=False))
            response = generate(prompt,256,'agreement:'+item.expert_id)
            label = decision(response['text'])
            out['decisions'].append({'expert_id':item.expert_id,'label':label})
            if label == 'DISAGREES':
                edit = row['benchmark_prompt']+'\nSpecialist observation: '+json.dumps(data['observation'],ensure_ascii=False)
                current = generate(edit+edit_instruction+json.dumps(current['text']),1024,'editor:'+item.expert_id)
        if not a.editors_only:
            out['arms']['rarr_agreement'] = dict(current, reused_generalist=current is draft)
        out['complete'] = True
        native.atomic_json(path,out)
        print('DONE',row['id'],out['decisions'],flush=True)


if __name__ == '__main__':
    main()
