"""Full matching TRAIN comparison of isolated categorical packet admission.

Answer generation is unchanged. Clinical empty outputs remain empty. No reference
or previous answer is visible to the gate. Source/model/parity failures stay fatal.
"""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from run_anchored_revision_train import GPUS, ROOT, read, resources, sha

from merit_feddg.context_admission import POLICIES, admission_prompt, admitted_view
from merit_feddg.open_study import atomic_json, fingerprint

ARMS = ('generalist', 'compact', 'relevance', 'scope', 'scope_complement')


def prepare(root):
    previous, rows, compact, baseline, source = resources()
    cfg = {'schema': 'context-admission-train-v1', 'source': previous,
           'selected': list(rows), 'n': len(rows), 'arms': ARMS,
           'code': {p: sha(ROOT / p) for p in ('scripts/run_context_admission_train.py',
                                            'merit_feddg/context_admission.py')},
           'gate': 'same frozen model; separate KV context; packet-level categorical classification',
           'choices': {'relevance': ['A', 'C', 'D'], 'scope': ['A', 'B', 'C', 'D']},
           'gate_max_tokens': 8, 'scope_admission': 'A only; B/C/D withheld, not certified wrong',
           'answer_context': 'original question plus retained original packets; no gate labels/rationale',
           'universe': 'only packets actually delivered in historical compact context',
           'no_threshold': True, 'no_training': True, 'no_reference_at_inference': True,
           'gate_failure': 'explicit unknown and reason; retained baseline is not gate success',
           'generalist_reuse': 'exact complete same-manifest cache; source and model pinned',
           'train_test_image_overlap': 'Official full TRAIN includes images shared with test; no test labels used',
           'scientific_claim': 'engineering comparison, not a novel or validated correctness gate'}
    cfg = json.loads(json.dumps(cfg))
    root.mkdir(parents=True, exist_ok=True)
    frozen = root / 'frozen.json'
    if frozen.exists() and read(frozen) != cfg:
        raise ValueError('Frozen source/config/model changed; do not mix results')
    if not frozen.exists():
        atomic_json(frozen, cfg)
    return cfg, rows, compact, baseline, source


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--check-only', action='store_true')
    p.add_argument('--merge-only', action='store_true')
    p.add_argument('--canary', action='store_true')
    p.add_argument('--gpu-uuid', choices=sorted(GPUS))
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--shard-count', type=int, default=1)
    a = p.parse_args()
    cfg, rows, compact, baseline, source = prepare(a.output)
    identity = fingerprint(cfg)
    print('PREFLIGHT', identity, cfg['n'], flush=True)
    if a.check_only:
        return
    if a.merge_only:
        paths = {p.stem: p for p in (a.output / 'cases').glob('*.json')}
        if set(paths) != set(rows):
            raise ValueError('Exact full TRAIN coverage required, no missing rows filled')
        for key, path in paths.items():
            r = read(path)
            if r['identity'] != identity or not r['complete'] or set(r['arms']) != set(ARMS):
                raise ValueError('Incomplete or incompatible case')
        atomic_json(a.output / 'complete.json', {'identity':identity,'n':len(rows),
                    'shards_complete':True,'quality_is_separate':True})
        return
    if not 0 <= a.shard_index < a.shard_count:
        raise ValueError('Invalid shard')
    device = subprocess.check_output(['nvidia-smi', '-i', '0', '--query-gpu=uuid',
        '--format=csv,noheader'], text=True, timeout=10).strip()
    free = int(subprocess.check_output(['nvidia-smi', '-i', '0', '--query-gpu=memory.free',
        '--format=csv,noheader,nounits'], text=True, timeout=10).strip())
    if device != a.gpu_uuid or os.environ.get('CUDA_VISIBLE_DEVICES') != '0' or free < 22000:
        raise RuntimeError('GPU mapping/headroom failed; existing jobs not interrupted')
    import torch
    from transformers import set_seed

    from merit_feddg.capabilities import EvidenceItem
    from merit_feddg.capability_runtime import NativeSession, NativeState, ValueGenerationConfig
    from merit_feddg.generalist_factory import load_generalist
    from merit_feddg.matched_evaluation import generation_prompt
    from merit_feddg.open_data import pixel_digest
    torch.set_num_threads(4)
    torch.ones(1, device='cuda:0').sum().item()
    start = time.perf_counter()
    probe = load_generalist(source['config']['generalist'], str(ROOT / 'artifacts'))
    probe.model.eval().requires_grad_(False)
    atomic_json(a.output / f'load-{time.time_ns()}.json', {
        'identity':identity,'seconds':time.perf_counter()-start,'gpu_uuid':device})
    canaries = []
    for name in ('chexagent_description', 'cxr_anatomy', 'biomed_anatomy'):
        canaries.extend([k for k in rows if any(e['expert_id'] == name
                         for e in compact[k]['evidence'])][:2])
    canaries = set(canaries)
    for index, (key, row) in enumerate(rows.items()):
        if a.canary and key not in canaries:
            continue
        if index % a.shard_count != a.shard_index:
            continue
        path = a.output / 'cases' / (key + '.json')
        if path.exists():
            prior = read(path)
            if prior['identity'] != identity:
                raise ValueError('Case identity changed')
            if prior['complete']:
                continue
        if pixel_digest(row['image']) != row['image_sha256']:
            raise ValueError('Image changed')
        old, base = compact[key], baseline[key]
        gen = ValueGenerationConfig(**old['generation_config'])
        if gen.semantic_spatial or gen.visual_views:
            raise ValueError('Frozen source uses text evidence, not spatial intervention')
        native = tuple(EvidenceItem(**e) for e in old['evidence'])
        prompt = generation_prompt(row, source['config'])
        clean = NativeSession(probe, row['image'], prompt, row['question'], gen)
        _, full_prompt = clean.context(NativeState(items=native))
        delivered = {(x['expert_id'], x['evidence_id']) for x in clean.last_transport['presented']}
        items = tuple(e for e in native if (e.expert_id, e.evidence_id) in delivered)
        reduced = NativeSession(probe, row['image'], prompt, row['question'], gen)
        _, replay_prompt = reduced.context(NativeState(items=items))
        if replay_prompt != full_prompt:
            raise RuntimeError('Restricting to delivered universe changed historical prompt')
        out = {'id':key,'identity':identity,'complete':False,'gpu_uuid':device,
               'arms':{},'gates':{},'delivered_packets':len(items),
               'native_evidence_sha256':fingerprint(old['evidence'])}
        for arm, cached in (('generalist',base),('compact',old)):
            out['arms'][arm] = {'text':cached['text'],'token_ids':cached['token_ids'],
                'new_calls':0,'new_seconds':0,'inherited_seconds':cached['seconds'],
                'reuse':'same_full_train_historical','status':'generated' if cached['text'] else 'empty'}
        if a.canary:
            out['parity'] = {}
            for arm, text_prompt, cached in (('generalist',prompt,base),('compact',full_prompt,old)):
                set_seed(42)
                start = time.perf_counter()
                block = probe.new_answer_session(row['image'], text_prompt).propose((),1,gen.max_new_tokens)[0]
                parity = list(block.tokens) == cached['token_ids'] and clean.decode(block.tokens).strip() == cached['text']
                out['parity'][arm] = {'passed':parity,'calls':1,'seconds':time.perf_counter()-start}
                if not parity:
                    atomic_json(path,out)
                    raise RuntimeError('Historical canary parity failed')
        for policy in POLICIES:
            judgments = []
            for item in items:
                spec = source['config']['experts'][item.expert_id]
                question = admission_prompt(row['question'],item,spec,policy)
                budget = probe.context_token_budget(row['image'],question,cfg['gate_max_tokens'])
                if not budget['fits']:
                    verdict = {'label':'D','reason':'gate_context_unavailable','calls':0,'seconds':0}
                else:
                    set_seed(42)
                    start = time.perf_counter()
                    response = probe.generate_with_usage(row['image'],question,
                        max_new_tokens=cfg['gate_max_tokens'],allowed_texts=cfg['choices'][policy])
                    verdict = {'label':response['text'],'reason':'model_categorical',
                               'calls':1,'seconds':time.perf_counter()-start,'usage':response}
                judgments.append({**verdict,'expert_id':item.expert_id,'evidence_id':item.evidence_id,
                                  'budget':budget,'prompt_sha256':fingerprint(question)})
            out['gates'][policy] = judgments
        for arm in ARMS[2:]:
            policy = 'scope' if arm == 'scope_complement' else arm
            kept = admitted_view(items, [v['label'] for v in out['gates'][policy]],
                                 complement=arm == 'scope_complement')
            selected = [(e.expert_id,e.evidence_id) for e in kept]
            if not kept:
                result = {**out['arms']['generalist'],'reuse':'no_admitted_packets',
                          'gate_success_claim':False}
            elif kept == items:
                result = {**out['arms']['compact'],'reuse':'all_original_packets'}
            else:
                session = NativeSession(probe,row['image'],prompt,row['question'],gen)
                image, compiled = session.context(NativeState(items=kept))
                actual = [(x['expert_id'],x['evidence_id']) for x in session.last_transport['presented']]
                if actual != selected:
                    raise RuntimeError('Allowed subset was not delivered exactly')
                set_seed(42)
                start = time.perf_counter()
                block = probe.new_answer_session(image,compiled).propose((),1,gen.max_new_tokens)[0]
                text = session.decode(block.tokens).strip()
                result = {'text':text,'token_ids':list(block.tokens),'new_calls':1,
                    'new_seconds':time.perf_counter()-start,'status':'generated' if text else 'empty',
                    'transport':session.last_transport,'reuse':None,'finished':block.finished}
            out['arms'][arm] = {**result,'kept':selected}
        out['complete'] = True
        atomic_json(path,out)
        print('DONE',index+1,key,'packets',len(items),
              {p:[v['label'] for v in j] for p,j in out['gates'].items()},flush=True)
    print('SCHEDULE_COMPLETE',a.shard_index,a.shard_count,'canary',a.canary,flush=True)


if __name__ == '__main__':
    main()
