"""Fixed read-only reuse of full-test caches; no references enter generation.

This is a tiny segmentation-interface probe, not a replacement full evaluation.
"""
import argparse
import copy
import json
from pathlib import Path
from time import perf_counter

import torch

from merit_feddg.capabilities import EvidenceItem
from merit_feddg.capability_runtime import NativeSession, NativeState, ValueGenerationConfig
from merit_feddg.generalist_factory import load_generalist
from merit_feddg.matched_evaluation import generation_prompt
from merit_feddg.open_study import fingerprint
from merit_feddg.soft_guidance import decode

SOURCES = {
    'vqarad': ('matched-verified-packets-anchor/72f36cd3907456c90166699cd0a7fe007de3eff937f6696e04f41a0dae5e5082',
               'vqarad-official-full-test'),
    'slake': ('slake-matched-verified-packets-anchor/a57a640a1eb9e395da0523c9e999f3b28e67ece1edfc4c5861e9883f0855e465',
              'slake-official-full-test')}


def save(path, obj):
    with path.open('x') as stream:
        json.dump(obj, stream, indent=2, ensure_ascii=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='runs/soft-guidance-pilot-v1')
    parser.add_argument('--ids',help='Offline preselected ID lists only, not scores/references')
    args=parser.parse_args()
    chosen=None if args.ids is None else json.loads(Path(args.ids).read_text())
    if chosen is not None and (set(chosen)!=set(SOURCES) or any(
            not isinstance(v,list) or any(not isinstance(k,str) for k in v) for v in chosen.values())):
        raise ValueError('selection must contain only dataset -> ID lists')
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    options = json.loads(Path('configs/soft_guidance_pilot.json').read_text())
    if chosen is not None:
        options['selection']='offline historical-harm IDs; diagnostic only; no parameter selection'
        options['selected_ids']=chosen
    if (root/'frozen.json').exists():
        raise RuntimeError('use a new output directory; do not overwrite the fixed probe')
    save(root/'frozen.json', options)
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError('authorized CUDA device unavailable')
    torch.ones(1,device='cuda:0').sum().item()
    probe, loaded_spec = None, None
    for dataset, (source, data) in SOURCES.items():
        base = Path('/home/dbw/merit-feddg/runs')/source
        manifest = Path('/home/dbw/merit-feddg/runs')/data/'manifest.jsonl'
        rows = [json.loads(s) for s in manifest.read_text().splitlines()]
        protocol = json.loads((base/'protocol.json').read_text())
        assert protocol['shards_complete'] is True and protocol['n'] == len(rows)
        selected = []
        for row in rows:
            if chosen is not None and row['id'] not in chosen[dataset]:
                continue
            cache = base/'case-cache/compact_rows'/f"{fingerprint(row['id'])}.json"
            entry = json.loads(cache.read_text())
            assert entry['identity'] == protocol['identity']
            historical = entry['output']
            if any(i['capability']=='segmentation' for i in historical['evidence']):
                selected.append((row, historical, cache))
            if len(selected) == options['cases_per_dataset']:
                break
        if chosen is not None and {r['id'] for r,_,_ in selected} != set(chosen[dataset]):
            raise ValueError('selected IDs not covered exactly by cached segmentation cases')
        save(root/f'{dataset}-selection.json', {'complete_manifest_size': len(rows),
            'ids': [r['id'] for r,_,_ in selected], 'base_identity': protocol['identity'],
            'selection_uses_historical_scores': chosen is not None,
            'scores_or_references_loaded_by_generation': False, 'source': str(base), 'manifest': str(manifest)})
        spec = protocol['config']['generalist']
        if probe is None:
            started=perf_counter()
            probe=load_generalist(spec,'artifacts')
            probe.model.eval().requires_grad_(False)
            loaded_spec=copy.deepcopy(spec)
            save(root/'model.json', {'spec':spec,'load_seconds':perf_counter()-started})
        if loaded_spec != spec:
            raise RuntimeError('dataset generalists differ; cannot silently share model')
        generalists=json.loads((base/'generalist.json').read_text())
        for row,historical,cache in selected:
            print('START',dataset,row['id'],flush=True)
            cfg=ValueGenerationConfig(**historical['generation_config'])
            assert not cfg.semantic_spatial and cfg.vector_gate=='off'
            prompt=generation_prompt(row,protocol['config'])
            native=tuple(EvidenceItem(**i) for i in historical['evidence'])
            text=NativeSession(probe,row['image'],prompt,row['question'],cfg)
            with torch.inference_mode():
                replay=text.propose(NativeState(items=native),cfg.max_new_tokens)
            parity=list(replay.tokens)==historical['token_ids'] and text.decode(replay.tokens).strip()==historical['text']
            record={'dataset':dataset,'id':row['id'],'source_cache':str(cache),
                    'parity':parity,'full_manifest_complete':False,'alpha':options['alpha'],
                    'text_transport':text.last_transport,
                    'generalist_cached':{'text':generalists[row['id']]['text'],'token_ids':generalists[row['id']]['token_ids']},
                    'compact_rows_cached':{'text':historical['text'],'token_ids':historical['token_ids']},
                    'compact_rows_replay':{'text':text.decode(replay.tokens).strip(),'token_ids':list(replay.tokens)}}
            if not parity:
                save(root/f"{row['id']}.json",record)
                raise RuntimeError('exact compact parity failed; STOP without evaluating guidance')
            visible={(i['expert_id'],i['evidence_id']) for i in text.last_transport['presented']}
            presented=tuple(i for i in native if (i.expert_id,i.evidence_id) in visible)
            masks=tuple(i for i in presented if i.capability=='segmentation')
            other=tuple(i for i in presented if i.capability!='segmentation')
            # Do not introduce formerly omitted non-spatial evidence after removing masks.
            base_session=NativeSession(probe,row['image'],prompt,row['question'],cfg)
            image,base_prompt=base_session.context(NativeState(items=other))
            plain=probe.new_answer_session(image,base_prompt)
            spatial=probe.new_tensor_answer_session(image,base_prompt,masks,question=row['question'],weighting='equal')
            record['spatial_records']=len(spatial.tensor_packet)
            record['spatial_rejected']=list(spatial.tensor_packet.rejected)
            record['kept_other_experts']=[i.expert_id for i in other]
            record['mask_experts']=[i.expert_id for i in masks]
            if not len(spatial.tensor_packet):
                record['unavailable']='no_presented_usable_segmentation'
                save(root/f"{row['id']}.json",record)
                print('UNAVAILABLE',row['id'],flush=True)
                continue
            eos=probe.tokenizer.eos_token_id
            eos_ids=set(eos if isinstance(eos,list) else [eos])
            record['other_text_only']=decode(plain,None,alpha=0,max_tokens=cfg.max_new_tokens,eos_ids=eos_ids)
            # Alpha=0 must also equal production decoding with these exact inputs.
            direct=plain.propose((),count=1,length=cfg.max_new_tokens)[0]
            if record['other_text_only']['token_ids'] != list(direct.tokens):
                raise RuntimeError('zero-guidance parity failure')
            record['zero_parity']=True
            record['text_soft']=decode(plain,text._session,alpha=options['alpha'],max_tokens=cfg.max_new_tokens,eos_ids=eos_ids)
            record['native_spatial_soft']=decode(plain,spatial,alpha=options['alpha'],max_tokens=cfg.max_new_tokens,eos_ids=eos_ids)
            record['native_operator_audit']=dict(probe.tensor_bridge.last_audit)
            save(root/f"{row['id']}.json",record)
            print('DONE',dataset,row['id'],'native_changed',record['native_spatial_soft']['token_ids']!=historical['token_ids'],flush=True)
    save(root/'finished.json',{'pilot_finished':True,'full_dataset_complete':False})


if __name__=='__main__':
    main()
