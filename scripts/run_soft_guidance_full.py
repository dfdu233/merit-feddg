"""Full manifests, frozen strength, fresh matched controls; reference-free inference.

Historical output drift is audited, not called parity. The separate pilot's
strict historical-parity guard is unchanged. Runtime failures stop the run.
"""
import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import torch
from run_soft_guidance_pilot import SOURCES

from merit_feddg.capabilities import EvidenceItem
from merit_feddg.capability_runtime import NativeSession, NativeState, ValueGenerationConfig
from merit_feddg.generalist_factory import generalist_provenance, load_generalist
from merit_feddg.matched_evaluation import generation_prompt
from merit_feddg.open_study import atomic_json, fingerprint
from merit_feddg.soft_guidance import decode

ARMS = ('historical_generalist', 'historical_compact', 'generalist_matched',
        'compact_matched', 'other_text_only', 'text_soft', 'native_spatial_soft')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def output(block, seconds):
    return {'text': block.text.strip(), 'token_ids': list(block.tokens), 'seconds': seconds}


def fallback(value, reason):
    return {'text':value['text'], 'token_ids':list(value['token_ids']), 'seconds':0.,
            'reused_same_case_control':True, 'guidance_applied':False,
            'reason':reason, 'base_score_calls':0, 'conditioned_score_calls':0}


def full_case(probe, row, historical, generalist, protocol, alpha):
    started=perf_counter()
    cfg=ValueGenerationConfig(**historical['generation_config'])
    if cfg.semantic_spatial or cfg.vector_gate!='off' or cfg.evidence_style!='semantic':
        raise ValueError('expected original ungated semantic compact control')
    prompt=generation_prompt(row,protocol['config'])
    native=tuple(EvidenceItem(**i) for i in historical['evidence'])
    eos=probe.tokenizer.eos_token_id
    eos_ids=set(eos if isinstance(eos,list) else [eos])
    text=NativeSession(probe,row['image'],prompt,row['question'],cfg)
    before=perf_counter()
    with torch.inference_mode():
        replay=text.propose(NativeState(items=native),cfg.max_new_tokens)
    compact=output(replay,perf_counter()-before)
    old_transport=next((e['evidence_transport'] for e in reversed(historical.get('trace',[]))
                        if e.get('event')=='decode' and e.get('evidence_transport')),None)
    if old_transport:
        for key in ('prompt_sha256','evidence_sha256','presented','omitted','context'):
            if key in old_transport and old_transport[key]!=text.last_transport.get(key):
                raise RuntimeError('historical evidence/prompt delivery changed: '+key)
    base_generalist=probe.new_answer_session(row['image'],prompt)
    before=perf_counter()
    with torch.inference_mode():
        block=base_generalist.propose((),count=1,length=cfg.max_new_tokens)[0]
    fresh_generalist=output(block,perf_counter()-before)
    arms={'historical_generalist':{k:generalist[k] for k in ('text','token_ids','seconds')},
          'historical_compact':{k:historical[k] for k in ('text','token_ids','seconds')},
          'generalist_matched':fresh_generalist,'compact_matched':compact}
    result={'id':row['id'],'arms':arms,'generation_config':asdict(cfg),
            'historical_expert_outputs_reused':True,'new_expert_calls':0,
            'text_transport':text.last_transport,
            'historical_transport_available':old_transport is not None,
            'historical_compact_token_parity':compact['token_ids']==historical['token_ids'],
            'historical_generalist_token_parity':fresh_generalist['token_ids']==generalist['token_ids'],
            'historical_compact_text_parity':compact['text']==historical['text'],
            'historical_generalist_text_parity':fresh_generalist['text']==generalist['text']}
    visible={(i['expert_id'],i['evidence_id']) for i in text.last_transport['presented']}
    presented=tuple(i for i in native if (i.expert_id,i.evidence_id) in visible)
    masks=tuple(i for i in presented if i.capability=='segmentation')
    other=tuple(i for i in presented if i.capability!='segmentation')
    result['segmentation_experts']=[i.expert_id for i in masks]
    result['retained_other_experts']=[i.expert_id for i in other]
    packet=probe.tensor_packet(masks,row['image'],question=row['question'],weighting='equal')
    result['spatial_records']=len(packet)
    result['spatial_rejected']=list(packet.rejected)
    result['applicable']=bool(len(packet))
    if not len(packet):
        reason='no_presented_usable_segmentation; frozen incumbent fallback'
        for arm in ('other_text_only','text_soft','native_spatial_soft'):
            arms[arm]=fallback(compact,reason)
        result['zero_check']='not_applicable_no_operator'
    else:
        context=NativeSession(probe,row['image'],prompt,row['question'],cfg)
        image,base_prompt=context.context(NativeState(items=other))
        plain=probe.new_answer_session(image,base_prompt)
        spatial=probe.new_tensor_answer_session(image,base_prompt,masks,
                                               question=row['question'],weighting='equal')
        arms['other_text_only']=decode(plain,None,alpha=0,max_tokens=cfg.max_new_tokens,eos_ids=eos_ids)
        before=perf_counter()
        with torch.inference_mode():
            direct=plain.propose((),count=1,length=cfg.max_new_tokens)[0]
        result['zero_verification_seconds']=perf_counter()-before
        if arms['other_text_only']['token_ids']!=list(direct.tokens):
            raise RuntimeError('current-control zero-guidance parity failure')
        result['zero_check']='exact_current_control_pass'
        arms['text_soft']=decode(plain,text._session,alpha=alpha,max_tokens=cfg.max_new_tokens,eos_ids=eos_ids)
        arms['native_spatial_soft']=decode(plain,spatial,alpha=alpha,max_tokens=cfg.max_new_tokens,eos_ids=eos_ids)
        result['native_operator_audit']=dict(probe.tensor_bridge.last_audit)
        for arm in ('text_soft','native_spatial_soft'):
            arms[arm]['guidance_applied']=True
    if set(arms)!=set(ARMS) or any(not a['text'] or not a['token_ids'] for a in arms.values()):
        raise RuntimeError('missing/empty arm; never impute engineering failures')
    result['wall_seconds']=perf_counter()-started
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='runs/soft-guidance-full-v1')
    parser.add_argument('--max-cases',type=int,default=0,help='Scheduling stop only, same full manifests')
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    if args.max_cases<0:
        raise ValueError('max-cases must be nonnegative')
    datasets={}
    for name,(source,data) in SOURCES.items():
        base=Path('/home/dbw/merit-feddg/runs')/source
        manifest=Path('/home/dbw/merit-feddg/runs')/data/'manifest.jsonl'
        rows=[json.loads(s) for s in manifest.read_text().splitlines()]
        protocol=json.loads((base/'protocol.json').read_text())
        if protocol['shards_complete'] is not True or protocol['n']!=len(rows) or len({r['id'] for r in rows})!=len(rows):
            raise ValueError('complete aligned source protocol required')
        datasets[name]={'base':str(base),'manifest':str(manifest),'manifest_sha256':sha(manifest),
                        'protocol':protocol,'rows':rows}
    spec=datasets['vqarad']['protocol']['config']['generalist']
    if datasets['slake']['protocol']['config']['generalist']!=spec:
        raise ValueError('generalist mismatch')
    config={'schema':'soft-guidance-full-matched-v1','alpha':.5,'arms':list(ARMS),
            'comparison':'fresh controls primary; historical drift explicitly audited',
            'scope':'segmentation only; other presented evidence retained as compact text',
            'unavailable':'all intervention arms reuse fresh compact; not successful guidance',
            'fit_or_test_parameter_selection':False,'datasets':datasets,
            'model':generalist_provenance(spec,'artifacts'),
            'source':{str(p):sha(p) for p in sorted(Path('merit_feddg').rglob('*.py'))},
            'runner_sha256':sha(__file__),
            'runtime':{'torch':torch.__version__,'cuda':torch.version.cuda,'threads':4}}
    config=json.loads(json.dumps(config))
    identity=fingerprint(config)
    root=Path(args.output)/identity
    root.mkdir(parents=True,exist_ok=True)
    if (root/'frozen.json').exists():
        if json.loads((root/'frozen.json').read_text())!=config:
            raise RuntimeError('frozen identity mismatch')
    else:
        atomic_json(root/'frozen.json',config)
    print('ROOT',root.resolve(),flush=True)
    if args.check_only:
        return
    import fcntl
    with (root/'.worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        torch.set_num_threads(4)
        if not torch.cuda.is_available():
            raise RuntimeError('authorized CUDA device unavailable')
        torch.ones(1,device='cuda:0').sum().item()
        before=perf_counter()
        probe=load_generalist(spec,'artifacts')
        probe.model.eval().requires_grad_(False)
        # Separate startup records preserve restart overhead instead of overwriting it.
        atomic_json(root/f'load-{len(list(root.glob("load-*.json"))):03d}.json',
                    {'seconds':perf_counter()-before,'device':torch.cuda.get_device_name(0)})
        completed=0
        for dataset,data in datasets.items():
            dest=root/dataset
            dest.mkdir(exist_ok=True)
            base=Path(data['base'])
            generalists=json.loads((base/'generalist.json').read_text())
            if set(generalists)!={r['id'] for r in data['rows']}:
                raise ValueError('historical generalist ID set mismatch')
            for index,row in enumerate(data['rows']):
                target=dest/f"{row['id']}.json"
                if target.exists():
                    cached=json.loads(target.read_text())
                    if cached['identity']!=identity or cached['id']!=row['id'] or set(cached['arms'])!=set(ARMS):
                        raise RuntimeError('bad resume record')
                    continue
                if args.max_cases and completed>=args.max_cases:
                    print('SCHEDULING STOP, full manifests incomplete',flush=True)
                    return
                entry=json.loads((base/'case-cache/compact_rows'/f"{fingerprint(row['id'])}.json").read_text())
                if entry['identity']!=data['protocol']['identity']:
                    raise RuntimeError('source expert cache identity mismatch')
                try:
                    result=full_case(probe,row,entry['output'],generalists[row['id']],data['protocol'],.5)
                except Exception as exc:
                    atomic_json(root/f'failure-{row["id"]}-{len(list(root.glob("failure-*.json"))):03d}.json',
                                {'id':row['id'],'error_type':type(exc).__name__,'error':str(exc)})
                    raise
                atomic_json(target,{'identity':identity,'dataset':dataset,**result})
                completed+=1
                print('DONE',dataset,index+1,len(data['rows']),row['id'],
                      'applicable',result['applicable'],'seconds',round(result['wall_seconds'],3),
                      'historical_compact_equal',result['historical_compact_token_parity'],flush=True)
            expected={r['id'] for r in data['rows']}
            if {p.stem for p in dest.glob('*.json')}!=expected:
                raise RuntimeError('full output ID set mismatch')
            atomic_json(root/f'{dataset}-complete.json',{'n':len(expected),'identity':identity})
        atomic_json(root/'complete.json',{'identity':identity,'datasets':{k:len(v['rows']) for k,v in datasets.items()},
                                         'full_dataset_complete':True})


if __name__=='__main__':
    main()
