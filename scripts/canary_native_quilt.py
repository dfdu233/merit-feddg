"""Two fixed cases through the unmodified formal MERIT all_evidence engine."""
from dataclasses import asdict
import json,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def persist(path,value):
    if path.exists():assert old.read_json(path)==value,'resume identity mismatch'
    else:old.write_new(path,value)
sys.path.insert(0,'/home/dbw/merit-feddg-pathology-quilt/scripts')
import run_pathology_quilt_formal as old
from run_quilt_worker import assert_single_gpu
from merit_feddg.capability_runtime import CapabilityRuntime,NativeSession,ValueGenerationConfig,INFERENCE_FIELDS
from merit_feddg.capability_experts import CapabilityPool
from merit_feddg.capabilities import CapabilityResult,EvidenceItem,CapabilityRequest
from merit_feddg.evidence_need import evidence_need
from merit_feddg.matched_evaluation import SharedExpertPool
from merit_feddg.generalist_factory import load_generalist

def main():
    import argparse,torch
    from transformers import set_seed
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu-uuid',required=True)
    p.add_argument('--expert-cache',type=Path)
    p.add_argument('--full',action='store_true')
    p.add_argument('--shard-count',type=int,default=1)
    p.add_argument('--shard-index',type=int,default=0)
    a=p.parse_args()
    assert 0<=a.shard_index<a.shard_count
    source=old.ROOT/'runs/pathology-quilt-formal-budgeted-v2'
    frozen=old.read_json(source/'frozen.json');old.verify(frozen)
    protocol=old.read_json(frozen['protocol'])
    case_ids=frozen['full_ids'][a.shard_index::a.shard_count] if a.full else frozen['canary_ids']
    eligible={r['id'] for r in frozen['rows']}
    # User-authorized Quilt-specific budget; do not inherit CheXagent's cap.
    native_budget=1024
    from merit_feddg.capability_study import _filter_optional_experts
    base_specs,excluded=_filter_optional_experts(protocol['config']['experts'],str(ROOT/'artifacts'))
    base_specs.pop('source_cases',None)
    specs=base_specs.copy()
    specs['quilt_pathology']={'id':'/home/dbw/merit-feddg/artifacts/models/wisdomik--Quilt-Llava-v1.5-7b',
        'factory':'native_quilt_factory:build','capabilities':['generation'],'modalities':['pathology'],
        'tasks':['open_vqa'],'scope':'histology_question_observation','requires_region':False,
        'description':'Unverified image-grounded histopathology observations',
        'factory_kwargs':{'expert_id':'quilt_pathology','scope':'histology_question_observation',
                          'output':str(a.expert_cache or a.output/'native-quilt-cache'),'gpu_uuid':a.gpu_uuid,
                          'max_new_tokens':native_budget,'cache_only':True}}
    identity=old.digest({'source':old.digest(frozen),'specs':specs,'files':{name:old.file_sha(ROOT/'scripts'/name)
        for name in ('canary_native_quilt.py','native_quilt_factory.py','native_quilt_infer.py')}})
    persist(a.output/'protocol.json',{'identity':identity,'source':old.digest(frozen),'specs':specs,
        'cases':case_ids,'mode':'all_evidence','full_evaluation':a.full,
        'shard_count':a.shard_count,'shard_index':a.shard_index,
        'quilt_expert_cap':1024,'merit_quilt_input_limit':8192})
    assert_single_gpu(torch,a.gpu_uuid);torch.set_num_threads(4)
    rows={r['id']:r for r in map(json.loads,Path(frozen['manifest']).read_text().splitlines())}
    # all_evidence acquires tools before decoding: the native prefix is empty.
    # Prefetch exact requests, then require cache-only inference in the engine.
    # No model is kept resident in this process during specialist prefetch.
    from native_quilt_factory import NativeQuiltExpert,DeferredRequest
    spec=specs['quilt_pathology']
    kwargs={**spec['factory_kwargs'],'cache_only':False}
    expert=NativeQuiltExpert(spec['id'],**kwargs)
    if a.full:expert.probe.queued_jobs=[]
    requests={}
    for key in case_ids:
        if key not in eligible:continue
        row=rows[key];descriptor={'capability':'generation','scope':spec['scope']}
        need=evidence_need(row['question'],descriptor)
        request=CapabilityRequest(sample_id=key,image=row['image'],question=row['question'],
            modality='pathology',task='open_vqa',domain='official-test',group_id=row['image_sha256'],
            capability='generation',scope=spec['scope'],query=need.query,generated_prefix='')
        requests[key]=request
        try:result=expert.infer(request)
        except DeferredRequest:continue
        persist(a.output/'prefetch'/(key+'.json'),{'request':asdict(request),'result':asdict(result)})
        print('NATIVE_PREFETCH',key,flush=True)
    if a.full:
        import subprocess
        if not (a.output/'quilt-jobs.json').exists():old.write_new(a.output/'quilt-jobs.json',expert.probe.queued_jobs)
        subprocess.run(['/home/dbw/.runtime/quilt-env/bin/python',str(ROOT/'scripts/native_quilt_infer.py'),
                        '--jobs',str(a.output/'quilt-jobs.json')],check=True)
        del expert.probe.queued_jobs
        expert.probe.cache_only=True
        for key,request in requests.items():
            if not (a.output/'prefetch'/(key+'.json')).exists():
                result=expert.infer(request)
                old.write_new(a.output/'prefetch'/(key+'.json'),{'request':asdict(request),'result':asdict(result)})
    probe=load_generalist(protocol['config']['generalist'],str(ROOT/'artifacts'))
    probe.model.eval().requires_grad_(False);probe.benchmark_single_block_context=True;probe.benchmark_evidence_reserve_tokens=64
    summary=[]
    for key in case_ids:
        saved=a.output/'cases'/(key+'.json')
        if saved.exists():
            assert old.read_json(saved)['identity']==identity
            continue
        before,path=old.cached(key);assert old.file_sha(path)==frozen['native_cache_hashes'][str(path)]
        if a.full and key not in eligible:
            old.write_new(a.output/'cases'/(key+'.json'),{'id':key,'identity':identity,
                'reused_outside_quilt_scope':True,'outputs':{'merit_quilt':before}})
            continue
        tools=[v for v in before['trace'] if v.get('event')=='tool']
        row={k:rows[key][k] for k in ('id','image','question')}
        row.update(modality=before['input_modality'],task='open_vqa',domain='official-test',
                   group_id=rows[key]['image_sha256'],image_sha256=rows[key]['image_sha256'],
                   capability='classification',role='target',domain_kind='official_dataset_split')
        assert set(row)==INFERENCE_FIELDS
        basepool=CapabilityPool(specs,str(ROOT/'artifacts'))
        class ReplayOld:
            last_origin='unknown'
            def infer(self,expert,request):
                if expert=='quilt_pathology':
                    self.last_origin='live_native_output';return basepool.infer(expert,request)
                candidates=[t for t in tools if t['expert']==expert and t['request']==asdict(request)]
                assert len(candidates)==1,f'old expert request mismatch: {expert} {asdict(request)}; no silent replay'
                trace=candidates[0]
                # Use exact native result recorded by the original execution.
                items=trace.get('native_evidence')
                if items is None:raise RuntimeError('old native result unavailable for exact request replay')
                self.last_origin='reused_compatible_native_output'
                return CapabilityResult(expert,request.capability,tuple(EvidenceItem(**v) for v in items))
        shared=SharedExpertPool(ReplayOld(),a.output/'expert-cache'/old.fingerprint(key),identity)
        cfg=ValueGenerationConfig(**before['generation_config'])
        # Parity run includes only original experts; same core, request sequence and renderer.
        outputs={}
        original_input_limit=probe.model.config.tokenizer_model_max_length
        for name,current_specs in (('parity',base_specs),('merit_quilt',specs)):
            if a.full and name=='parity' and key not in frozen['canary_ids']:continue
            probe.model.config.tokenizer_model_max_length=(original_input_limit if name=='parity' else
                min(8192,probe.model.config.max_position_embeddings-cfg.max_new_tokens))
            set_seed(42)
            session=NativeSession(probe,row['image'],rows[key]['benchmark_prompt'],row['question'],cfg)
            engine=CapabilityRuntime(session,shared,row,current_specs,cfg,None)
            outputs[name]=engine.run('all_evidence')
            if name=='parity':
                original=[t for t in before['trace'] if t.get('event')=='decode'][-1]['evidence_transport']
                replayed=[t for t in outputs[name]['trace'] if t.get('event')=='decode'][-1]['evidence_transport']
                for field in ('prompt_sha256','evidence_sha256'):
                    assert original[field]==replayed[field],f'baseline {field} drift'
                assert outputs[name]['token_ids']==before['token_ids'],'baseline token parity failure'
            replayed_calls=[t for t in outputs[name]['trace'] if t.get('event')=='tool' and t.get('expert')!='quilt_pathology']
            assert [(t['expert'],t['request']) for t in replayed_calls]==[(t['expert'],t['request']) for t in tools], 'old expert request order drift'
            assert all(t['executed'] for t in replayed_calls),'old expert replay failed'
        q=[t for t in outputs['merit_quilt']['trace'] if t.get('event')=='tool' and t.get('expert')=='quilt_pathology']
        result={'id':key,'identity':identity,'outputs':outputs}
        old.write_new(a.output/'cases'/(key+'.json'),result)
        probe.model.config.tokenizer_model_max_length=original_input_limit
        assert len(q)==1 and q[0]['executed'],'native Quilt call failed; inspect saved trace'
        summary.append({'id':key,'baseline_token_parity':True if 'parity' in outputs else None,'quilt_executed':q[0]['executed'],
            'quilt_adopted':q[0]['adopted'],'packing_preview':q[0].get('packing_preview'),
            'answer_nonempty':bool(outputs['merit_quilt']['text'].strip())})
        print('NATIVE_CANARY',key,'adopted',q[0]['adopted'],flush=True)
    assert {p.stem for p in (a.output/'cases').glob('*.json')}==set(case_ids)
    old.write_new(a.output/('summary.json' if not (a.output/'summary.json').exists() else 'summary-resumed.json'),{'identity':identity,'cases':summary,'complete':True,
        'mechanism_passed':all(v['quilt_adopted'] and v['answer_nonempty'] for v in summary)})

if __name__=='__main__':main()
