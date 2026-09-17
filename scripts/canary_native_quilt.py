"""Two fixed cases through the unmodified formal MERIT all_evidence engine."""
from dataclasses import asdict
import json,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
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
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu-uuid',required=True);a=p.parse_args()
    source=old.ROOT/'runs/pathology-quilt-formal-budgeted-v2'
    frozen=old.read_json(source/'frozen.json');old.verify(frozen)
    protocol=old.read_json(frozen['protocol'])
    native_budget=protocol['config']['experts']['chexagent_description']['factory_kwargs']['max_new_tokens']
    specs=protocol['config']['experts'].copy()
    specs['quilt_pathology']={'id':'/home/dbw/merit-feddg/artifacts/models/wisdomik--Quilt-Llava-v1.5-7b',
        'factory':'native_quilt_factory:build','capabilities':['generation'],'modalities':['pathology'],
        'tasks':['open_vqa'],'scope':'histology_question_observation','requires_region':False,
        'description':'Unverified image-grounded histopathology observations',
        'factory_kwargs':{'expert_id':'quilt_pathology','scope':'histology_question_observation',
                          'output':str(a.output/'native-quilt-cache'),'gpu_uuid':a.gpu_uuid,
                          'max_new_tokens':native_budget,'cache_only':True}}
    identity=old.digest({'source':old.digest(frozen),'specs':specs,'files':{name:old.file_sha(ROOT/'scripts'/name)
        for name in ('canary_native_quilt.py','native_quilt_factory.py','native_quilt_infer.py')}})
    old.write_new(a.output/'protocol.json',{'identity':identity,'source':old.digest(frozen),'specs':specs,
        'cases':frozen['canary_ids'],'mode':'all_evidence','full_evaluation':False})
    assert_single_gpu(torch,a.gpu_uuid);torch.set_num_threads(4)
    rows={r['id']:r for r in map(json.loads,Path(frozen['manifest']).read_text().splitlines())}
    # all_evidence acquires tools before decoding: the native prefix is empty.
    # Prefetch exact requests, then require cache-only inference in the engine.
    # No model is kept resident in this process during specialist prefetch.
    from native_quilt_factory import NativeQuiltExpert
    spec=specs['quilt_pathology']
    kwargs={**spec['factory_kwargs'],'cache_only':False}
    expert=NativeQuiltExpert(spec['id'],**kwargs)
    for key in frozen['canary_ids']:
        row=rows[key];descriptor={'capability':'generation','scope':spec['scope']}
        need=evidence_need(row['question'],descriptor)
        request=CapabilityRequest(sample_id=key,image=row['image'],question=row['question'],
            modality='pathology',task='open_vqa',domain='official-test',group_id=row['image_sha256'],
            capability='generation',scope=spec['scope'],query=need.query,generated_prefix='')
        result=expert.infer(request)
        old.write_new(a.output/'prefetch'/(key+'.json'),{'request':asdict(request),'result':asdict(result)})
        print('NATIVE_PREFETCH',key,flush=True)
    probe=load_generalist(protocol['config']['generalist'],str(ROOT/'artifacts'))
    probe.model.eval().requires_grad_(False);probe.benchmark_single_block_context=True;probe.benchmark_evidence_reserve_tokens=64
    summary=[]
    for key in frozen['canary_ids']:
        before,path=old.cached(key);assert old.file_sha(path)==frozen['native_cache_hashes'][str(path)]
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
                assert len(candidates)==1,'old expert request mismatch; no silent replay'
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
        for name,current_specs in (('parity',protocol['config']['experts']),('merit_quilt',specs)):
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
        assert len(q)==1 and q[0]['executed'],'native Quilt call failed; inspect saved trace'
        summary.append({'id':key,'baseline_token_parity':True,'quilt_executed':q[0]['executed'],
            'quilt_adopted':q[0]['adopted'],'packing_preview':q[0].get('packing_preview'),
            'answer_nonempty':bool(outputs['merit_quilt']['text'].strip())})
        print('NATIVE_CANARY',key,'adopted',q[0]['adopted'],flush=True)
    old.write_new(a.output/'summary.json',{'identity':identity,'cases':summary,'complete':True,
        'mechanism_passed':all(v['quilt_adopted'] and v['answer_nonempty'] for v in summary)})

if __name__=='__main__':main()
