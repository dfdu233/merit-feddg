"""Refresh incompatible native TEST controls with the same frozen model/prompts.

No reference answers are read. Original evidence, transport, prompt, model and
generation settings remain fixed. Historical answers are retained in the source.
Run only after explicit authorization for the additional compatibility repair.
"""
import argparse
from copy import deepcopy
import importlib
import inspect
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from merit_feddg.algorithm_chain.storage import read, write, digest, fingerprint, pin_files, lock
from merit_feddg.algorithm_chain.native import check_device, ForwardBudget, legacy_runner, verify_transport
from prepare_algorithm_test import write_case


def refresh(job_path, output, remaining_forwards):
    job = read(job_path)
    if job['node'] != 'TEST' or job['candidate'] != dict(algorithm='project',control='layout_average',layer=None):
        raise ValueError('Only the fixed official TEST candidate input is supported')
    if [(i['dataset'],len(i['records']),i['split']) for i in job['inventories']] != [
            ('vqa_rad',451,'official_test'),('slake',2094,'official_test')]:
        raise ValueError('Keep all official TEST cases')
    with lock(output/'.refresh.lock'):
        code_paths = [Path(__file__).resolve(), ROOT/'scripts/prepare_algorithm_test.py',
                      ROOT/'scripts/run_huatuo_pathway.py',*sorted((ROOT/'merit_feddg').rglob('*.py'))]
        spec = dict(input_job_sha256=digest(job_path),remaining_forwards=remaining_forwards,
                    runtime=job['runtime'],generation=job['generation'],
                    code_pins={str(p):digest(p) for p in code_paths},
                    source_protocols={i['cohort']:i['protocol_sha256'] for i in job['inventories']},
                    reason='Historical native cache failed real current native/off token parity; regenerate both native controls without inspecting scores')
        identity=fingerprint(spec)
        if (output/'refresh-plan.json').exists():
            if read(output/'refresh-plan.json') != dict(identity=identity,**spec):
                raise ValueError('Refresh resume identity changed')
        else:
            write(output/'refresh-plan.json',dict(identity=identity,**spec))
        runtime=job['runtime']
        pin_files(runtime['pins'])
        check_device(runtime['gpu_uuid'],runtime.get('allowed_display_contexts',()))
        for root in runtime.get('import_roots',[]): sys.path.append(root)
        import torch
        from merit_feddg.huatuo_pathway import native_inputs,prepare_pair,_sync
        from merit_feddg.algorithm_chain.decoding import DecodePolicy,generate
        factory_module,factory_name=runtime['factory'].split(':',1)
        factory=getattr(importlib.import_module(factory_module),factory_name)
        if str(Path(inspect.getfile(factory)).resolve()) not in runtime['pins']:
            raise ValueError('Wrong adapter import')
        started=time.perf_counter();adapter=factory(**runtime.get('kwargs',{}))
        model=adapter.model.eval().requires_grad_(False)
        legacy=legacy_runner();legacy.check_native_generation_defaults(model.generation_config)
        if Path(model.config._name_or_path).resolve()!=Path(runtime['checkpoint_dir']).resolve():
            raise ValueError('Wrong checkpoint')
        _sync(model.device)
        write(output/('load-'+str(time.time_ns())+'.json'),dict(seconds=time.perf_counter()-started))
        inventories=[]
        with ForwardBudget(model,remaining_forwards,output/'budget.json') as budget:
            for inv in job['inventories']:
                source=Path(inv['source_run']);original=read(source/'protocol.json')
                if digest(source/'protocol.json')!=inv['protocol_sha256']:
                    raise ValueError('Source protocol changed')
                target=output/inv['dataset'];settings=job['generation'][inv['cohort']]
                legacy.validate_generation_kwargs(settings,original['generation_config']['max_new_tokens'])
                new_spec=dict(original);new_spec.pop('identity')
                new_spec.update(native_controls_refreshed=True,refresh_identity=identity,
                    old_source_identity=original['identity'],old_protocol_sha256=inv['protocol_sha256'],
                    old_source_directory=str(source),reused_only=False,
                    new_model_forwards=None,forward_count_artifact=str(output/'complete.json'),
                    selection='Complete official TEST manifest order, current native controls; original expert evidence and prompts unchanged')
                new_identity=fingerprint(new_spec)
                if not (target/'protocol.json').exists():
                    write(target/'protocol.json',dict(identity=new_identity,**new_spec))
                records=[]
                for index,(row,frozen) in enumerate(zip(original['rows'],inv['records'])):
                    path=target/'cases'/(row['id']+'.json')
                    if path.exists():
                        case=read(path)
                        if case['identity']!=new_identity or case.get('complete') is not True:
                            raise ValueError('Invalid refreshed case resume')
                    else:
                        old_path=source/'cases'/(row['id']+'.json')
                        if digest(old_path)!=frozen['case_sha256'] or digest(row['image'])!=frozen['image_sha256']:
                            raise ValueError('Original case/image changed')
                        case=read(old_path)
                        prompt,transport=legacy.restore_context(adapter,row,case,original['generation_config'],model.config.max_position_embeddings)
                        verify_transport(case,prompt,transport)
                        case.update(identity=new_identity,reused_only=False,refresh_identity=identity,
                                    original_case_sha256=frozen['case_sha256'],original_source_id=original['identity'],
                                    compact_raw_role='Original historical evidence/transport audit; refreshed answer tokens are exclusively in arms')
                        torch.cuda.reset_peak_memory_stats(model.device)
                        refreshed={};start_budget=budget.used;case_start=time.perf_counter()
                        for arm,text in [('generalist',row['benchmark_prompt']),('compact',prompt)]:
                            ids,pixels=native_inputs(adapter,row['image'],text)
                            before=budget.used;_sync(model.device);start=time.perf_counter()
                            with torch.inference_mode(),adapter._generation_last_token_logits():
                                native=model.generate(ids,images=pixels,return_dict_in_generate=True,output_scores=True,**settings)
                            count=len(native.scores);tokens=native.sequences[0,-count:].tolist() if count else []
                            _sync(model.device)
                            answer=adapter.tokenizer.decode(tokens,skip_special_tokens=True).strip()
                            if not tokens or not answer: raise ValueError('Empty refreshed native control')
                            refreshed[arm]=dict(text=answer,token_ids=tokens,seconds=time.perf_counter()-start,
                                forwards=budget.used-before,historical_token_match=tokens==case['arms'][arm]['token_ids'],
                                source='actual frozen model.generate; no label access')
                            del native,ids,pixels
                        case['arms']=refreshed
                        if index<2:
                            ref,rec=prepare_pair(adapter,row['image'],row['benchmark_prompt'],prompt)
                            bos=tuple(model._maybe_initialize_input_ids_for_generation(None,model.generation_config.bos_token_id,
                                model_kwargs={'inputs_embeds':torch.empty(1,1,model.config.hidden_size,device=model.device)})[0].tolist())
                            eos=settings['eos_token_id']
                            policy=DecodePolicy(settings['max_new_tokens'],tuple(eos) if isinstance(eos,list) else (eos,),
                                                settings['repetition_penalty'],settings['min_new_tokens'],bos)
                            checks={}
                            for arm,prepared in [('generalist',ref),('compact',rec)]:
                                off=generate(model,ref,prepared,policy,algorithm='off',context_limit=model.config.max_position_embeddings)
                                checks[arm]=off['token_ids']==refreshed[arm]['token_ids']
                            case['refresh_canary']=checks
                            if not all(checks.values()): raise ValueError('Current native/off mismatch during refresh')
                        case.update(complete=True,refresh_forwards=budget.used-start_budget,
                                    refresh_seconds=time.perf_counter()-case_start,
                                    refresh_peak_allocated_bytes=torch.cuda.max_memory_allocated(model.device))
                        write_case(path,case)
                        print('REFRESHED',inv['dataset'],index+1,len(original['rows']),flush=True)
                    records.append(dict(frozen,case_sha256=digest(path)))
                if not (target/'complete.json').exists():
                    write(target/'complete.json',dict(identity=new_identity,n=len(records),reused_only=False,
                        meaning='Complete actual native control regeneration; original experts and prompts reused exactly'))
                inventories.append(dict(inv,source_run=str(target.resolve()),source_identity=new_identity,
                                        protocol_sha256=digest(target/'protocol.json'),records=records))
            write(output/'inventory.json',inventories)
            write(output/'complete.json',dict(identity=identity,n=2545,forwards=budget.used,reference_answers_read=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--job',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--remaining-forwards',type=int,required=True)
    a=p.parse_args();refresh(a.job.resolve(),a.output.resolve(),a.remaining_forwards)
