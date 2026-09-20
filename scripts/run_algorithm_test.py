"""Explicitly authorized full TEST evaluation of the already fixed C decoder.

Reuses the unchanged chain worker and isolated scorer. No A/B/D/E transitions,
candidate selection, threshold changes or READY_FOR_SCALE certification.
"""
import argparse
from copy import deepcopy
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from merit_feddg.algorithm_chain.storage import (read, write, digest, fingerprint,
                                               pin_files, verify_plan, lock)
from merit_feddg.algorithm_chain.cli import invoke_child, validate_predictions


def freeze(inputs, predecessor, output, max_total_forwards, previous_test=None):
    if output.exists():
        raise FileExistsError('Use a new explicit TEST evaluation directory')
    prior, state = read(predecessor/'plan.json'), read(predecessor/'state.json')
    verify_plan(prior)
    if state['plan_sha'] != prior['identity'] or state['node'] != 'EXPLORATORY_COMPLETE':
        raise ValueError('Require the completed predecessor without reopening its chain')
    spent = sum(read(predecessor/h['directory']/'budget.json')['used']
                for h in state['history'] if (predecessor/h['directory']/'budget.json').exists())
    spent += sum(x['forwards'] for x in state.get('preflight_costs', []))
    if spent != state['forwards_used'] or max_total_forwards <= spent:
        raise ValueError('Invalid inherited forward ledger or remaining budget')
    inherited_test_attempts, test_lineage = 0, None
    if previous_test:
        previous, previous_state = read(previous_test/'plan.json'), read(previous_test/'state.json')
        if (fingerprint({k:v for k,v in previous.items() if k!='identity'}) != previous['identity']
                or previous_state['plan_sha'] != previous['identity']
                or previous_state['status'] != 'BLOCKED_TEST_TECHNICAL'):
            raise ValueError('Only a preserved technical failure can be repaired')
        if (previous['candidate'] != state['candidate'] or previous['runtime'] != prior['runtime']
                or previous['predecessor_plan_sha'] != prior['identity']):
            raise ValueError('TEST repair cannot change the candidate or model')
        for path, sha in prior['code_pins'].items():
            if previous['code_pins'].get(path) != sha:
                raise ValueError('TEST repair must retain the inherited scientific code')
        previous_used = read(previous_test/'budget.json')['used'] if (previous_test/'budget.json').exists() else 0
        if previous_used != previous_state['new_forwards']:
            raise ValueError('Previous TEST cost changed')
        spent = previous['inherited_forwards'] + previous_used
        inherited_test_attempts = previous_state['attempts']
        if inherited_test_attempts >= 3 or spent >= max_total_forwards:
            raise ValueError('TEST attempt/forward budget exhausted')
        test_lineage = dict(directory=str(previous_test),plan_sha=previous['identity'],
            state_sha256=digest(previous_test/'state.json'),new_forwards=previous_used,
            inherited_test_attempts=inherited_test_attempts,
            reason='Preserve nested evidence field order during cache adaptation; no decoder change')
    for entry in state['history']:
        if digest(predecessor/entry['directory']/'decision.json') != entry['report_sha']:
            raise ValueError('Historical decision changed')
    inventories = read(inputs/'inventory.json')
    if [(x['dataset'],len(x['records']),x['split']) for x in inventories] != [
            ('vqa_rad',451,'official_test'),('slake',2094,'official_test')]:
        raise ValueError('Only the two complete declared official TEST queues')
    scorer = deepcopy(prior['scorer'])
    scorer['references'] = read(inputs/'references-map.json')
    scorer['pins'].update({p:digest(p) for p in scorer['references'].values()})
    code_pins = dict(prior['code_pins'])
    code_pins.update({str(ROOT/'scripts'/name):digest(ROOT/'scripts'/name)
                     for name in ['prepare_algorithm_test.py','run_algorithm_test.py']})
    source_pins = {}
    for inv in inventories:
        source = Path(inv['source_run'])
        if digest(source/'protocol.json') != inv['protocol_sha256']:
            raise ValueError('Source adaptation changed')
        protocol = read(source/'protocol.json')
        if protocol['split'] != 'official_test' or protocol['identity'] != inv['source_identity']:
            raise ValueError('Do not relabel TEST as TRAIN')
        source_pins.update(protocol['original_pins'])
        source_pins[str(source/'protocol.json')] = inv['protocol_sha256']
        for r in inv['records']:
            source_pins[str(source/'cases'/(r['source_id']+'.json'))] = r['case_sha256']
    pin_files(source_pins)
    plan = dict(schema='merit-explicit-official-test-v1',authorization='User explicitly requested both complete official TEST datasets',
        candidate=state['candidate'],prior_scientific_verdict='fail',ready_for_scale=False,
        runtime=prior['runtime'],scorer=scorer,code_pins=code_pins,source_pins=source_pins,
        inventories=inventories,generation={i['cohort']:next(iter(prior['generation'].values())) for i in inventories},
        policy=prior['policy'],max_total_forwards=max_total_forwards,inherited_forwards=spent,
        predecessor_plan_sha=prior['identity'],predecessor_state_sha256=digest(predecessor/'state.json'),
        inherited_attempts=state['attempts'],new_evaluation_max_attempts=3,
        previous_test=test_lineage,
        interpretation='Full fixed-candidate TEST measurement after a failed development screen; not independent confirmation or retuning')
    plan['identity'] = fingerprint(plan)
    job = dict(schema='merit-explicit-official-test-job-v1',node='TEST',candidate=plan['candidate'],
        policy=plan['policy'],inventories=inventories,runtime=plan['runtime'],generation=plan['generation'],
        plan_sha=plan['identity'],remaining_forwards=max_total_forwards-spent)
    write(output/'plan.json',plan)
    write(output/'input-job.json',job)
    write(output/'scoring-config.json',scorer)
    write(output/'state.json',dict(status='FROZEN_TEST',attempts=inherited_test_attempts,inherited_forwards=spent,
                                   plan_sha=plan['identity'],job_sha=fingerprint(job),new_forwards=0))
    print('FROZEN OFFICIAL TEST',plan['identity'],flush=True)


def run(output):
    with lock(output/'.test.lock'):
        plan, state = read(output/'plan.json'), read(output/'state.json')
        if fingerprint({k:v for k,v in plan.items() if k!='identity'}) != plan['identity'] or state['plan_sha'] != plan['identity']:
            raise ValueError('TEST plan/state identity changed')
        if state['status'] in ('TEST_COMPLETE','TEST_COMPLETE_WITH_FAILURES'):
            print(state['status']); return
        for pins in [plan['code_pins'],plan['source_pins'],plan['runtime']['pins'],plan['scorer']['pins']]:
            pin_files(pins)
        job = read(output/'input-job.json')
        if (fingerprint(job) != state['job_sha'] or job['plan_sha'] != plan['identity']
                or job['remaining_forwards'] != plan['max_total_forwards']-plan['inherited_forwards']):
            raise ValueError('Job identity or cumulative budget changed')
        if state['attempts'] >= plan['new_evaluation_max_attempts']:
            raise ValueError('TEST evaluation attempt limit reached')
        state.update(status='RUNNING_TEST',attempts=state['attempts']+1)
        write(output/'state.json',state,replace=True)
        stamp=str(time.time_ns()); env=dict(os.environ)
        env.update(CUDA_VISIBLE_DEVICES=plan['runtime']['gpu_uuid'],HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='1')
        script=str(ROOT/'scripts/run_algorithm_chain.py')
        command=[sys.executable,script,'worker','--job',str(output/'input-job.json'),'--output',str(output)]
        rc=invoke_child(command,output/'inference.log',env,output/('inference-exit-'+stamp+'.json'))
        state['new_forwards']=read(output/'budget.json')['used'] if (output/'budget.json').exists() else 0
        state['total_forwards']=plan['inherited_forwards']+state['new_forwards']
        if rc:
            state['status']='BLOCKED_TEST_TECHNICAL'
        else:
            validate_predictions(job,read(output/'predictions.json'))
            env['CUDA_VISIBLE_DEVICES']=''
            if not (output/'scores.json').exists():
                rc=invoke_child([sys.executable,script,'score','--job',str(output/'input-job.json'),
                    '--predictions',str(output/'predictions.json'),'--scorer',str(output/'scoring-config.json'),
                    '--output',str(output/'scores.json')],output/'scoring.log',env,output/('scoring-exit-'+stamp+'.json'))
            if rc:
                state['status']='BLOCKED_TEST_SCORING'
            else:
                records=read(output/'predictions.json')['records']
                failed=sum(any(a['status']=='failed' for a in r['arms'].values()) for r in records)
                state.update(status='TEST_COMPLETE_WITH_FAILURES' if failed else 'TEST_COMPLETE',
                             planned=2545,processed=len(records),failed_cases=failed)
        write(output/'state.json',state,replace=True)
        print(state,flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['freeze','run'])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--inputs',type=Path)
    p.add_argument('--predecessor',type=Path)
    p.add_argument('--previous-test',type=Path)
    p.add_argument('--max-total-forwards',type=int,default=200000)
    a=p.parse_args()
    if a.command=='freeze':
        if not a.inputs or not a.predecessor: p.error('freeze requires inputs and predecessor')
        freeze(a.inputs.resolve(),a.predecessor.resolve(),a.output.resolve(),a.max_total_forwards,
               a.previous_test.resolve() if a.previous_test else None)
    else:
        run(a.output.resolve())
