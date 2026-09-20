"""Freeze the explicitly authorized third A attempt without erasing prior costs.

This bounded migration accepts only the already blocked A=2 campaign. Scientific
fields remain identical; code, physical device, the identified desktop context,
and the explicitly authorized attempt ceiling may change. Standard retry still
has to grant the next attempt after migration.
"""
import argparse
from copy import deepcopy
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from merit_feddg.algorithm_chain.storage import canonical,digest,fingerprint,freeze_plan,lock,read,write


def migrate(source, output, plan_input, preflight):
    source,output,preflight=map(lambda p:Path(p).resolve(),(source,output,preflight))
    if output.exists():raise ValueError('New output directory required')
    with lock(source/'.controller.lock'):
        old=read(source/'plan.json');state=read(source/'state.json')
        if fingerprint({k:v for k,v in old.items() if k!='identity'})!=old['identity'] or state['plan_sha']!=old['identity']:
            raise ValueError('Predecessor identity mismatch')
        if state['node']!='BLOCKED_TECHNICAL' or state.get('blocked_node')!='A' or state['attempts']!={'A':2} or state['holdout_consumed']:
            raise ValueError('Expected two blocked A attempts, no holdout use')
        if len(state['history'])!=2:raise ValueError('Missing predecessor history')
        spent=0
        for entry in state['history']:
            directory=source/entry['directory']
            if entry['node']!='A' or entry['verdict']!='technical_failure' or digest(directory/'decision.json')!=entry['report_sha']:
                raise ValueError('Invalid failure history')
            spent+=read(directory/'budget.json')['used'] if (directory/'budget.json').exists() else 0
        if spent!=state['forwards_used']:raise ValueError('Predecessor cost mismatch')
        probe=read(preflight/'summary.json');budget=read(preflight/'budget.json')
        if probe['phase']!='A-attempt-3-technical-preflight' or probe['forwards']!=budget['used']:
            raise ValueError('Invalid technical preflight ledger')
        spec=read(plan_input)
        for key in ('schema','development','confirmation','extension','exposed_pixels','test_pixels','scorer'):
            if spec[key]!=old[key]:raise ValueError('Scientific field changed: '+key)
        expected=old['policy']|{'max_node_attempts':3}
        if spec['policy']!=expected:raise ValueError('Only authorized attempt ceiling may change')
        runtime=deepcopy(spec['runtime']); prior_runtime=deepcopy(old['runtime'])
        for key in ('gpu_uuid','allowed_display_contexts'):
            runtime.pop(key,None);prior_runtime.pop(key,None)
        if runtime!=prior_runtime:raise ValueError('Model/runtime pins changed')
        spec['repair_inheritance']=dict(predecessor_plan_sha=old['identity'],
            predecessor_state_sha256=digest(source/'state.json'),predecessor_code_pins=old['code_pins'],
            authorization='User explicitly approved per-node maximum of 3; retain failures, forwards and scientific thresholds',
            preflight_summary_sha256=digest(preflight/'summary.json'),preflight_forwards=budget['used'],
            migration_script_sha256=digest(Path(__file__)))
        plan=freeze_plan(spec,Path(__file__).resolve().parents[1])
        for key in ('inventory','generation','generation_pins','registry_pins'):
            if plan[key]!=old[key]:raise ValueError('Frozen data/generation changed: '+key)
        inherited=deepcopy(state)
        inherited.update(plan_sha=plan['identity'],forwards_used=spent+budget['used'])
        inherited.pop('runtime_sha',None) # New device/runtime must establish a new fingerprint.
        inherited['preflight_costs']=[dict(directory='A-attempt-3-preflight',forwards=budget['used'])]
        if inherited['forwards_used']>=plan['policy']['max_total_forwards']:raise ValueError('No forward budget remains')
        with lock(output/'.controller.lock'):
            for entry in state['history']:shutil.copytree(source/entry['directory'],output/entry['directory'])
            shutil.copytree(preflight,output/'A-attempt-3-preflight')
            write(output/'predecessor-state.json',state)
            write(output/'plan.json',plan);write(output/'state.json',inherited)
        print(plan['identity'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('source','output','plan-input','preflight'):p.add_argument('--'+key,required=True)
    a=p.parse_args();migrate(a.source,a.output,a.plan_input,a.preflight)
