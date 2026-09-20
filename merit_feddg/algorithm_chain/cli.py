"""Finite autonomous controller: inference child -> scoring child -> branch decision.

No shell command strings, labels in inference jobs, opportunistic layer sweeps,
or automatic official TEST launch. READY writes a frozen scale plan only.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from . import SCHEMA
from .policy import DEFAULTS, TERMINAL, initial_state, retry_technical, transition
from .statistics import localization, quality, scored_rows
from .storage import canonical, digest, fingerprint, freeze_plan, lock, read, verify_plan, write

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT/'scripts/run_algorithm_chain.py'


def freeze_expansion(plan_input, source, output):
    """Explicit post-stop C expansion; never reopen the original scientific chain."""
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError('Expansion requires a new output directory')
    with lock(source/'.controller.lock'):
        prior, state = read(source/'plan.json'), read(source/'state.json')
        if fingerprint({k:v for k,v in prior.items() if k!='identity'}) != prior['identity'] or state['plan_sha'] != prior['identity']:
            raise ValueError('Predecessor identity mismatch')
        if state['node'] != 'STOP_NO_CANDIDATE' or state.get('holdout_consumed'):
            raise ValueError('Only explicit exploratory expansion after an unconfirmed scientific stop')
        candidate = dict(algorithm='project', layer=None, control='layout_average')
        if state['candidate'] != candidate or state['attempts'].get('C',0) >= prior['policy']['max_node_attempts']:
            raise ValueError('Require the fixed C candidate and remaining inherited attempt budget')
        spec = read(plan_input)
        runtime, prior_runtime = deepcopy(spec['runtime']), deepcopy(prior['runtime'])
        for key in ('gpu_uuid','allowed_display_contexts'):
            runtime.pop(key,None); prior_runtime.pop(key,None)
        if spec.get('confirmation') or spec.get('extension') or spec['policy'] != prior['policy'] or runtime != prior_runtime:
            raise ValueError('Expansion may not change scientific thresholds/model runtime or declare holdouts')
        for key in ('exposed_pixels','test_pixels'):
            if spec[key] != prior[key]:
                raise ValueError('Keep the existing exposure and TEST audit')
        if spec['scorer']['import_roots'] != prior['scorer']['import_roots'] or any(spec['scorer']['pins'].get(k)!=v for k,v in prior['scorer']['pins'].items()):
            raise ValueError('Scorer pins must be retained; only new reference files may be added')
        if not state['history'] or state['history'][-1]['node'] != 'C' or state['history'][-1]['verdict'] not in ('fail','inconclusive'):
            raise ValueError('Require the preserved completed C scientific decision')
        spent=0
        for entry in state['history']:
            if digest(source/entry['directory']/'decision.json') != entry['report_sha']:
                raise ValueError('Predecessor decision changed')
            budget_path=source/entry['directory']/'budget.json'
            spent += read(budget_path)['used'] if budget_path.exists() else 0
        for entry in state.get('preflight_costs',[]):
            if read(source/entry['directory']/'budget.json')['used'] != entry['forwards']:
                raise ValueError('Preflight cost changed')
            spent += entry['forwards']
        if spent != state['forwards_used'] or spent >= prior['policy']['max_total_forwards']:
            raise ValueError('Inherited budget mismatch or exhaustion')
        spec.update(experiment_mode='fixed_candidate_exploratory_expansion',
            expansion=dict(predecessor_plan_sha=prior['identity'],predecessor_state_sha256=digest(source/'state.json'),
                candidate=candidate,prior_verdict=state['node'],inherited_forwards=state['forwards_used'],
                interpretation='User-requested exploratory evaluation of additional exposed TRAIN data; never independent confirmation'))
        plan=freeze_plan(spec,ROOT)
        administrative={str(ROOT/'merit_feddg/algorithm_chain/cli.py'),str(ROOT/'merit_feddg/algorithm_chain/policy.py')}
        if any(plan['code_pins'].get(k)!=v for k,v in prior['code_pins'].items() if k not in administrative):
            raise ValueError('Fixed candidate generation implementation changed')
        old_pixels={r['pixel_sha256'] for inv in prior['inventory']['development'] for r in inv['records']}
        new_pixels={r['pixel_sha256'] for inv in plan['inventory']['development'] for r in inv['records']}
        if old_pixels & new_pixels:
            raise ValueError('Additional cases must not repeat predecessor images')
        settings=list(prior['generation'].values())
        if not settings or any(v!=settings[0] for v in plan['generation'].values()):
            raise ValueError('Keep the frozen generation parameters')
        inherited=deepcopy(state)
        inherited.update(plan_sha=plan['identity'],node='C',experiment_mode=plan['experiment_mode'])
        inherited.pop('blocked_node',None)
        if spec['runtime']['gpu_uuid'] != prior['runtime']['gpu_uuid']:
            inherited.pop('runtime_sha',None)
        inherited['attempts']['C'] += 1
        with lock(output/'.controller.lock'):
            for entry in state['history']:
                shutil.copytree(source/entry['directory'],output/entry['directory'])
            for entry in state.get('preflight_costs',[]):
                shutil.copytree(source/entry['directory'],output/entry['directory'])
            write(output/'predecessor-state.json',state)
            write(output/'plan.json',plan)
            write(output/'state.json',inherited)
        print(plan['identity'])


def stage_job(plan, state):
    node = state['node']
    role = 'development' if node in ('A','B','C') else 'confirmation' if node=='D' else 'extension'
    inventories = plan['inventory'][role]
    if not inventories:
        raise ValueError('No predeclared '+role+' cohort. Old TRAIN128 cannot become new confirmation.')
    return dict(schema=SCHEMA,node=node,candidate=state['candidate'],
        policy=plan['policy'],inventories=inventories,runtime=plan['runtime'],
        generation={i['cohort']:plan['generation'][i['cohort']] for i in inventories},
        plan_sha=plan['identity'],remaining_forwards=plan['policy']['max_total_forwards']-state['forwards_used'])


def validate_predictions(job, predictions):
    if predictions.get('job_sha') != fingerprint(job) or predictions.get('complete') is not True:
        raise ValueError('Wrong job hash or incomplete generation')
    expected = {r['id']:r for inv in job['inventories'] for r in inv['records']}
    records = predictions.get('records',[])
    if predictions.get('planned_n') != len(expected) or len(records) != len(expected):
        raise ValueError('Missing planned cases')
    if len({r['id'] for r in records}) != len(expected) or {r['id'] for r in records} != set(expected):
        raise ValueError('Extra/duplicate/missing prediction ID')
    expected_arms = {'generalist','compact'} if job['node']=='A' else {'generalist','compact','candidate','control'}
    for r in records:
        if r['pixel_sha256'] != expected[r['id']]['pixel_sha256'] or r.get('job_sha') != fingerprint(job):
            raise ValueError('Wrong image/job identity in case')
        if set(r['arms']) != expected_arms or r.get('complete') is not True:
            raise ValueError('Missing arm or incomplete case')
        if job['node'] == 'A':
            d = r.get('diagnostic', {})
            if d.get('status') not in ('failed', 'divergence', 'same_answer', 'no_divergence'):
                raise ValueError('Missing or invalid diagnostic result')
            if d['status'] == 'divergence' and len(d.get('sites', [])) != 2:
                raise ValueError('Both predeclared diagnostic sites are required')
        for name, arm in r['arms'].items():
            if arm.get('status') not in ('complete','failed'):
                raise ValueError('Unknown arm status')
            if arm['status']=='complete' and (not isinstance(arm.get('text'),str) or not arm['token_ids']):
                raise ValueError('Completed arm lacks prediction')
            if name in ('candidate','control') and arm['status']=='complete':
                if type(arm.get('seconds')) not in (float,int) or not 0 <= arm['seconds'] < float('inf'):
                    raise ValueError('Actual finite cost required')
    # Native canary evidence must exist for every source cohort, not just one overall.
    for inv in job['inventories']:
        first = [r for r in records if r['id'] in {x['id'] for x in inv['records'][:2]}]
        if any(not all(r.get('canary',{}).get(k) is True for k in ('generalist','compact','audit')) for r in first):
            raise ValueError('Native technical canary absent or failed')
        if not any(r['canary']['expert_exposed'] for r in first):
            raise ValueError('All-bypass canary')
    return records


def evaluate_artifacts(job, predictions_path, scores_path, plan):
    predictions = read(predictions_path)
    records = validate_predictions(job,predictions)
    scores = read(scores_path)
    if scores['predictions_sha256'] != digest(predictions_path) or scores['job_sha'] != fingerprint(job):
        raise ValueError('Scores belong to another prediction artifact')
    if scores['scorer_pins'] != plan['scorer']['pins']:
        raise ValueError('Scorer changed after freeze')
    merged = scored_rows(records,scores['records'])
    report = localization(merged,plan['policy']) if job['node']=='A' else quality(
        merged,config=plan['policy'],confirmation=job['node'] in ('D','E'))
    return report, merged


def invoke_child(argv, log, env, exit_path):
    start = time.time()
    with Path(log).open('ab') as stream:
        completed = subprocess.run(argv,stdout=stream,stderr=subprocess.STDOUT,env=env,check=False)
    write(exit_path,dict(argv=argv,returncode=completed.returncode,started_unix=start,ended_unix=time.time()))
    return completed.returncode


def execute(output, *, once=False):
    output = Path(output).resolve()
    with lock(output/'.controller.lock'):
        plan = read(output/'plan.json')
        verify_plan(plan)
        state = read(output/'state.json')
        if state['plan_sha'] != plan['identity']:
            raise ValueError('State/plan mismatch')
        while state['node'] not in TERMINAL:
            verify_plan(plan)
            node = state['node']
            if state['forwards_used'] >= plan['policy']['max_total_forwards']:
                state['node'] = 'BLOCKED_BUDGET'
                write(output/'state.json',state,replace=True)
                break
            try:
                job = stage_job(plan,state)
            except ValueError as error:
                state.update(node='BLOCKED_DATA',reason=str(error),blocked_node=node)
                write(output/'state.json',state,replace=True)
                break
            attempt = state['attempts'].get(node,1)
            directory = output/(node+'-attempt-'+str(attempt))
            directory.mkdir(exist_ok=True)
            job_path = directory/'input-job.json'
            if job_path.exists():
                if read(job_path) != job:
                    raise ValueError('Frozen stage job changed')
            else:
                write(job_path,job)
            pred_path = directory/'predictions.json'
            scores_path = directory/'scores.json'
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=plan['runtime']['gpu_uuid'],HF_HUB_OFFLINE='1',
                       TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false')
            env['PYTHONPATH'] = str(ROOT)
            report = None
            if not pred_path.exists():
                argv = [sys.executable,str(SCRIPT),'worker','--job',str(job_path),'--output',str(directory)]
                rc = invoke_child(argv,directory/'inference.log',env,
                                  directory/('inference-exit-'+str(time.time_ns())+'.json'))
                if rc:
                    report = dict(verdict='technical_failure',reason='Inference child failed; see durable log/exit')
            runtime_path = directory/'runtime.json'
            if report is None:
                if not runtime_path.exists():
                    report = dict(verdict='technical_failure',reason='Native runtime provenance missing')
                elif state.get('runtime_sha') not in (None, digest(runtime_path)):
                    report = dict(verdict='technical_failure',reason='Model/library/runtime changed between stages')
                else:
                    state['runtime_sha'] = digest(runtime_path)
            budget_path = directory/'budget.json'
            spent = read(budget_path)['used'] if budget_path.exists() else 0
            if report is None and not scores_path.exists():
                # Reference paths go exclusively to this separate scoring process.
                scorer_path = directory/'scoring-config.json'
                if not scorer_path.exists():
                    write(scorer_path,plan['scorer'])
                argv = [sys.executable,str(SCRIPT),'score','--job',str(job_path),
                        '--predictions',str(pred_path),'--scorer',str(scorer_path),'--output',str(scores_path)]
                scoring_env = dict(env,CUDA_VISIBLE_DEVICES='')
                rc = invoke_child(argv,directory/'scoring.log',scoring_env,
                                  directory/('scoring-exit-'+str(time.time_ns())+'.json'))
                if rc:
                    report = dict(verdict='technical_failure',reason='Frozen scoring child failed')
            if report is None:
                report, rows = evaluate_artifacts(job,pred_path,scores_path,plan)
                if node == 'E':
                    previous = [x for x in state['history'] if x['node']=='D'][-1]
                    prior_directory = output/previous['directory']
                    old_job = read(prior_directory/'input-job.json')
                    if old_job['candidate'] != job['candidate'] or old_job['plan_sha'] != job['plan_sha']:
                        raise ValueError('Confirmation candidate or plan changed before extension')
                    _, old_rows = evaluate_artifacts(old_job,prior_directory/'predictions.json',prior_directory/'scores.json',plan)
                    if {r['pixel_sha256'] for r in rows}&{r['pixel_sha256'] for r in old_rows}:
                        raise ValueError('Extension reuses confirmation images')
                    report = quality([*old_rows,*rows],config=plan['policy'],confirmation=True)
                    report['confirmation_plus_extension_n'] = len(old_rows)+len(rows)
            report.update(node=node,job_sha=fingerprint(job),stage_forwards=spent)
            report_path = directory/'decision.json'
            if report_path.exists():
                if read(report_path) != report:
                    raise ValueError('Existing immutable decision changed')
            else:
                write(report_path,report)
            state = transition(state,report,digest(report_path))
            if plan.get('experiment_mode') == 'fixed_candidate_exploratory_expansion' and report['verdict'] != 'technical_failure':
                # A favorable exploratory score cannot resurrect the failed
                # release chain or consume any confirmation dataset.
                state.update(node='EXPLORATORY_COMPLETE',holdout_consumed=False)
            if budget_path.exists() and read(budget_path).get('exhausted'):
                state['node'] = 'BLOCKED_BUDGET'
            state['forwards_used'] += spent
            state['history'][-1]['directory'] = directory.name
            state['attempts'][node] = attempt
            write(output/'state.json',state,replace=True)
            print(canonical(dict(node=node,verdict=report['verdict'],next=state['node'])),flush=True)
            if once:
                break
        if state['node']=='READY_FOR_SCALE':
            release = dict(status='READY_FOR_SCALE',candidate=state['candidate'],plan_sha=plan['identity'],
                code_pins=plan['code_pins'],runtime_pins=plan['runtime']['pins'],scorer_pins=plan['scorer']['pins'],
                evidence=state['history'],official_test_launched=False,clinical_safety_claim=False,
                scope='Larger frozen research evaluation for this receiver; not deployment or universal noninferiority')
            path = output/'scale_plan.json'
            if path.exists() and read(path) != release:
                raise ValueError('Frozen release artifact changed')
            if not path.exists():
                write(path,release)
        print(canonical(state),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command',required=True)
    p = commands.add_parser('freeze'); p.add_argument('--plan',required=True); p.add_argument('--output',required=True)
    p = commands.add_parser('freeze-expansion'); p.add_argument('--plan',required=True)
    p.add_argument('--source',required=True); p.add_argument('--output',required=True)
    p = commands.add_parser('run'); p.add_argument('--output',required=True); p.add_argument('--once',action='store_true')
    p = commands.add_parser('status'); p.add_argument('--output',required=True)
    p = commands.add_parser('retry'); p.add_argument('--output',required=True)
    p = commands.add_parser('worker'); p.add_argument('--job',required=True); p.add_argument('--output',required=True)
    p = commands.add_parser('score'); p.add_argument('--job',required=True); p.add_argument('--predictions',required=True)
    p.add_argument('--scorer',required=True); p.add_argument('--output',required=True)
    p = commands.add_parser('policy'); p.add_argument('--output')
    a = parser.parse_args()
    if a.command=='policy':
        if a.output: write(a.output,DEFAULTS)
        else: print(canonical(DEFAULTS))
    elif a.command=='freeze-expansion':
        freeze_expansion(a.plan,a.source,a.output)
    elif a.command=='freeze':
        out = Path(a.output).resolve()
        with lock(out/'.controller.lock'):
            plan = freeze_plan(read(a.plan),ROOT)
            write(out/'plan.json',plan)
            write(out/'state.json',initial_state(plan['identity']))
            print(plan['identity'])
    elif a.command=='run':
        execute(a.output,once=a.once)
    elif a.command=='status':
        print(canonical(read(Path(a.output)/'state.json')))
    elif a.command=='retry':
        out = Path(a.output)
        with lock(out/'.controller.lock'):
            plan=read(out/'plan.json'); verify_plan(plan)
            state=retry_technical(read(out/'state.json'),plan['policy'])
            write(out/'state.json',state,replace=True)
    elif a.command=='worker':
        from .native import run_job
        run_job(read(a.job),a.output)
    else:
        from .native import score_job
        score_job(read(a.job),a.predictions,read(a.scorer),a.output)
