"""Finite autonomous controller: inference child -> scoring child -> branch decision.

No shell command strings, labels in inference jobs, opportunistic layer sweeps,
or automatic official TEST launch. READY writes a frozen scale plan only.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from . import SCHEMA
from .policy import DEFAULTS, TERMINAL, initial_state, retry_technical, transition
from .statistics import localization, quality, scored_rows
from .storage import canonical, digest, fingerprint, freeze_plan, lock, read, verify_plan, write

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT/'scripts/run_algorithm_chain.py'


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
