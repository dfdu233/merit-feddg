"""Controller integration with explicitly synthetic, deterministic child artifacts."""
import copy
from pathlib import Path

import pytest
from merit_feddg.algorithm_chain import cli
from merit_feddg.algorithm_chain.policy import DEFAULTS, initial_state
from merit_feddg.algorithm_chain.storage import digest, fingerprint, read, write


def inventory(role, n):
    return dict(cohort=role, records=[dict(id=role+':'+str(i), pixel_sha256=role+str(i)) for i in range(n)])


def setup(tmp_path, monkeypatch, *, confirmation=True, extension=False):
    plan = dict(identity='frozen-test-plan', policy=DEFAULTS|{'bootstrap_repetitions':200},
        runtime={'gpu_uuid':'GPU-synthetic', 'pins':{'synthetic-model':'pinned'}},
        scorer={'pins':{'synthetic-scorer':'pinned'}, 'references':{'private':'only-scoring-child'}},
        inventory=dict(development=[inventory('dev',240)],
                       confirmation=[inventory('confirm-a',120),inventory('confirm-b',120)] if confirmation else [],
                       extension=[inventory('extension-a',120),inventory('extension-b',120)] if extension else []),
        generation={k:{} for k in ('dev','confirm-a','confirm-b','extension-a','extension-b')}, code_pins={})
    write(tmp_path/'plan.json',plan);write(tmp_path/'state.json',initial_state(plan['identity']))
    monkeypatch.setattr(cli,'verify_plan',lambda _:None) # Content pinning has separate filesystem tests.
    calls=[]
    def child(argv, log, env, exit_path):
        kind=argv[2];job=read(argv[argv.index('--job')+1]);out=Path(argv[argv.index('--output')+1])
        calls.append((kind,job['node']))
        if kind=='worker':
            assert 'scorer' not in job and 'references' not in job
            assert env['CUDA_VISIBLE_DEVICES']=='GPU-synthetic'
            rows=[]
            for inv in job['inventories']:
                for i,r in enumerate(inv['records']):
                    # Distributed old improvements/harms across each cohort.
                    b,e,c=(0,1,1) if i%6==0 else (1,0,1) if i%6==1 else (0,0,1) if i%6==2 else (1,1,1)
                    values=dict(generalist=b,compact=e)
                    if job['node']!='A':values.update(candidate=c,control=e)
                    arms={k:dict(text=str(v),token_ids=[v+2],status='complete',seconds=.1,bypass=False) for k,v in values.items()}
                    row=dict(r,cohort=inv['cohort'],job_sha=fingerprint(job),complete=True,arms=arms,
                             canary=dict(generalist=True,compact=True,audit=True,expert_exposed=True))
                    if job['node']=='A':row['diagnostic']=dict(status='no_divergence',sites=[])
                    rows.append(row)
            write(out/'predictions.json',dict(job_sha=fingerprint(job),planned_n=len(rows),complete=True,records=rows))
            write(out/'runtime.json',dict(engine='synthetic-fixture',version='fixed'))
            write(out/'budget.json',dict(used=len(rows),limit=job['remaining_forwards'],exhausted=False))
        else:
            assert kind=='score' and env['CUDA_VISIBLE_DEVICES']==''
            assert read(argv[argv.index('--scorer')+1])['references']
            predpath=Path(argv[argv.index('--predictions')+1]);pred=read(predpath)
            scores=[dict(id=r['id'],scores={k:float(a['text']) for k,a in r['arms'].items()}) for r in pred['records']]
            write(out,dict(job_sha=fingerprint(job),predictions_sha256=digest(predpath),
                           scorer_pins=plan['scorer']['pins'],records=scores))
        return 0
    monkeypatch.setattr(cli,'invoke_child',child)
    return plan,calls


def test_full_chain_A_to_C_to_D_and_release_resume(tmp_path,monkeypatch):
    _,calls=setup(tmp_path,monkeypatch)
    cli.execute(tmp_path)
    state=read(tmp_path/'state.json')
    assert state['node']=='READY_FOR_SCALE' and state['candidate']['algorithm']=='project'
    assert [x['node'] for x in state['history']]==['A','C','D']
    assert state['forwards_used']==720
    assert calls==[(k,n) for n in ('A','C','D') for k in ('worker','score')]
    release=read(tmp_path/'scale_plan.json');assert not release['official_test_launched']
    (tmp_path/'scale_plan.json').unlink() # Crash after state write, before release publication.
    cli.execute(tmp_path)
    assert read(tmp_path/'scale_plan.json')==release and len(calls)==6


def test_once_and_resume_never_repeat_completed_stage(tmp_path,monkeypatch):
    _,calls=setup(tmp_path,monkeypatch)
    cli.execute(tmp_path,once=True)
    assert read(tmp_path/'state.json')['node']=='C' and len(calls)==2
    cli.execute(tmp_path,once=True)
    assert read(tmp_path/'state.json')['node']=='D' and len(calls)==4
    cli.execute(tmp_path,once=True)
    assert read(tmp_path/'state.json')['node']=='READY_FOR_SCALE' and len(calls)==6


def test_missing_holdout_is_terminal_data_block_not_new_split(tmp_path,monkeypatch):
    _,calls=setup(tmp_path,monkeypatch,confirmation=False)
    cli.execute(tmp_path)
    assert read(tmp_path/'state.json')['node']=='BLOCKED_DATA'
    assert len(calls)==4 and not (tmp_path/'scale_plan.json').exists()


def test_predeclared_extension_reuses_fixed_candidate(tmp_path,monkeypatch):
    plan,calls=setup(tmp_path,monkeypatch,extension=True)
    original=cli.quality
    def uncertain_once(rows,**kwargs):
        if kwargs.get('confirmation') and len(rows)==240:
            return dict(verdict='inconclusive',reason='Synthetic uncertainty for branch coverage')
        return original(rows,**kwargs)
    monkeypatch.setattr(cli,'quality',uncertain_once)
    cli.execute(tmp_path)
    state=read(tmp_path/'state.json')
    assert state['node']=='READY_FOR_SCALE'
    assert [x['node'] for x in state['history']]==['A','C','D','E']
    d=read(tmp_path/'D-attempt-1/input-job.json');e=read(tmp_path/'E-attempt-1/input-job.json')
    assert d['candidate']==e['candidate']
    assert read(tmp_path/'E-attempt-1/decision.json')['confirmation_plus_extension_n']==480
    assert len(calls)==8


def test_failed_child_does_not_launch_another_algorithm(tmp_path,monkeypatch):
    _,calls=setup(tmp_path,monkeypatch)
    def fail(*a,**k):return 9
    monkeypatch.setattr(cli,'invoke_child',fail)
    cli.execute(tmp_path)
    assert read(tmp_path/'state.json')['node']=='BLOCKED_TECHNICAL'
    assert not (tmp_path/'C-attempt-1').exists()


def test_immutable_scores_reject_wrong_prediction_hash(tmp_path,monkeypatch):
    plan,_=setup(tmp_path,monkeypatch)
    cli.execute(tmp_path,once=True)
    directory=tmp_path/'A-attempt-1'
    scores=read(directory/'scores.json');scores['predictions_sha256']='wrong'
    write(directory/'scores.json',scores,replace=True)
    with pytest.raises(ValueError,match='another prediction'):
        cli.evaluate_artifacts(read(directory/'input-job.json'),directory/'predictions.json',directory/'scores.json',plan)


def test_runtime_change_cannot_pass_confirmation(tmp_path,monkeypatch):
    _,calls=setup(tmp_path,monkeypatch)
    cli.execute(tmp_path,once=True)
    original=cli.invoke_child
    def changed(argv,log,env,exit_path):
        rc=original(argv,log,env,exit_path)
        if argv[2]=='worker':
            out=Path(argv[argv.index('--output')+1])
            write(out/'runtime.json',dict(engine='other-runtime'),replace=True)
        return rc
    monkeypatch.setattr(cli,'invoke_child',changed)
    cli.execute(tmp_path)
    assert read(tmp_path/'state.json')['node']=='BLOCKED_TECHNICAL'
    assert not (tmp_path/'D-attempt-1').exists()
