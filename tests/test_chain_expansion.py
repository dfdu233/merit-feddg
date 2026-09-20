"""Exploratory expansion never promotes a stopped candidate into confirmation."""
import pytest
from test_algorithm_chain_controller import setup
from merit_feddg.algorithm_chain import cli
from merit_feddg.algorithm_chain.storage import read,write


@pytest.mark.parametrize('verdict',['pass','fail','inconclusive'])
def test_expansion_reports_then_stops_without_confirmation(tmp_path,monkeypatch,verdict):
    plan,calls=setup(tmp_path,monkeypatch)
    plan['experiment_mode']='fixed_candidate_exploratory_expansion'
    write(tmp_path/'plan.json',plan,replace=True)
    state=read(tmp_path/'state.json')
    state.update(node='C',candidate=dict(algorithm='project',layer=None,control='layout_average'),attempts={'C':2},forwards_used=10164)
    write(tmp_path/'state.json',state,replace=True)
    monkeypatch.setattr(cli,'quality',lambda *a,**k:dict(verdict=verdict))
    cli.execute(tmp_path)
    final=read(tmp_path/'state.json')
    assert final['node']=='EXPLORATORY_COMPLETE'
    assert final['forwards_used']==10164+240
    assert final['attempts']['C']==2 and not final['holdout_consumed']
    assert calls==[('worker','C'),('score','C')]
    assert not (tmp_path/'scale_plan.json').exists()
    cli.execute(tmp_path)
    assert len(calls)==2


def test_expansion_technical_failure_remains_retryable(tmp_path,monkeypatch):
    plan,calls=setup(tmp_path,monkeypatch)
    plan['experiment_mode']='fixed_candidate_exploratory_expansion'
    write(tmp_path/'plan.json',plan,replace=True)
    state=read(tmp_path/'state.json');state.update(node='C',candidate=dict(algorithm='project',layer=None,control='layout_average'),attempts={'C':2})
    write(tmp_path/'state.json',state,replace=True)
    monkeypatch.setattr(cli,'quality',lambda *a,**k:dict(verdict='technical_failure'))
    cli.execute(tmp_path)
    assert read(tmp_path/'state.json')['node']=='BLOCKED_TECHNICAL'


def test_expansion_device_change_preserves_cost_but_requires_new_runtime(tmp_path,monkeypatch):
    from merit_feddg.algorithm_chain.policy import DEFAULTS
    from merit_feddg.algorithm_chain.storage import digest,fingerprint
    source=tmp_path/'source';source.mkdir()
    directory=source/'C-attempt-1';directory.mkdir()
    write(directory/'decision.json',{'verdict':'fail'})
    write(directory/'budget.json',{'used':17})
    prior=dict(policy=DEFAULTS|{'max_node_attempts':3},runtime=dict(gpu_uuid='GPU-old',pins={'weights':'same'}),scorer=dict(import_roots=[],pins={'scorer':'same'}),
        exposed_pixels='exposed',test_pixels='test',inventory={'development':[{'records':[{'pixel_sha256':'old-pixel'}]}]},generation={'prior':{'max_new_tokens':1024}},code_pins={})
    prior['identity']=fingerprint(prior);write(source/'plan.json',prior)
    state=dict(plan_sha=prior['identity'],node='STOP_NO_CANDIDATE',candidate=dict(algorithm='project',layer=None,control='layout_average'),
        holdout_consumed=False,attempts={'C':1},forwards_used=17,runtime_sha='GPU-old-runtime',history=[dict(node='C',verdict='fail',directory=directory.name,report_sha=digest(directory/'decision.json'))])
    write(source/'state.json',state)
    spec={k:v for k,v in prior.items() if k!='identity'}
    spec.update(runtime=dict(gpu_uuid='GPU-new',pins={'weights':'same'}),confirmation=[],extension=[])
    inp=tmp_path/'input.json';write(inp,spec)
    def freeze(p,root):
        return p|dict(identity='new-plan',inventory={'development':[{'records':[{'pixel_sha256':'new-pixel'}]}]},generation={'added':{'max_new_tokens':1024}})
    monkeypatch.setattr(cli,'freeze_plan',freeze)
    out=tmp_path/'out';cli.freeze_expansion(inp,source,out)
    result=read(out/'state.json')
    assert result['forwards_used']==17 and result['attempts']=={'C':2}
    assert 'runtime_sha' not in result and result['node']=='C'
    assert read(source/'state.json')==state
    assert read(out/'C-attempt-1/decision.json')=={'verdict':'fail'}
