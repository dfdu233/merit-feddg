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
