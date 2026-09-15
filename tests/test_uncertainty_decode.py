import numpy as np
import pytest
from types import SimpleNamespace
from merit_feddg.uncertainty_decode import (
    adaptive_step, binary_uncertainty, categorical_uncertainty,
    semantic_frequency_uncertainty, decode_adaptive)
from merit_feddg.uncertainty_experiment import applicability, swap_decision


def test_native_entropies():
    assert binary_uncertainty(.5) == 1
    assert binary_uncertainty(0) == 0
    assert binary_uncertainty(1) == 0
    assert categorical_uncertainty([.25]*4) == 1
    assert semantic_frequency_uncertainty([0,0,0,0,0]) == 0
    assert semantic_frequency_uncertainty([0,1,2,3,4]) == pytest.approx(1)
    with pytest.raises(ValueError): categorical_uncertainty([.6,.6])
    with pytest.raises(TypeError): binary_uncertainty(True)


def test_acd_reference_and_source_discount():
    a,b=np.array([0.,1.,2.]),np.array([0.,1.,4.])
    mixed,audit=adaptive_step(a,b,mode='acd')
    rho=audit['base_entropy_nats']/(audit['base_entropy_nats']+audit['evidence_entropy_nats'])
    assert np.array_equal(mixed,(1-rho)*a+rho*b)
    for u in np.linspace(0,1,100):
        _,d=adaptive_step(a,b,mode='source_acd',source_u=float(u))
        assert d['effective_weight']==pytest.approx((1-u)*rho)
    assert np.array_equal(adaptive_step(a,b,mode='fixed',fixed_weight=1.5)[0],1.5*b-.5*a)


def test_missing_not_confident_and_zero_entropy_boundary():
    a,b=[-1000.,1000.],[1000.,-1000.]
    assert adaptive_step(a,b,mode='acd')[1]['acd_weight']==.5
    assert adaptive_step(a,b,mode='source_acd')[1]['reason']=='source_uncertainty_missing'
    assert adaptive_step(a,b,mode='acd',applicable=False)[1]['effective_weight']==0


def test_same_prefix_and_endpoint():
    seen=[[],[]]
    def session(i):
        def scores(prefix):
            seen[i].append(prefix)
            return np.array([0.,2.,1.]) if not prefix else np.array([0.,1.,2.])
        return SimpleNamespace(next_scores=scores,decode=lambda p:str(p),
            propose=lambda *a,**k:[SimpleNamespace(text='fixed',tokens=(1,2))])
    a,b=session(0),session(1)
    r=decode_adaptive(a,b,max_tokens=4,eos_ids={2},mode='acd')
    assert seen[0]==seen[1]==[(),(1,)]
    assert r['token_ids']==[1,2]
    assert decode_adaptive(a,None,max_tokens=4,eos_ids={2},mode='source_only')['reason']=='source_uncertainty_missing'


@pytest.mark.parametrize('question,modality,allowed',[
    ('is there a pneumothorax?', 'cxr', True),
    ('is a pleural effusion present?', 'cxr', True),
    ('is cardiomegaly shown?', 'cxr', True),
    ('is there a right-sided pleural effusion?', 'cxr', False),
    ('is cardiomegaly shown?', 'ct', False),
    ('is this a tension pneumothorax?', 'cxr', False),
    ('which disease is this?', 'cxr', False),
])
def test_native_task_scope(question,modality,allowed):
    assert applicability({'question':question},{'input_modality':modality})['allowed']==allowed


def test_verifier_swap_fail_closed():
    assert swap_decision('B','A')['replace']
    for a,b in [('A','B'),('B','B'),('TIE','A'),('UNCERTAIN','A'),('probably B','A')]:
        assert not swap_decision(a,b)['replace']
