"""CPU algebra, real synthetic transformer forwards, and decision graph tests.

No medical model weights, patient data, or manufactured medical efficacy.
"""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from merit_feddg.algorithm_chain.decoding import (
    DecodePolicy, ResidualRelay, _full_logits, cut_layers, diagnose, generate, project_residual)
from merit_feddg.algorithm_chain.policy import (
    DEFAULTS, TERMINAL, initial_state, retry_technical, transition, validate_policy)
from merit_feddg.algorithm_chain.statistics import (
    localization, paired_interval, quality, scored_rows, wilson)
from merit_feddg.algorithm_chain.storage import (
    canonical, digest, fingerprint, freeze_plan, pixel_hash, read, source_inventory, write)
from merit_feddg.algorithm_chain.native import ForwardBudget, verify_transport, _check_cached_decision
from merit_feddg.algorithm_chain.cli import stage_job, validate_predictions
from merit_feddg.huatuo_pathway import PreparedPrompt
from merit_feddg.pathway_restore import PathwayError, VisualSpan

# Independent native-shaped tiny GQA decoder (not a mocked logits-only callback).
class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(8,8); self.k=nn.Linear(8,4); self.v=nn.Linear(8,4)
        self.o=nn.Linear(8,8)
    def forward(self,x,past=None):
        b,n,d=x.shape
        q=self.q(x).view(b,n,4,2).transpose(1,2)
        k=self.k(x).view(b,n,2,2).transpose(1,2)
        v=self.v(x).view(b,n,2,2).transpose(1,2)
        if past is not None:
            k=torch.cat([past[0],k],2); v=torch.cat([past[1],v],2)
        keys=k.shape[2]
        scores=q@k.repeat_interleave(2,1).transpose(-1,-2)/2**.5
        mask=torch.arange(keys)[None,:] <= torch.arange(n)[:,None]+keys-n
        a=scores.masked_fill(~mask,-torch.inf).softmax(-1)
        return self.o((a@v.repeat_interleave(2,1)).transpose(1,2).reshape(b,n,d)),(k,v)

class Block(nn.Module):
    def __init__(self):
        super().__init__();self.self_attn=Attention();self.mlp=nn.Sequential(nn.Linear(8,12),nn.GELU(),nn.Linear(12,8))
    def forward(self,x,past=None):
        a,c=self.self_attn(x,past);h=x+a;return h+self.mlp(h),c

class Tiny(nn.Module):
    def __init__(self):
        super().__init__();self.embed_tokens=nn.Embedding(19,8);self.layers=nn.ModuleList([Block() for _ in range(6)])
        self.head=nn.Linear(8,19);self.consumed=[]
    def get_model(self):return self
    def forward(self,input_ids=None,inputs_embeds=None,past_key_values=None,use_cache=True,**kwargs):
        self.consumed.append(None if input_ids is None else input_ids.tolist())
        h=self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        caches=[]
        for i,layer in enumerate(self.layers):
            h,c=layer(h,None if past_key_values is None else past_key_values[i]);caches.append(c)
        return SimpleNamespace(logits=self.head(h),past_key_values=tuple(caches) if use_cache else None)

@pytest.fixture
def model():
    torch.manual_seed(5)
    return Tiny().eval().requires_grad_(False)


def prompt(n,seed):
    x=torch.randn(1,n,8,generator=torch.Generator().manual_seed(seed))
    x[:,1:3]=1 # identical native image tokens
    return PreparedPrompt(x,torch.ones(1,n,dtype=torch.long),torch.arange(n).unsqueeze(0),VisualSpan(1,3))


def policy():return DecodePolicy(6,(18,),1.2,1,(18,))


def test_relative_sites_are_predeclared(model):
    assert cut_layers(model)==(1,3)

@pytest.mark.parametrize('settings',[{'max_new_tokens':0},{'min_new_tokens':6},{'eos_token_ids':()},
    {'repetition_penalty':float('nan')},{'processor_prefix':(-1,)}])
def test_bad_decode_policy(settings):
    with pytest.raises(ValueError):
        DecodePolicy(**(dict(max_new_tokens=6,eos_token_ids=(18,))|settings))


def test_bos_processor_not_output_length():
    p=DecodePolicy(4,(2,),2,1,(0,))
    scores=p.scores(torch.tensor([4.,3.,9.]),[])
    assert scores[0]==2 and scores[1]==3 and torch.isneginf(scores[2])
    assert p.select(torch.tensor([4.,3.,9.]),[])==1

@pytest.mark.parametrize('algo',['relay','relay_roll','audit'])
def test_same_prefix_private_caches_and_weight_freeze(model,algo):
    before={k:v.clone() for k,v in model.state_dict().items()}
    result=generate(model,prompt(5,1),prompt(8,2),policy(),algorithm=algo,layer=1,context_limit=40)
    calls=[x for x in model.consumed if x is not None]
    assert all(calls[i]==calls[i+1] for i in range(0,len(calls),2))
    assert result['forwards'][0]==result['forwards'][1]
    assert result['token_ids'] and not result['weights_updated']
    assert all(torch.equal(v,before[k]) for k,v in model.state_dict().items())
    assert not getattr(model,'_merit_chain_active',False)
    assert all(not x._forward_hooks for x in model.layers)


def test_audit_no_hook_parity(model):
    r,e=prompt(5,1),prompt(8,2)
    a=generate(model,r,e,policy(),algorithm='off',context_limit=40)
    b=generate(model,r,e,policy(),algorithm='audit',layer=1,context_limit=40)
    assert a['token_ids']==b['token_ids']

@pytest.mark.parametrize('algo',['relay','relay_roll','project','layout_average'])
def test_identical_input_bypass(model,algo):
    r=prompt(5,1)
    value=generate(model,r,r,policy(),algorithm=algo,layer=1,controls=[r,r],context_limit=40)
    assert value['bypass'] and value['effective_algorithm']=='off' and value['forwards'][0]==0


def test_residual_restore_last_query_only(model):
    r,e=prompt(5,1),prompt(8,2)
    ref,last={},{}
    h=model.layers[1].register_forward_hook(lambda m,a,o: ref.update(value=o[0].detach().clone()))
    with torch.inference_mode():_full_logits(model,r,[])
    h.remove()
    h=model.layers[1].register_forward_hook(lambda m,a,o: last.update(value=o[0].detach().clone()))
    with torch.inference_mode():_full_logits(model,e,[])
    h.remove()
    seen={}
    with torch.inference_mode(),ResidualRelay(model,(1,)) as relay:
        with relay.use('reference',0):_full_logits(model,r,[])
        h=model.layers[1].register_forward_hook(lambda m,a,o: seen.update(value=o[0].detach().clone()))
        with relay.use('restore',0,1):_full_logits(model,e,[])
        h.remove()
    assert torch.equal(seen['value'][:,-1:],ref['value'][:,-1:])
    assert torch.equal(seen['value'][:,:-1],last['value'][:,:-1])


def test_exception_safe_and_stale_reference(model):
    with pytest.raises(PathwayError):
        with ResidualRelay(model,(1,)) as relay:
            with relay.use('restore',0,1):pass
    assert not getattr(model,'_merit_chain_active',False)
    assert not model.layers[1]._forward_hooks


def test_diagnostic_real_forward(model):
    value=diagnose(model,prompt(5,1),prompt(8,2),policy(),context_limit=40)
    assert value['status'] in ('divergence','same_answer','no_divergence')
    assert value['forwards']>=2 and not value['labels_used']
    if value['status']=='divergence':
        assert len(value['sites'])==2 and value['a']!=value['b']
        assert all({'restore','roll'}<=set(x) for x in value['sites'])


def test_projection_zero_nuisance_equals_expert_distribution():
    z0=torch.tensor([1.,0.,-1.]);ze=torch.tensor([0.,2.,-1.])
    out,audit=project_residual(z0,ze,[ze,ze])
    torch.testing.assert_close(out.softmax(-1),ze.softmax(-1))
    assert audit['rank']==0


def test_projection_removes_exact_nuisance_signal():
    z0=torch.zeros(4);ze=torch.tensor([1.,-1.,0.,0.])
    out,audit=project_residual(z0,ze,[2*ze,3*ze])
    torch.testing.assert_close(out.softmax(-1),z0.softmax(-1),atol=1e-6,rtol=1e-6)
    assert audit['rank']==1 and audit['orthogonality_error']<1e-8


def test_projection_retains_orthogonal_task_signal():
    z0=torch.zeros(4);r=torch.tensor([1.,-1.,2.,-2.]);n=torch.tensor([1.,-1.,0.,0.])
    out,a=project_residual(z0,r,[r+n,r+2*n])
    expected=torch.tensor([0.,0.,2.,-2.])
    torch.testing.assert_close(out.softmax(-1),expected.softmax(-1),atol=1e-6,rtol=1e-6)
    assert a['retained_weighted_norm']>0

@pytest.mark.parametrize('bad',[[],[torch.zeros(3)],[torch.zeros(3),torch.ones(4)],
                                    [torch.zeros(3),torch.full((3,),float('nan'))]])
def test_bad_projection_inputs(bad):
    with pytest.raises((PathwayError,ValueError)):
        project_residual(torch.zeros(3),torch.ones(3),bad)

@pytest.mark.parametrize('algo',['project','layout_average'])
def test_four_branch_real_decode(model,algo):
    result=generate(model,prompt(5,1),prompt(8,2),policy(),algorithm=algo,
                    controls=[prompt(7,3),prompt(9,4)],context_limit=40)
    assert len(set(result['forwards']))==1 and len(result['forwards'])==4
    calls=[x for x in model.consumed if x is not None]
    assert all(calls[i:i+4]==[calls[i]]*4 for i in range(0,len(calls),4))


def test_unknown_algorithm_and_overflow(model):
    with pytest.raises(ValueError):generate(model,prompt(5,1),prompt(8,2),policy(),algorithm='invented',context_limit=40)
    with pytest.raises(PathwayError):generate(model,prompt(5,1),prompt(8,2),policy(),algorithm='off',context_limit=9)


def test_hard_forward_budget_is_persisted(model,tmp_path):
    path=tmp_path/'budget.json'
    with pytest.raises(PathwayError,match='budget'):
        with ForwardBudget(model,1,path):
            with torch.inference_mode():
                _full_logits(model,prompt(5,1),[]);_full_logits(model,prompt(5,1),[])
    assert read(path)==dict(used=1,limit=1,exhausted=True)
    assert not model._forward_pre_hooks


def test_transport_uses_actual_decode_trace():
    text='native prompt';t={'prompt_sha256':__import__('hashlib').sha256(text.encode()).hexdigest()}
    case={'compact_raw':{'trace':[{'event':'decode','evidence_transport':t}]}}
    verify_transport(case,text,t)
    with pytest.raises(ValueError):verify_transport(case,text+'wrong',t)
    with pytest.raises(ValueError):verify_transport({'compact_raw':{'trace':[]}},text,t)


def test_cached_prefix_check_rejects_misalignment():
    d=dict(status='divergence',prefix_ids=[1],a=2,b=3)
    _check_cached_decision(d,[1,2],[1,3])
    with pytest.raises(ValueError):_check_cached_decision(d,[9,2],[1,3])

@pytest.mark.parametrize('node,verdict,next_node',[('A','pass','B'),('A','fail','C'),('A','inconclusive','C'),
    ('B','pass','D'),('B','fail','C'),('B','inconclusive','C'),('C','pass','D'),
    ('C','fail','STOP_NO_CANDIDATE'),('C','inconclusive','STOP_NO_CANDIDATE'),
    ('D','pass','READY_FOR_SCALE'),('D','fail','STOP_FAILED_CONFIRMATION'),('D','inconclusive','E'),
    ('E','pass','READY_FOR_SCALE'),('E','fail','STOP_FAILED_CONFIRMATION'),('E','inconclusive','STOP_INCONCLUSIVE')])
def test_every_scientific_branch(node,verdict,next_node):
    state=initial_state('plan');state.update(node=node,candidate={'algorithm':'relay','layer':1,'control':'relay_roll'})
    value=transition(state,dict(verdict=verdict,layer=1),'report')
    assert value['node']==next_node
    if node in ('D','E'):assert value['candidate']==state['candidate']

@pytest.mark.parametrize('node',['A','B','C','D','E'])
def test_technical_failure_never_hops_to_alternative(node):
    s=initial_state('x');s['node']=node
    t=transition(s,{'verdict':'technical_failure'},'r')
    assert t['node']=='BLOCKED_TECHNICAL' and t['blocked_node']==node
    retry=retry_technical(t,DEFAULTS);assert retry['node']==node
    retry['node']='BLOCKED_TECHNICAL';retry['blocked_node']=node
    with pytest.raises(ValueError):retry_technical(retry,DEFAULTS)

@pytest.mark.parametrize('terminal',sorted(TERMINAL))
def test_no_success_forcing_cycle(terminal):
    s=initial_state('x');s['node']=terminal
    with pytest.raises(ValueError):transition(s,{'verdict':'pass'},'r')


def rows_for_quality(n=240):
    rows=[]
    for i in range(n):
        # 40 old gains, 40 old harms, plus 40 genuine new improvements.
        b,e,c = (0,1,1) if i<40 else (1,0,1) if i<80 else (0,0,1) if i<120 else (1,1,1)
        rows.append(dict(id=str(i),cohort='source-a' if i%2 else 'source-b',pixel_sha256='image-'+str(i),
            scores=dict(generalist=b,compact=e,candidate=c,control=e),
            arms=dict(candidate=dict(token_ids=[c+2],seconds=.3,bypass=False),compact=dict(token_ids=[e]))))
    return rows


def test_quality_requires_both_controls_and_retained_gains():
    rows=rows_for_quality();cfg=DEFAULTS|{'bootstrap_repetitions':200}
    assert quality(rows,config=cfg)['verdict']=='pass'
    assert quality(rows,config=cfg,confirmation=True)['verdict']=='pass'
    for r in rows:r['scores']['control']=r['scores']['candidate']
    assert quality(rows,config=cfg)['verdict']=='fail'
    assert quality(rows,config=cfg,confirmation=True)['verdict']=='fail'


def test_all_baseline_noop_is_not_success():
    rows=rows_for_quality()
    for r in rows:
        r['arms']['candidate']['token_ids']=r['arms']['compact']['token_ids']
    assert quality(rows,config=DEFAULTS)['verdict']=='fail'


def test_failures_not_imputed_or_dropped():
    rows=rows_for_quality();rows[0]['scores']['candidate']=None
    result=quality(rows,config=DEFAULTS)
    assert result['verdict']=='technical_failure' and result['planned_n']==240
    pred=[dict(id='x',arms={'candidate':{'status':'failed'}})]
    with pytest.raises(ValueError):scored_rows(pred,[dict(id='x',scores={'candidate':0})])
    assert scored_rows(pred,[dict(id='x',scores={'candidate':None})])[0]['scores']['candidate'] is None


def test_confirmation_no_iid_claim_on_repeated_images():
    rows=rows_for_quality()
    for r in rows:r['pixel_sha256']='image-'+str(int(r['id'])//2)
    assert quality(rows,config=DEFAULTS|{'bootstrap_repetitions':200},confirmation=True)['verdict']=='inconclusive'


def test_grouped_bootstrap_reproducible():
    a=paired_interval([1,1,-1],['a','a','b'],repetitions=200,seed=0)
    assert a==paired_interval([1,1,-1],['a','a','b'],repetitions=200,seed=0)
    assert a['groups']==2 and a['mean']==pytest.approx(1/3)
    assert not paired_interval([1],['a'],repetitions=200)['sufficient']


def test_localization_orients_good_and_bad_symmetrically():
    rows=[]
    for i in range(24):
        good=i%2==0
        rows.append(dict(id=str(i),pixel_sha256=str(i),scores=dict(generalist=0 if good else 1,compact=1 if good else 0),
            diagnostic=dict(status='divergence',sites=[dict(layer=1,restore={'probability_margin':-.5 if good else .5},
                                                               roll={'probability_margin':0})])))
    assert localization(rows,DEFAULTS|{'bootstrap_repetitions':200})['verdict']=='pass'
    for r in rows:r['diagnostic']['sites'][0]['restore']['probability_margin']=.5
    assert localization(rows,DEFAULTS|{'bootstrap_repetitions':200})['verdict']!='pass'

@pytest.mark.parametrize('key,value',[('min_delta',-1),('alpha',1),('min_gain_retention',float('nan')),
                                    ('max_node_attempts',99),('max_total_forwards',True)])
def test_invalid_policy_rejected(key,value):
    with pytest.raises(ValueError):validate_policy(DEFAULTS|{key:value})


def test_exclusive_write_and_nan(tmp_path):
    path=tmp_path/'a.json';write(path,{'a':1})
    with pytest.raises(FileExistsError):write(path,{'a':2})
    with pytest.raises(ValueError):write(tmp_path/'bad.json',{'a':float('nan')})
    assert read(path)=={'a':1}


def create_cohort(tmp_path,name,pixel):
    from PIL import Image
    root=tmp_path/name;root.mkdir()
    image=root/'im.png';Image.new('RGB',(2,2),(pixel,0,0)).save(image)
    row=dict(id='case',image=str(image),image_sha256=digest(image),question='q',benchmark_prompt='q')
    proto=dict(identity=name,rows=[row],selection='frozen TRAIN queue',test_image_exclusion_sha256='audit')
    write(root/'protocol.json',proto);write(root/'complete.json',dict(identity=name,n=1))
    write(root/'cases/case.json',dict(id='case',identity=name,complete=True,
          arms={a:dict(token_ids=[1],text='pred') for a in ('generalist','compact')}))
    generation=root/'generation.json';write(generation,{'max_new_tokens':4})
    return dict(name=name,dataset=name,split='train',source_run=str(root),generation_json=str(generation))


def make_plan(tmp_path,overlap=False):
    a=create_cohort(tmp_path,'a',10);b=create_cohort(tmp_path,'b',10 if overlap else 20)
    exposed=tmp_path/'exposed.json';write(exposed,[])
    tests=tmp_path/'test.json';write(tests,['f'*64])
    pin=tmp_path/'pinned.txt';pin.write_text('pin')
    return dict(schema='merit-algorithm-chain-v1',policy=DEFAULTS,development=[a],confirmation=[b],extension=[],
        exposed_pixels=str(exposed),test_pixels=str(tests),runtime={'pins':{str(pin):digest(pin)}},
        scorer={'pins':{str(pin):digest(pin)}})


def test_frozen_plan_rejects_reencoded_image_overlap(tmp_path):
    plan=make_plan(tmp_path,overlap=True)
    with pytest.raises(ValueError,match='Confirmation'):
        freeze_plan(plan,tmp_path)


def test_frozen_plan_identity_and_labelblind_job(tmp_path):
    plan=freeze_plan(make_plan(tmp_path),tmp_path)
    assert plan['identity']==fingerprint({k:v for k,v in plan.items() if k!='identity'})
    job=stage_job(plan,initial_state(plan['identity']))
    assert 'scorer' not in job and 'references' not in job and job['node']=='A'


def test_old_development_cannot_be_renamed_unseen(tmp_path):
    plan=make_plan(tmp_path)
    pixel=source_inventory(plan['confirmation'][0])['records'][0]['pixel_sha256']
    write(plan['exposed_pixels'],[pixel],replace=True)
    with pytest.raises(ValueError,match='Confirmation'):freeze_plan(plan,tmp_path)


def test_no_confirmation_is_blocked_not_auto_split(tmp_path):
    plan=make_plan(tmp_path);plan['confirmation']=[]
    frozen=freeze_plan(plan,tmp_path);state=initial_state(frozen['identity']);state['node']='D'
    with pytest.raises(ValueError,match='No predeclared'):stage_job(frozen,state)


def test_native_size_prefixed_pixel_hash_preserves_raw_registry(tmp_path):
    from merit_feddg.open_data import pixel_digest
    cohort = create_cohort(tmp_path, 'native', 30)
    path = Path(cohort['source_run'])/'protocol.json'
    protocol = read(path)
    image = protocol['rows'][0]['image']
    protocol['rows'][0]['pixel_sha256'] = pixel_digest(image)
    write(path, protocol, replace=True)
    record = source_inventory(cohort)['records'][0]
    assert record['pixel_sha256'] == pixel_hash(image)[0]
    assert record['source_pixel_scheme'] == 'size_prefixed_rgb'
    protocol['rows'][0]['pixel_sha256'] = 'f'*64
    write(path, protocol, replace=True)
    with pytest.raises(ValueError, match='pixels'):
        source_inventory(cohort)
