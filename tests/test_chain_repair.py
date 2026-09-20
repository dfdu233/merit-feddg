import pytest
from merit_feddg.algorithm_chain import native
from merit_feddg.algorithm_chain.policy import DEFAULTS,validate_policy,retry_technical


def test_three_attempt_amendment_is_explicit_and_bounded():
    assert DEFAULTS['max_node_attempts']==2
    amended=DEFAULTS|{'max_node_attempts':3}
    validate_policy(amended)
    state=dict(node='BLOCKED_TECHNICAL',blocked_node='A',attempts={'A':2},forwards_used=1046,history=['old'])
    updated=retry_technical(state,amended)
    assert updated['attempts']=={'A':3} and updated['forwards_used']==1046 and updated['history']==['old']
    with pytest.raises(ValueError):retry_technical(updated|{'node':'BLOCKED_TECHNICAL'},amended)
    with pytest.raises(ValueError):validate_policy(DEFAULTS|{'max_node_attempts':4})


@pytest.mark.parametrize('kind,name,memory,allowed_ok',[
    ('C+G','/usr/bin/nautilus','34 MiB',True),
    ('C','/usr/bin/nautilus','34 MiB',False),
    ('C+G','/usr/bin/python','34 MiB',False),
    ('C+G','/usr/bin/nautilus','65 MiB',False)])
def test_display_exception_is_exact(monkeypatch,kind,name,memory,allowed_ok):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','GPU-test')
    def query(argv,**kw):
        if argv[1]=='--query-gpu=uuid':return 'GPU-test\n'
        if argv[1]=='--query-compute-apps=gpu_uuid,pid':return 'GPU-test, 42\n'
        return f'<nvidia_smi_log><gpu><uuid>GPU-test</uuid><processes><process_info><pid>42</pid><type>{kind}</type><process_name>{name}</process_name><used_memory>{memory}</used_memory></process_info></processes></gpu></nvidia_smi_log>'
    monkeypatch.setattr(native.subprocess,'check_output',query)
    args=('GPU-test',[dict(pid=42,process_name='/usr/bin/nautilus',max_memory_mib=64)])
    if allowed_ok:native.check_device(*args)
    else:
        with pytest.raises(RuntimeError):native.check_device(*args)
    with pytest.raises(RuntimeError):native.check_device('GPU-test')


def test_prefix_rebuild_keeps_native_shapes_and_only_patches_final_query():
    import torch
    from test_algorithm_chain import Tiny,prompt
    from merit_feddg.huatuo_pathway import _Stream
    from merit_feddg.algorithm_chain.decoding import ResidualRelay,_replay_logits
    torch.manual_seed(5)
    model=Tiny().eval().requires_grad_(False)
    prepared=prompt(5,1);prefix=[3,4]
    with torch.inference_mode():
        incremental=_Stream(model,prepared)
        for end in range(3): expected=incremental.advance(prefix[:end])
        with ResidualRelay(model,(1,3)) as relay:
            observed=[]
            hook=model.layers[1].register_forward_hook(lambda m,a,o:observed.append((o[0].shape[1],relay.mode)))
            actual=_replay_logits(model,prepared,prefix,relay,'reference',2)
            assert observed==[(5,None),(1,None),(1,'reference')]
            assert torch.equal(expected,actual)
            observed.clear()
            _replay_logits(model,prepared,prefix,relay,'restore',2,1)
            assert observed==[(5,None),(1,None),(1,'restore')]
            assert len(relay.events)==1 and relay.events[0]['step']==2
            hook.remove()
    assert not model.layers[1]._forward_hooks


def test_diagnostic_counts_every_rebuilt_prefix_forward():
    import torch
    from test_algorithm_chain import Tiny,prompt,policy
    from merit_feddg.algorithm_chain.decoding import diagnose
    torch.manual_seed(5)
    model=Tiny().eval().requires_grad_(False)
    calls=[]
    handle=model.register_forward_pre_hook(lambda m,a:calls.append(1))
    result=diagnose(model,prompt(5,1),prompt(8,2),policy(),context_limit=40)
    handle.remove()
    assert result['forwards']==len(calls)
    assert result['replay_mode']=='native_incremental_prefix_rebuild'


def test_repair_migration_inherits_budget_and_requires_standard_retry(tmp_path,monkeypatch):
    import importlib.util
    from pathlib import Path
    from merit_feddg.algorithm_chain.storage import read,write,digest,fingerprint
    spec=importlib.util.spec_from_file_location('repair_migration',Path(__file__).resolve().parents[1]/'scripts/migrate_chain_repair.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    old=dict(schema='merit-algorithm-chain-v1',development=[{'name':'dev'}],confirmation=[],extension=[],
        exposed_pixels='exposed',test_pixels='test',scorer={'pins':{'f':'x'}},runtime={'gpu_uuid':'GPU-old','pins':{'m':'h'}},
        policy=DEFAULTS,inventory={'dev':1},generation={'dev':{}},generation_pins={'g':'a'},registry_pins={'r':'b'},code_pins={'c':'old'})
    old['identity']=fingerprint(old)
    source=tmp_path/'old';source.mkdir();write(source/'plan.json',old)
    history=[]
    for n in (1,2):
        directory=source/f'A-attempt-{n}';directory.mkdir();write(directory/'decision.json',{'verdict':'technical_failure'})
        write(directory/'budget.json',{'used':0 if n==1 else 1046})
        history.append(dict(node='A',verdict='technical_failure',directory=directory.name,report_sha=digest(directory/'decision.json')))
    state=dict(plan_sha=old['identity'],node='BLOCKED_TECHNICAL',blocked_node='A',attempts={'A':2},forwards_used=1046,holdout_consumed=False,history=history,runtime_sha='old')
    write(source/'state.json',state)
    probe=tmp_path/'probe';probe.mkdir()
    write(probe/'summary.json',dict(phase='A-attempt-3-technical-preflight',forwards=28))
    write(probe/'budget.json',dict(used=28))
    amended=old|{'policy':DEFAULTS|{'max_node_attempts':3}}
    inp=tmp_path/'input.json';write(inp,amended)
    monkeypatch.setattr(module,'freeze_plan',lambda p,r:p|{'identity':'successor'})
    dest=tmp_path/'new';module.migrate(source,dest,inp,probe)
    inherited=read(dest/'state.json')
    assert inherited['node']=='BLOCKED_TECHNICAL' and inherited['attempts']=={'A':2}
    assert inherited['forwards_used']==1074 and inherited['history']==history
    assert 'runtime_sha' not in inherited
    assert read(source/'state.json')==state
    resumed=retry_technical(inherited,amended['policy']);assert resumed['attempts']=={'A':3}
    altered=amended|{'policy':amended['policy']|{'min_delta':0.001}}
    write(inp,altered,replace=True)
    with pytest.raises(ValueError,match='Only authorized attempt ceiling'):module.migrate(source,tmp_path/'bad',inp,probe)
