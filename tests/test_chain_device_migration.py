"""Device migration never creates retry budget or discards prior evidence."""
import importlib.util
from pathlib import Path
import pytest
from merit_feddg.algorithm_chain.policy import DEFAULTS, retry_technical
from merit_feddg.algorithm_chain.storage import digest, read, write

spec = importlib.util.spec_from_file_location('migration', Path(__file__).resolve().parents[1] / 'scripts/migrate_chain_device.py')
migration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(migration)


def fixture(tmp_path, monkeypatch):
    source = tmp_path / 'old'
    source.mkdir()
    previous = source / 'A-attempt-1'
    previous.mkdir()
    write(previous / 'decision.json', {'verdict': 'technical_failure'})
    (previous / 'inference.log').write_text('Another compute process owns the authorized GPU')
    plan = dict(identity='old-plan', runtime={'gpu_uuid': 'GPU-old'}, policy=DEFAULTS, inventory={'development': [1]}, code_pins={'original': 'unchanged'})
    state = dict(plan_sha='old-plan', node='BLOCKED_TECHNICAL', blocked_node='A', attempts={'A': 1}, forwards_used=0,
                 holdout_consumed=False, history=[dict(directory='A-attempt-1', verdict='technical_failure', report_sha=digest(previous / 'decision.json'))])
    write(source / 'plan.json', plan)
    write(source / 'state.json', state)
    monkeypatch.setattr(migration, 'verify_plan', lambda p: None)
    return source, tmp_path / 'new', plan, state


def test_inherits_blocked_state_then_standard_retry(tmp_path, monkeypatch):
    source, output, plan, state = fixture(tmp_path, monkeypatch)
    migration.migrate(source, output, 'GPU-new')
    assert read(source / 'state.json') == state
    assert read(source / 'plan.json') == plan
    inherited = read(output / 'state.json')
    assert inherited['node'] == 'BLOCKED_TECHNICAL'
    assert inherited['history'] == state['history']
    assert inherited['attempts'] == {'A': 1}
    successor = read(output / 'plan.json')
    assert successor['code_pins'] == plan['code_pins']
    assert successor['inventory'] == plan['inventory']
    assert successor['runtime']['gpu_uuid'] == 'GPU-new'
    resumed = retry_technical(inherited, DEFAULTS)
    assert resumed['attempts'] == {'A': 2}
    resumed.update(node='BLOCKED_TECHNICAL', blocked_node='A')
    with pytest.raises(ValueError):
        retry_technical(resumed, DEFAULTS)
    with pytest.raises(ValueError):
        migration.migrate(source, output, 'GPU-new')


@pytest.mark.parametrize('change', ['forwards', 'runtime', 'holdout', 'attempts'])
def test_rejects_consumed_or_unexpected_history(tmp_path, monkeypatch, change):
    source, output, _, state = fixture(tmp_path, monkeypatch)
    if change == 'forwards': state['forwards_used'] = 1
    if change == 'runtime': write(source / 'A-attempt-1/runtime.json', {'loaded': True})
    if change == 'holdout': state['holdout_consumed'] = True
    if change == 'attempts': state['attempts']['A'] = 2
    write(source / 'state.json', state, replace=True)
    with pytest.raises(ValueError):
        migration.migrate(source, output, 'GPU-new')
    assert not output.exists()


def test_expansion_resource_migration_retains_all_prior_costs(tmp_path,monkeypatch):
    source=tmp_path/'expansion';source.mkdir()
    plan=dict(identity='old',experiment_mode='fixed_candidate_exploratory_expansion',runtime={'gpu_uuid':'GPU-old'},policy=DEFAULTS|{'max_node_attempts':3},code_pins={})
    history=[]
    for n in (1,2):
        d=source/f'C-attempt-{n}';d.mkdir()
        verdict='fail' if n==1 else 'technical_failure'
        write(d/'decision.json',{'verdict':verdict})
        if n==1:write(d/'budget.json',{'used':11})
        else:(d/'inference.log').write_text('Another compute process owns the authorized GPU')
        history.append(dict(node='C',directory=d.name,verdict=verdict,report_sha=digest(d/'decision.json')))
    state=dict(plan_sha='old',node='BLOCKED_TECHNICAL',blocked_node='C',attempts={'A':3,'C':2},forwards_used=11,holdout_consumed=False,history=history)
    write(source/'plan.json',plan);write(source/'state.json',state)
    monkeypatch.setattr(migration,'verify_plan',lambda p:None)
    target=tmp_path/'new';migration.migrate(source,target,'GPU-new',[dict(pid=42,process_name='/usr/bin/nautilus',max_memory_mib=64)])
    inherited=read(target/'state.json')
    assert inherited['forwards_used']==11 and inherited['history']==history
    assert inherited['attempts']=={'A':3,'C':2} and inherited['node']=='BLOCKED_TECHNICAL'
    assert retry_technical(inherited,plan['policy'])['attempts']['C']==3
    assert read(source/'state.json')==state
    assert read(target/'plan.json')['runtime']['allowed_display_contexts'][0]['pid']==42
    write(source/'state.json',state|{'forwards_used':12},replace=True)
    with pytest.raises(ValueError,match='budget mismatch'):migration.migrate(source,tmp_path/'bad','GPU-new')
