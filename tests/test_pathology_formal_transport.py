"""Exercise the formal adapter's actual delivery branch without loading a GPU.

The formal script intentionally imports a separate, pinned benchmark checkout.
Extract its unchanged branch AST to avoid replacing the test suite's package.
"""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/run_pathology_quilt_formal.py'


def run_branch(*, policy, delivered, reason='token_budget', displaced=False,
               changed_prompt=False, missing_text=False):
    import json
    tree = ast.parse(SCRIPT.read_text())
    actor = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'actor')
    loop = next(n for n in ast.walk(actor) if isinstance(n, ast.For)
                and isinstance(n.target, ast.Tuple)
                and [v.id for v in n.target.elts] == ['variant', 'arm'])
    item = SimpleNamespace(expert_id='new', evidence_id='new-id')
    old = ('old', 'old-id')
    sources = ([] if displaced else [old]) + ([('new', 'new-id')] if delivered else [])
    transport = {'prompt_sha256':'original', 'evidence_sha256':'original-evidence'}
    t = dict(transport, presented=[dict(expert_id=a,evidence_id=b) for a,b in sources],
             omitted=[] if delivered else [dict(expert_id='new',evidence_id='new-id',reason=reason)])
    if changed_prompt:
        t['prompt_sha256'] = 'changed'
    jobs = {('case',v): {'key':v} for v in ('matched','wrong_image')}
    predictions = {v: {'job':j,'text':'observation','token_ids':[1],'seconds':1}
                   for (_,v),j in jobs.items()}
    calls = []
    def answer(*args):
        calls.append(args)
        return {'text':'candidate'}
    env = dict(json=json, frozen={'transport_policy':policy}, jobs=jobs, key='case',
               predictions=predictions, quilt_item=lambda _:item, kept=(),
               context=lambda _:('image','missing' if missing_text else 'observation',t),
               visible=lambda x:{(r['expert_id'],r['evidence_id']) for r in x['presented']},
               allowed={old}, delivery={}, arms={}, transports={}, transport=transport,
               incumbent={'text':'incumbent'}, calls=0, answer=answer)
    exec(compile(ast.Module(body=[loop],type_ignores=[]), str(SCRIPT), 'exec'),env)
    return env, calls


def test_budgeted_omission_reuses_only_exact_incumbent():
    env,calls = run_branch(policy='formal-budgeted',delivered=False)
    assert not calls and env['calls'] == 0
    assert env['delivery'] == {'compact_quilt':False,'compact_wrong_image':False}
    assert env['arms']['compact_quilt']['reused_incumbent']


@pytest.mark.parametrize('policy',['strict','formal-budgeted'])
def test_real_delivery_generates_both_candidates(policy):
    env,calls = run_branch(policy=policy,delivered=True)
    assert len(calls) == env['calls'] == 2
    assert all(env['delivery'].values())


@pytest.mark.parametrize('kwargs',[
    dict(policy='strict',delivered=False),
    dict(policy='formal-budgeted',delivered=False,reason='unknown'),
    dict(policy='formal-budgeted',delivered=False,changed_prompt=True),
    dict(policy='formal-budgeted',delivered=True,displaced=True),
    dict(policy='formal-budgeted',delivered=True,missing_text=True),
])
def test_unsafe_changes_still_stop(kwargs):
    with pytest.raises(AssertionError):
        run_branch(**kwargs)
