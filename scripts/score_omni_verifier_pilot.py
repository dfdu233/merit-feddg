"""Score all frozen verifier arms on the exact Omni pilot IDs."""
import hashlib
import argparse
import json
import subprocess
import sys
import types
from pathlib import Path

src=subprocess.check_output(['ssh','anchor-autodl','cat /tmp/anchor_v14_omni_progress_audit.py'],text=True)
assert hashlib.sha256(src.encode()).hexdigest()=='69e5a7dcac60113ed73a366d19e93f0e4341de49c6580a2a93f94a4b97dce381'
sys.path.insert(0,'/home/dbw/ANCHOR')
m=types.ModuleType('frozen_omni');m.__file__='/home/dbw/ANCHOR/anchor/corrected_sgta/evaluate_medheval_answers.py';sys.modules[m.__name__]=m
exec(compile(src,m.__file__,'exec'),m.__dict__)
parser=argparse.ArgumentParser();parser.add_argument('--run-dir',default='runs/test-omni256');parser.add_argument('--report-dir',default='reports/test-omni256');args=parser.parse_args()
r=Path(args.run_dir);inputs=json.loads((r/'inputs.json').read_text());ids={x['id'] for x in inputs}
cases={p.stem:json.loads(p.read_text()) for p in (r/'cases').glob('*.json')};assert set(cases)==ids and len(ids)==len(inputs)
pilot_ids={x['id'] for x in json.loads(Path('runs/test-omni256/inputs.json').read_text())};new_ids=ids-pilot_ids
refs=m._load_questions(Path('/home/dbw/ANCHOR/data/omnimedvqa/eight_modality_full_v1.json'));refs={k:refs[k] for k in ids}
out={'n':len(ids),'protocol':m.PROTOCOL_VERSION,'partial_test':True,'unqualified_exploratory':True,'arms':{}}
correct={}
for arm in ['baseline','candidate','biomedclip_claim_verifier','conch_claim_verifier','plip_claim_verifier','all_three']:
    preds=[];accepted=0
    for x in inputs:
        c=cases[x['id']];vals=[v['differential_effect'] for e,v in c['effects'].items() if arm=='all_three' or arm==e]
        use=False if arm=='baseline' else c['candidate']!=c['baseline'] if arm=='candidate' else any(v>0 for v in vals) and not any(v<0 for v in vals)
        accepted+=use;preds.append({'question_id':x['id'],'text':c['candidate'] if use else c['baseline']})
    result=m.evaluate_rows(m.align_answers_with_questions(preds,refs));correct[arm]={x['question_id']:x['correct'] for x in result['details']}
    out['arms'][arm]={**result['decoded_strict'],'accepted':accepted,'corrected':sum(correct[arm][k] and not correct['baseline'][k] for k in ids),'broken':sum(not correct[arm][k] and correct['baseline'][k] for k in ids)}
    out['arms'][arm].update(new_n=len(new_ids),new_correct=sum(correct[arm][k] for k in new_ids),new_accuracy=sum(correct[arm][k] for k in new_ids)/len(new_ids) if new_ids else None)
out['changed_cases']=[c for c in cases.values() if c['baseline']!=c['candidate']]
if len(ids)>256:
    import numpy as np
    groups={}
    for row in inputs:groups.setdefault(row['image_sha256'],[]).append(row['id'])
    clusters=list(groups.values());sizes=np.array([len(g) for g in clusters]);rng=np.random.default_rng(20260922)
    deltas={ref:np.array([sum(int(correct['all_three'][k])-int(correct[ref][k]) for k in g) for g in clusters]) for ref in ['baseline','candidate']}
    draws={ref:[] for ref in deltas}
    for _ in range(2000):
        sample=rng.integers(len(clusters),size=len(clusters));denom=sizes[sample].sum()
        for ref,delta in deltas.items():draws[ref].append(100*delta[sample].sum()/denom)
    out['joint_paired_image_cluster_bootstrap']={'replicates':2000,'seed':20260922,'clusters':len(clusters),'qualification':'Exploratory marginal intervals on fixed completed pool; not multiplicity-adjusted or whole-test inference','ci95_delta_percentage_points':{ref:np.quantile(values,[0.025,0.975]).tolist() for ref,values in draws.items()}}
dest=Path(args.report_dir);dest.mkdir(exist_ok=True);(dest/'results.json').write_text(json.dumps(out,indent=2))
print(json.dumps({k:v for k,v in out.items() if k!='changed_cases'},indent=2))
