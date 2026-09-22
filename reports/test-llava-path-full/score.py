import json,sys
from pathlib import Path
sys.path.insert(0,'/home/dbw/ANCHOR')
from anchor.medeval.evaluate_mixed_vqa_table import score,source_task_group,answer_token_recall
from anchor.corrected_sgta.evaluate_medheval_answers import _load_questions,align_answers_with_questions,evaluate_rows,PROTOCOL_VERSION
r=Path('runs/test-llava-path-full');inputs=json.load(open(r/'inputs.json'));cases={x['id']:json.load(open(r/'cases'/f"{x['id']}.json")) for x in inputs};assert len(cases)==6719
refs=_load_questions(Path('/home/dbw/ANCHOR/data/pathvqa/official_test_v1.json'));refs={k:refs[k] for k in cases};methods={'baseline':{},'candidate':{},'biomedclip_claim_verifier':{},'conch_claim_verifier':{},'plip_claim_verifier':{},'all_three':{}};accepted={k:0 for k in methods}
for sid,x in cases.items():
 for name,outputs in methods.items():
  if name=='baseline':use=False
  elif name=='candidate':use=x['candidate']!=x['baseline']
  else:
   vals=[v['differential_effect'] for e,v in x['effects'].items() if name=='all_three' or name==e];use=any(v>0 for v in vals) and not any(v<0 for v in vals)
  outputs[sid]=x['candidate'] if use else x['baseline'];accepted[name]+=use
result={'scope':'full6719 PathVQA TEST exploratory postprocessing; not qualified v3','protocol':PROTOCOL_VERSION,'arms':{}};vectors={}
for name,pred in methods.items():
 rows=align_answers_with_questions([{'question_id':k,'text':v} for k,v in pred.items()],refs);result['arms'][name]={'scores':score(rows),'accepted':accepted[name]};details=evaluate_rows(rows)['details'];vectors[name]=[float(d['correct']) if source_task_group(row)=='ce' else answer_token_recall(row['text'],row['gt_ans']) for row,d in zip(rows,details)]
for name,v in vectors.items():result['arms'][name]['paired_vs_baseline']={'improved':sum(x>y+1e-12 for x,y in zip(v,vectors['baseline'])),'harmed':sum(x<y-1e-12 for x,y in zip(v,vectors['baseline']))}
import numpy as np
image_by_id={x['id']:x['image_sha256'] for x in inputs}
ordered=list(cases);images=sorted(set(image_by_id.values()));groups=[[i for i,sid in enumerate(ordered) if image_by_id[sid]==im] for im in images]
rng=np.random.default_rng(20260922);draws=rng.integers(0,len(groups),size=(3000,len(groups)));counts=np.array([len(g) for g in groups])
for name,v in vectors.items():
 if name=='baseline':continue
 delta=np.array(v)-np.array(vectors['baseline']);sums=np.array([delta[g].sum() for g in groups]);boot=sums[draws].sum(1)/counts[draws].sum(1)*100
 result['arms'][name]['image_cluster_gain_ci95_pp']=np.quantile(boot,[.025,.975]).tolist()
result['image_clusters']=len(groups)
prior_ids={x['id'] for x in json.load(open('runs/test256/inputs.json'))}
result['subsets']={}
for subset,indices in [('previous256',[i for i,sid in enumerate(ordered) if sid in prior_ids]),('new6463',[i for i,sid in enumerate(ordered) if sid not in prior_ids]),('pathology',[i for i,x in enumerate(inputs) if x['modality']=='pathology'])]:
 result['subsets'][subset]={'n':len(indices),'scores':{name:sum(v[i] for i in indices)/len(indices) for name,v in vectors.items()}}
result['failure_counts']={}
from collections import Counter
for expert in ['biomedclip_claim_verifier','conch_claim_verifier','plip_claim_verifier']:
 counts=Counter(x['errors'].get(expert,'available-or-no-transaction') for x in cases.values());result['failure_counts'][expert]=dict(counts)
Path('reports/test-llava-path-full/results.json').write_text(json.dumps(result,indent=2));print({k:(round(v['scores']['primary_unified']['score']*100,4),v['accepted'],v['paired_vs_baseline']) for k,v in result['arms'].items()})
