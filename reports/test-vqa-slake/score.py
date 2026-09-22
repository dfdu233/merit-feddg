import json,sys,collections
from pathlib import Path
import numpy as np
sys.path.insert(0,'/home/dbw/ANCHOR')
from anchor.corrected_sgta.evaluate_medheval_answers import _load_questions,align_answers_with_questions,evaluate_rows,PROTOCOL_VERSION
from anchor.medeval.evaluate_mixed_vqa_table import score,source_task_group,answer_token_recall
r=Path('runs/test-vqa-slake');inputs=json.load(open(r/'inputs.json'));assert len(inputs)==2545;cases={x['id']:json.load(open(r/'cases'/(x['id']+'.json'))) for x in inputs};refs=_load_questions(Path('/home/dbw/ANCHOR/data/vqa_rad/official_test_full_v1.json'))
legacy={x['question_id']:x for x in map(json.loads,open('/home/dbw/ANCHOR/corrected_runs/paper_baselines_v1/huatuo_final_v1/full/slake/greedy/answers.jsonl'))};cached={};adjusted=[];answer_differences=0
for x in inputs:
 if x['dataset']!='slake':continue
 p=Path(x['image']).parent/'question.json'
 if p not in cached:cached[p]=json.load(open(p))
 norm=lambda q:q.strip().casefold().replace('exsit','exist')
 match=[q for q in cached[p] if norm(q['question'])==norm(x['question'])];assert len(match)==1
 q=match[0]
 if q['question'].strip().casefold()!=x['question'].strip().casefold():adjusted.append(x['id'])
 gt=legacy[x['id']]['gt_ans'];answer_differences+=str(gt).strip().casefold()!=str(q['answer']).strip().casefold()
 refs[x['id']]={**x,'answer':gt,'answer_type':q['answer_type'].lower(),'source_question_type':'binary' if q['answer_type'].lower()=='closed' else 'open','q_lang':q['q_lang']}
methods=['baseline','candidate','biomedclip_claim_verifier','conch_claim_verifier','plip_claim_verifier','all_three'];pred={m:{} for m in methods};accepted={m:{} for m in methods}
for sid,x in cases.items():
 for m in methods:
  vals=[v['differential_effect'] for e,v in x['effects'].items() if m=='all_three' or e==m]
  use=False if m=='baseline' else x['candidate']!=x['baseline'] if m=='candidate' else any(v>0 for v in vals) and not any(v<0 for v in vals)
  pred[m][sid]=x['candidate'] if use else x['baseline'];accepted[m][sid]=use
out={'protocol':PROTOCOL_VERSION,'slake_metadata_typo_corrected_ids':adjusted,'slake_historical_reference_vs_native_answer_differences':answer_differences,'datasets':{}}
for dataset in ['vqa_rad','slake','slake_en','slake_zh']:
 ids=[x['id'] for x in inputs if x['dataset']==dataset or dataset.startswith('slake_') and x['dataset']=='slake' and refs[x['id']]['q_lang']==dataset.split('_')[1]]
 arms={};vectors={};subset={sid:refs[sid] for sid in ids}
 for m in methods:
  rows=align_answers_with_questions([{'question_id':sid,'text':pred[m][sid]} for sid in ids],subset);arms[m]={'scores':score(rows),'accepted':sum(accepted[m][sid] for sid in ids)};details=evaluate_rows(rows)['details'];vectors[m]=np.array([float(d['correct']) if source_task_group(row)=='ce' else answer_token_recall(row['text'],row['gt_ans']) for row,d in zip(rows,details)])
 image={x['id']:x['image_sha256'] for x in inputs};groups=[[i for i,sid in enumerate(ids) if image[sid]==im] for im in sorted({image[sid] for sid in ids})];counts=np.array([len(g) for g in groups]);draws=np.random.default_rng(20260922).integers(0,len(groups),size=(3000,len(groups)))
 for m,v in vectors.items():
  delta=v-vectors['baseline'];sums=np.array([delta[g].sum() for g in groups]);boot=sums[draws].sum(1)/counts[draws].sum(1)*100;arms[m].update(improved=int((delta>1e-12).sum()),harmed=int((delta< -1e-12).sum()),image_cluster_ci95_pp=np.quantile(boot,[.025,.975]).tolist())
 out['datasets'][dataset]={'n':len(ids),'arms':arms};print(dataset,{m:round(a['scores']['primary_unified']['score']*100,4) for m,a in arms.items()})
out['failure_counts']=dict(collections.Counter((e+':'+v) for x in cases.values() for e,v in x['errors'].items()));Path('reports/test-vqa-slake/results.json').write_text(json.dumps(out,indent=2))
