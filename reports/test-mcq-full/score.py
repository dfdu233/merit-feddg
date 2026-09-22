import json,sys,subprocess
from pathlib import Path
sys.path.insert(0,'/home/dbw/ANCHOR')
from anchor.corrected_sgta.evaluate_medheval_answers import _load_questions,align_answers_with_questions,evaluate_rows
r=Path('runs/test-mcq-full');inputs=json.load(open(r/'inputs.json'));cases={x['id']:json.load(open(r/'cases'/(x['id']+'.json'))) for x in inputs};assert len(cases)==43930
pilot_ids={x['id'] for x in json.load(open('runs/test-mcq/inputs.json'))}
methods=['baseline','candidate','biomedclip_claim_verifier','conch_claim_verifier','plip_claim_verifier','all_three'];out={};medical=('test_Basic_Medical_Science_','test_Clinical_Medicine_','test_Diagnostics_and_Laboratory_Medicine_','test_Pharmacy_','test_Public_Health_')
for dataset in ['pmcvqa','mmmu']:
 rows=[x for x in inputs if x['dataset']==dataset];ids={x['id'] for x in rows};refpath=Path('/home/dbw/ANCHOR/data')/dataset/('official_v2_test_v1.json' if dataset=='pmcvqa' else 'official_test_full_v1.json');refs=_load_questions(refpath);refs={k:v for k,v in refs.items() if k in ids}
 if dataset=='mmmu':(r/'mmmu-reference-subset.json').write_text(json.dumps([x for x in json.load(open(refpath)) if x['id'] in ids]))
 for scope in (['original_route','medical_subject_scope'] if dataset=='mmmu' else ['original_route']):
  arms={};correct_maps={}
  for m in methods:
   preds=[];accepted=0
   for x in rows:
    case=cases[x['id']];vals=[v['differential_effect'] for e,v in case['effects'].items() if m=='all_three' or m==e];use=False if m=='baseline' else case['candidate']!=case['baseline'] if m=='candidate' else any(v>0 for v in vals) and not any(v<0 for v in vals)
    if scope=='medical_subject_scope' and m not in ['baseline','candidate'] and not x['id'].startswith(medical):use=False
    accepted+=use;preds.append({'question_id':x['id'],'text':case['candidate'] if use else case['baseline']})
   if dataset=='pmcvqa':
    details=evaluate_rows(align_answers_with_questions(preds,refs))['details'];acc=sum(bool(x['correct']) for x in details)/len(details);extra={};correct_map={x['question_id']:bool(d['correct']) for x,d in zip(preds,details)}
   else:
    ans=r/f'{scope}-{m}.jsonl';ans.write_text(''.join(json.dumps(x)+'\n' for x in preds));dest=r/f'{scope}-{m}-score.json'
    subprocess.run([sys.executable,'/home/dbw/ANCHOR/anchor/medeval/evaluate_mmmu_official.py','--manifest',str(r/'mmmu-reference-subset.json'),'--answers',str(ans),'--output',str(dest),'--exact-option-label-repair'],check=True,stdout=subprocess.DEVNULL)
    payload=json.load(open(dest));acc=payload['strict_no_random_fallback']['score'];
    if acc is None:
     print('SCORE_KEYS',payload.keys());raise RuntimeError('inspect accuracy field')
    extra={};correct_map={k:v=='Correct' for k,v in payload['strict_no_random_fallback']['judgments'].items()}
   correct_maps[m]=correct_map
   new_ids=ids-pilot_ids
   arms[m]={'n':len(rows),'accuracy':acc,'accepted':accepted,'new_n':len(new_ids),'new_accuracy':sum(correct_map[k] for k in new_ids)/len(new_ids),**extra}
  for m,v in correct_maps.items():
   arms[m]['corrected']=sum(v[k] and not correct_maps['baseline'][k] for k in ids)
   arms[m]['broken']=sum(not v[k] and correct_maps['baseline'][k] for k in ids)
  out[dataset+'_'+scope]=arms
Path('reports/test-mcq-full/results.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
