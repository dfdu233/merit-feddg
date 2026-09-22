import json,gzip,hashlib,re,shutil
from pathlib import Path
old=Path('/home/dbw/merit-feddg-huatuo-spatial');anchor=Path('/home/dbw/ANCHOR/corrected_runs/paper_baselines_v1/huatuo_final_v1');allrows=[];audit={}
for dataset in ['pmcvqa','mmmu']:
 rows=[json.loads(x) for x in (old/f'runs/llava-test-v1/{dataset}-manifest.jsonl').read_text().splitlines()];rows=sorted(rows,key=lambda x:hashlib.sha256(x['id'].encode()).hexdigest())
 route=json.load(open(old/f'runs/llava-test-v1/{dataset}-routing.json'))
 bp=anchor/('full/pmcvqa/greedy/answers.jsonl' if dataset=='pmcvqa' else 'oldcloud_20260919/full/mmmu/greedy/answers.jsonl');base={x['question_id']:x for x in map(json.loads,bp.read_text().splitlines())}
 cand={x['id']:x for x in json.load(gzip.open(old/'reports/huatuo-test-v1/pmcvqa-full-raw-answers.json.gz'))} if dataset=='pmcvqa' else {x['question_id']:x for x in map(json.loads,(old/'reports/huatuo-test-v1/mmmu-full-answers-20260922.jsonl').read_text().splitlines())}
 assert len(rows)==(33430 if dataset=='pmcvqa' else 10500)
 skips=0
 for x in rows:
  sid=x['id'];b=base[sid];c=cand[sid];assert b['metadata']['effective_img_name']==x['image'];assert (c.get('image_sha256') or c.get('metadata',{}).get('image_sha256'))==x['image_sha256']
  x.update(modality=route[sid]['modality'],group_id=x['image_sha256'],baseline=b['text'],candidate=c['text'])
  options=dict(re.findall(r'^([A-Z])\. (.+)$',x['benchmark_prompt'],re.M))
  for key in ['baseline','candidate']:
   match=re.fullmatch(r'\s*\(?([A-Z])\)?[.。]?\s*',x[key])
   if match and match[1] in options:x['evidence_'+key]=options[match[1]]
   else:x['verification_skip']='answer-not-an-unambiguous-option-label'
  skips+=bool(x.get('verification_skip'));allrows.append(x)
 audit[dataset]={'baseline_source':str(bp),'n':len(rows),'unmappable':skips}
byid={x['id']:x for x in allrows};assert len(byid)==43930
for x in json.load(open('runs/test-mcq/inputs.json')):
 assert x==byid[x['id']]
 shutil.copy(Path('runs/test-mcq/cases')/(x['id']+'.json'),Path('runs/test-mcq-full/cases')/(x['id']+'.json'))
Path('runs/test-mcq-full/inputs.json').write_text(json.dumps(allrows));Path('reports/test-mcq-full/input-audit.json').write_text(json.dumps(audit,indent=2));print(audit)
