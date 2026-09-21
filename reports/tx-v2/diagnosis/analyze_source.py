import json,sys
from pathlib import Path
from collections import defaultdict
sys.path.insert(0,str(Path.cwd()))
from merit_feddg.contribution import answer_metrics, normalized_tokens
r=Path('runs/tx-v2'); out={}; examples=[]
def stats(rows):
 n=len(rows)
 return dict(n=n,improved=sum(x>1e-12 for x in rows),harmed=sum(x< -1e-12 for x in rows),unchanged=sum(abs(x)<=1e-12 for x in rows),mean_gain=sum(rows)/n if n else None)
for model in ['llava','huatuo']:
 base=json.load(open(r/f'{model}-qualification-baseline.json')); cand=json.load(open(r/f'{model}-qualification-candidate.json')); refs=json.load(open(r/'qualification-references.json'))
 manifest={x['id']:x for x in map(json.loads,(r/'qualification-routed.jsonl').read_text().splitlines())}
 groups=defaultdict(list)
 for sid in base:
  ref=refs[sid];ref=[ref] if isinstance(ref,str) else ref
  b=answer_metrics(base[sid]['text'],ref)['token_f1'];c=answer_metrics(cand[sid]['text'],ref)['token_f1']
  for group in ['all','pathvqa' if sid.startswith('pathvqa') else 'pathorob']:groups[group].append((b,c))
  if abs(c-b)>1e-12:examples.append(dict(model=model,id=sid,question=manifest[sid]['question'],references=ref,baseline=base[sid]['text'],candidate=cand[sid]['text'],gain=c-b))
 summary={g:dict(stats([c-b for b,c in vals]),baseline=sum(b for b,c in vals)/len(vals),candidate=sum(c for b,c in vals)/len(vals),oracle=sum(max(b,c) for b,c in vals)/len(vals)) for g,vals in groups.items()}
 obs=[json.loads(x) for x in (r/f'{model}-source-observations.jsonl').read_text().splitlines()]; experts={}
 for e in sorted({x['expert_id'] for x in obs}):
  rows=[x for x in obs if x['expert_id']==e]; selected={}
  for name,pred in [('D_positive',lambda x:x['differential_effect']>0),('real_positive',lambda x:x['real_effect']>0),('both_positive',lambda x:x['real_effect']>0 and x['differential_effect']>0),('D_positive_real_nonpositive',lambda x:x['differential_effect']>0 and x['real_effect']<=0)]:
   selected[name]=stats([x['outcome_delta'] for x in rows if pred(x)])
  good=[x for x in rows if x['outcome_delta']>1e-12];bad=[x for x in rows if x['outcome_delta']< -1e-12]
  auc={}
  for key in ['differential_effect','real_effect']:
   auc[key]=sum((a[key]>b[key])+.5*(a[key]==b[key]) for a in good for b in bad)/(len(good)*len(bad)) if good and bad else None
  experts[e]=dict(n=len(rows),actions=selected,auc_improved_vs_harmed=auc)
 out[model]=dict(proposals=summary,verifiers=experts)
Path('reports/tx-v2/diagnosis/source-diagnosis.json').write_text(json.dumps(out,indent=2))
Path('reports/tx-v2/diagnosis/source-changed-examples.json').write_text(json.dumps(examples,indent=2))
print(json.dumps(out,indent=2))
