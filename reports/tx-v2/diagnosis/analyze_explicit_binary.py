import json,re
from pathlib import Path
from collections import Counter
r=Path('runs/tx-v2');out={}
for m in ['llava','huatuo']:
 b=json.load(open(r/f'{m}-qualification-baseline.json'));c=json.load(open(r/f'{m}-qualification-candidate.json'));refs=json.load(open(r/'qualification-references.json'))
 rows=[]
 for sid,ref in refs.items():
  if isinstance(ref,list):
   if len(ref)!=1:continue
   ref=ref[0]
  if ref.lower() not in ['yes','no']:continue
  def lead(t):
   a=re.match(r'^\s*(yes|no)\b',t,re.I);return a.group(1).lower() if a else None
  bp=lead(b[sid]['text']);cp=lead(c[sid]['text']);rows.append(dict(id=sid,reference=ref.lower(),baseline=bp,candidate=cp))
 # Same subset with explicit leading yes/no in BOTH answers: no guesses about non-leading responses.
 pair=[x for x in rows if x['baseline'] and x['candidate']]
 out[m]=dict(binary_reference_n=len(rows),paired_explicit_n=len(pair),excluded_ambiguous_n=len(rows)-len(pair),baseline_correct=sum(x['baseline']==x['reference'] for x in pair),candidate_correct=sum(x['candidate']==x['reference'] for x in pair),corrected=sum(x['baseline']!=x['reference'] and x['candidate']==x['reference'] for x in pair),broken=sum(x['baseline']==x['reference'] and x['candidate']!=x['reference'] for x in pair),same_polarity=sum(x['baseline']==x['candidate'] for x in pair))
 out[m]['pathorob_candidate_distribution']=dict(Counter(x['text'].lower().strip(' .') for sid,x in c.items() if sid.startswith('pathorob')))
Path('reports/tx-v2/diagnosis/explicit-binary-diagnostic.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
