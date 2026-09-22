import json,re,sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import yaml
from merit_feddg.expert_policy import role_card
r=Path('runs/claim-unit');rows=[json.loads(x) for x in (r/'manifest.jsonl').read_text().splitlines()];b=json.load(open(r/'short-baseline.json'));c=json.load(open(r/'short-candidate.json'));refs=json.load(open(r/'references.json'));proposers=json.load(open(r/'proposers.json'));specs=yaml.safe_load(open('runs/v3/config.yaml'))['experts'];experts=['biomedclip_claim_verifier','conch_claim_verifier','plip_claim_verifier'];out={};detail={}
for mode in ['full','short']:
 obs={}
 for line in (r/f'{mode}.jsonl.transactions.jsonl').read_text().splitlines():
  x=json.loads(line);sid=x['transaction_id'].split(':adaptive-bard')[0];obs.setdefault(sid,{})[x['expert_id']]=x['differential_effect']
 methods={e:[] for e in ['baseline','candidate']+experts+['all_three']};cases=[]
 for row in rows:
  sid=row['id'];ref=refs[sid];ref=[ref] if isinstance(ref,str) else ref
  bs=float(b[sid]['text'] in ref);cs=float(c[sid]['text'] in ref);changed=b[sid]['text']!=c[sid]['text'];pg={role_card(e,specs[e]).fault_group for e in proposers[sid]}
  effects={e:v for e,v in obs.get(sid,{}).items() if role_card(e,specs[e]).fault_group not in pg}
  decisions={}
  for method in methods:
   if method=='baseline':accept=False
   elif method=='candidate':accept=changed
   else:
    vals=[v for e,v in effects.items() if method=='all_three' or e==method];accept=changed and any(v>0 for v in vals) and not any(v<0 for v in vals)
   methods[method].append((cs if accept else bs,bs,accept));decisions[method]=accept
  cases.append(dict(id=sid,polarity_flip=changed,baseline_correct=bs,candidate_correct=cs,effects=effects,decisions=decisions))
 out[mode]={k:dict(n=len(v),correct=sum(a for a,_,_ in v),accepted=sum(t for _,_,t in v),corrected=sum(a>bs for a,bs,_ in v),broken=sum(a<bs for a,bs,_ in v)) for k,v in methods.items()};detail[mode]=cases
Path('reports/claim-unit/results.json').write_text(json.dumps(out,indent=2));Path('reports/claim-unit/cases.json').write_text(json.dumps(detail,indent=2));print(json.dumps(out,indent=2))
