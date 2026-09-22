"""Unqualified verifier ablation on previously observed development cases."""
import json,sys,itertools
from pathlib import Path
sys.path[:0]=['/home/dbw/ANCHOR','/home/dbw/merit-tx-v2/scripts']
from tx_source_metric import source_score
import yaml
sys.path.insert(0,str(Path.cwd()))
from merit_feddg.expert_policy import role_card
r=Path('runs/exploratory');rows=[json.loads(x) for x in (r/'manifest.jsonl').read_text().splitlines()];refs=json.load(open(r/'references.json'));specs=yaml.safe_load(open('runs/v3/config.yaml'))['experts']
experts=['biomedclip_claim_verifier','conch_claim_verifier','plip_claim_verifier'];result={};caseout={}
for m in ['llava','huatuo']:
 b=json.load(open(r/f'{m}-baseline.json'));c=json.load(open(r/f'{m}-candidate.json'));proposers=json.load(open(r/f'{m}-proposers.json'));obs={}
 for line in (r/f'{m}-observations.jsonl.transactions.jsonl').read_text().splitlines():
  x=json.loads(line);sid=x['transaction_id'].split(':adaptive-bard')[0];obs.setdefault(sid,{})[x['expert_id']]=x
 arms={'baseline':[],'candidate':[]}; subsets={}
 for size in range(1,4):
  for group in itertools.combinations(experts,size):
   name='+'.join(group);subsets[name]=group;arms[name]=[]
 details=[]
 for row in rows:
  sid=row['id'];ref=refs[sid];ref=[ref] if isinstance(ref,str) else ref
  bs=source_score(row,b[sid]['text'],ref);cs=source_score(row,c[sid]['text'],ref)
  arms['baseline'].append((bs,False,bs));arms['candidate'].append((cs,c[sid]['text']!=b[sid]['text'],bs))
  proposer_groups={role_card(e,specs[e]).fault_group for e in proposers[sid]}
  effects={e:x['differential_effect'] for e,x in obs.get(sid,{}).items() if role_card(e,specs[e]).fault_group not in proposer_groups}
  decisions={}
  for name,group in subsets.items():
   values=[effects[e] for e in group if e in effects];accept=any(v>0 for v in values) and not any(v<0 for v in values)
   arms[name].append((cs if accept else bs,accept,bs));decisions[name]=accept
  details.append(dict(id=sid,baseline_score=bs,candidate_score=cs,effects=effects,decisions=decisions))
 result[m]={name:dict(n=len(v),score=sum(a for a,_,_ in v)/len(v),accepted=sum(a for _,a,_ in v),improved=sum(a>b+1e-12 for a,_,b in v),harmed=sum(a<b-1e-12 for a,_,b in v)) for name,v in arms.items()};caseout[m]=details
Path('reports/exploratory/results.json').write_text(json.dumps(result,indent=2));Path('reports/exploratory/cases.json').write_text(json.dumps(caseout,indent=2));print(json.dumps(result,indent=2))
