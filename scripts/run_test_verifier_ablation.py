"""Fixed TEST postprocessing ablation, unqualified and not deployment."""
import json
from pathlib import Path
from merit_feddg.io import load_experiment_yaml
from merit_feddg.open_experts import OpenExpertPool
from merit_feddg.transactional_claims import claimize_vqa,candidate_transactions
from merit_feddg.transactional_runtime import transaction_claim_spec,native_scores
from merit_feddg.knockoff import select_matched_knockoffs
from merit_feddg.merit_tx import differential_margin_controls
r=Path('runs/test256');rows=json.loads((r/'inputs.json').read_text());controls=[json.loads(x) for x in Path('/home/dbw/merit-tx-v2/runs/tx-v2/qualification-routed.jsonl').read_text().splitlines()];specs=load_experiment_yaml('runs/v3/config.yaml')['experts'];pool=OpenExpertPool(specs,'artifacts');out=r/'cases';out.mkdir(exist_ok=True)
for i,row in enumerate(rows):
 p=out/(row['id']+'.json')
 if p.exists():continue
 b=claimize_vqa(row['question'],row['baseline']);c=claimize_vqa(row['question'],row['candidate']);tx=candidate_transactions(task='open_vqa',question=row['question'],baseline_text=row['baseline'],candidate_text=row['candidate'],baseline_claims=b,candidate_claims=c,transaction_prefix=row['id']);effects={};errors={}
 if tx:
  claim=transaction_claim_spec(row,tx[0],b)
  for e in ['biomedclip_claim_verifier','conch_claim_verifier','plip_claim_verifier']:
   try:
    real=native_scores(pool,e,row['image'],claim);matched=select_matched_knockoffs(controls,row,expert_id=e,count=4);pairs=[native_scores(pool,e,x['image'],claim) for x in matched];effects[e]=differential_margin_controls(incumbent_real=real[0],candidate_real=real[1],knockoff_pairs=pairs)
   except (ValueError,RuntimeError) as exc:errors[e]=str(exc)
 result={'id':row['id'],'baseline':row['baseline'],'candidate':row['candidate'],'effects':effects,'errors':errors,'references_read':False,'unqualified_exploratory':True,'proposer_independence_verified':False}
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(result));tmp.replace(p);pool.reset_case();print('COMPLETE',i+1,row['id'],flush=True)
