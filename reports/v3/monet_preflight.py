import json,time
from pathlib import Path
from merit_feddg.experts.monet import MonetConceptExpert
rows=[json.loads(x) for x in Path('/home/dbw/merit-tx-v2/runs/tx-v2/qualification-routed.jsonl').read_text().splitlines()]
row=next(x for x in rows if x['modality']=='dermatology')
start=time.time();model=MonetConceptExpert('/home/dbw/merit-feddg/artifacts/models/chanwkim--monet',device='cuda')
claims=['The skin shows a lesion.','The skin shows no lesion.']
scores=model.score_claims(row['image'],row['question'],'',claims)
result={'id':row['id'],'modality':row['modality'],'claims':claims,'scores':scores.tolist(),'seconds':time.time()-start,'references_read':False,'purpose':'real checkpoint adapter preflight, not diagnostic accuracy','physical_gpu':1}
Path('reports/v3/monet-real-preflight.json').write_text(json.dumps(result,indent=2));print(json.dumps(result));model.close()
