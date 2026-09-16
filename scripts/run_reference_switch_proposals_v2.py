"""Question-independent SAM proposal coverage before graph candidate selection."""
import argparse,hashlib,json,time,zipfile,itertools
from io import BytesIO
from pathlib import Path
from collections import Counter
from run_crossmodal_correspondence_v2 import ROOT,DATA,case_id,iou,write,digest
RUN=ROOT/'runs/medsg-reference-switch-proposals-dev-v2'
MODEL='/home/dbw/models/sam-vit-base'
KW=dict(points_per_crop=16,points_per_batch=32,crops_n_layers=0,pred_iou_thresh=.88,stability_score_thresh=.95,mask_threshold=0,crops_nms_thresh=.7)

def prepare():
 from PIL import Image
 assert not RUN.exists()
 groups=json.loads((ROOT/'runs/medsg-reference-switch-eligibility/input_groups.json').read_text());old=json.loads((ROOT/'runs/medsg-correspondence-dev-v2/evaluation_only.json').read_text());raw=json.loads((DATA/'Task7.json').read_text())
 seen={v[k] for v in old.values() for k in ['source_case','target_case']};ranked=[]
 with zipfile.ZipFile(DATA/'Task7.zip') as z:
  for g in groups:
   t=Image.open(BytesIO(z.read('Task7/'+Path(g[0]['target']).name))).convert('RGB');ranked.append((digest(t),g))
  eligible=[]
  for h,g in sorted(ranked):
   ids={case_id(r[k]) for r in g for k in ['source','target']}
   if None not in ids and not ids&seen:eligible.append((h,g,ids))
  best=()
  for n in range(1,len(eligible)+1):
   found=None
   for c in itertools.combinations(eligible,n):
    if sum(len(x[2]) for x in c)!=len(set().union(*(x[2] for x in c))):continue
    cnt=Counter(x[1][0]['transition'] for x in c)
    if any(v>4 for v in cnt.values()):continue
    found=c;break
   if found is None:break
   best=found
  ranked=[(h,g) for h,g,ids in best]
  selected=[];gold={};counts=Counter()
  for h,g in sorted(ranked):
   transition=g[0]['transition'];ids={case_id(r[k]) for r in g for k in ['source','target']}
   if transition not in ['CT_MRI','MRI_CT','MRI_US'] or counts[transition]>=4 or None in ids or ids&seen:continue
   key=f'group-{len(selected):02d}';refs=[]
   for role,name in [('target',g[0]['target'])]+[(f'reference-{j}',r['source']) for j,r in enumerate(g)]:
    im=Image.open(BytesIO(z.read('Task7/'+Path(name).name))).convert('RGB');p=RUN/'images'/f'{key}-{role}.png';p.parent.mkdir(parents=True,exist_ok=True);im.save(p)
    if role=='target':target=str(p)
    else:refs.append(str(p))
   selected.append(dict(id=key,target=target,target_hash=h,references=refs))
   gold[key]=dict(transition=transition,indices=[r['index'] for r in g],boxes=[raw[r['index']]['answer'] for r in g],case_ids=sorted(ids))
   counts[transition]+=1;seen.update(ids)
 write(RUN/'manifest.json',dict(experiment=RUN.name,selected=selected,counts=dict(counts),model=MODEL,revision='70c1a07f894ebb5b307fd9eaaee97b9dfc16068f',kwargs=KW,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
 write(RUN/'evaluation_only.json',gold);print('SELECTED',len(selected),'groups',sum(len(r['references']) for r in selected),'queries',dict(counts),flush=True)

def run():
 import numpy as np
 import torch
 from PIL import Image
 from transformers import pipeline,SamModel,SamImageProcessor
 m=json.loads((RUN/'manifest.json').read_text());assert m['code_sha256']==hashlib.sha256(Path(__file__).read_bytes()).hexdigest();torch.manual_seed(197)
 model=SamModel.from_pretrained(MODEL,local_files_only=True);processor=SamImageProcessor.from_pretrained(MODEL,local_files_only=True)
 generator=pipeline('mask-generation',model=model,image_processor=processor,device=0)
 for r in m['selected']:
  p=RUN/'proposals'/f"{r['id']}.json"
  if p.exists():continue
  im=Image.open(r['target']).convert('RGB');assert digest(im)==r['target_hash'];start=time.perf_counter()
  out=generator(im,**m['kwargs']);boxes=[];masks=[]
  for mask,score in zip(out['masks'],out['scores']):
   a=np.asarray(mask,dtype=bool);y,x=np.where(a)
   if not len(x):continue
   boxes.append(dict(id=len(boxes),box=[int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)],score=float(score)))
   masks.append(a)
  p.parent.mkdir(exist_ok=True);np.savez_compressed(p.with_suffix('.npz'),masks=np.array(masks,dtype=bool))
  write(p,dict(id=r['id'],candidates=boxes,seconds=time.perf_counter()-start));print('PROPOSED',r['id'],len(boxes),flush=True)

def evaluate():
 m=json.loads((RUN/'manifest.json').read_text());gold=json.loads((RUN/'evaluation_only.json').read_text());rows=[]
 for r in m['selected']:
  p=RUN/'proposals'/f"{r['id']}.json"
  if not p.exists():continue
  d=json.loads(p.read_text());scores=[max((iou(c['box'],b) for c in d['candidates']),default=0) for b in gold[r['id']]['boxes']]
  rows.append(dict(id=r['id'],transition=gold[r['id']]['transition'],n_candidates=len(d['candidates']),best_ious=scores,all_covered=all(v>=.5 for v in scores)))
 n=sum(len(r['best_ious']) for r in rows);covered=sum(r['all_covered'] for r in rows)
 summary=dict(completed_groups=len(rows),planned_groups=len(m['selected']),queries=n,covered_queries=sum(v>=.5 for r in rows for v in r['best_ious']),fully_covered_groups=covered,mean_best_iou=sum(sum(r['best_ious']) for r in rows)/n if n else None,proceed=len(rows)==len(m['selected']) and covered>=4 and covered>=len(rows)/2)
 write(RUN/'evaluation.json',dict(summary=summary,rows=rows));print(json.dumps(summary,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run','evaluate']);a=p.parse_args();globals()[a.stage]()
