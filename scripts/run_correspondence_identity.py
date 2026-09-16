"""Source identity diagnosis, explicitly separating privileged source labels."""
import argparse,hashlib,json,time
from pathlib import Path
from run_crossmodal_correspondence_v2 import ROOT,MODEL,DATA,red_box,box_parse,iou,write,digest
SOURCE=ROOT/'runs/medsg-correspondence-dev-v2'
RUN=ROOT/'runs/medsg-identity-dev-v1'
NAMES={'BrainEnhancingTumor':'enhancing brain tumor','BrainCoreTumor':'brain tumor core','BrainWholeTumor':'whole brain tumor','LeftLung':'left lung','RightLung':'right lung','Duodenum':'duodenum','Stomach':'stomach','Spleen':'spleen','Liver':'liver','LeftVentricle':'left ventricle','LeftVentricleEpicardium':'left ventricular epicardium'}

def prepare():
 assert not RUN.exists()
 m=json.loads((SOURCE/'manifest.json').read_text());g=json.loads((SOURCE/'evaluation_only.json').read_text());raw=json.loads((DATA/'Task7.json').read_text())
 for r in m['selected']:
  name=Path(raw[g[r['id']]['original_index']]['images'][0]).name.split('_')[2]
  r['privileged_source_name']=NAMES[name]
 m.update(experiment=RUN.name,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),source='benchmark SOURCE class oracle; no target label or bbox in manifest')
 write(RUN/'manifest.json',m)

def run():
 import torch
 from PIL import Image
 from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
 from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
 m=json.loads((RUN/'manifest.json').read_text());assert m['code_sha256']==hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
 torch.manual_seed(197);t=time.perf_counter()
 processor=AutoProcessor.from_pretrained(MODEL,min_pixels=256*28*28,max_pixels=512*28*28,local_files_only=True)
 model=Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL,torch_dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='sdpa',local_files_only=True).eval()
 def generate(im,label,prompt,tokens):
  content=[{'type':'text','text':label},{'type':'image','image':im},{'type':'text','text':prompt}]
  text=processor.apply_chat_template([{'role':'user','content':content}],tokenize=False,add_generation_prompt=True)
  x=processor(text=[text],images=[im],return_tensors='pt').to('cuda');n=x['input_ids'].shape[1];torch.cuda.synchronize();start=time.perf_counter()
  with torch.inference_mode():o=model.generate(**x,max_new_tokens=tokens,do_sample=False,use_cache=True)
  torch.cuda.synchronize()
  return dict(text=processor.decode(o[0,n:],skip_special_tokens=True),seconds=time.perf_counter()-start,input_tokens=n,output_tokens=int(o.shape[1]-n))
 for r in m['selected']:
  dest=RUN/'cases'/f"{r['id']}.json"
  if dest.exists():continue
  src=Image.open(r['reference']).convert('RGB');tgt=Image.open(r['target']).convert('RGB');assert digest(src)==r['hashes'][0] and digest(tgt)==r['hashes'][1]
  b,_=red_box(src);assert b is not None
  crop=src.crop((b[0]+2,b[1]+2,max(b[0]+3,b[2]-1),max(b[1]+3,b[3]-1)))
  obs=generate(crop,'Cropped reference region.','Name the principal anatomical structure or finding shown in this medical image crop. Return only a short name, no explanation.',64)
  original=tgt.size;h,w=smart_resize(tgt.height,tgt.width,factor=28,min_pixels=256*28*28,max_pixels=512*28*28);tgt=tgt.resize((w,h),Image.Resampling.BICUBIC)
  out=dict(id=r['id'],crop_name=obs,arms={})
  for arm,name in [('crop_name',obs['text']),('oracle_source_name',r['privileged_source_name'])]:
   prompt='Locate the region described by this fallible REFERENCE observation tool output: '+json.dumps({'entity':name},ensure_ascii=False)+'. Use TARGET pixels to determine location.'
   prompt+=f' Return a JSON list with exactly one object, containing only bbox_2d: [x1,y1,x2,y2]. Use TARGET PIXEL coordinates; target width {tgt.width}, height {tgt.height}. No label or explanation. Do not use normalized coordinates.'
   v=generate(tgt,'TARGET image.',prompt,128);b=box_parse(v['text'],tgt.size);v['box']=[v*original[i%2]/tgt.size[i%2] for i,v in enumerate(b)] if b else None;out['arms'][arm]=v
  write(dest,out);print('DONE',r['id'],'crop name',obs['text'],flush=True)
 print('WALL_SECONDS',time.perf_counter()-t,flush=True)

def evaluate():
 gold=json.loads((SOURCE/'evaluation_only.json').read_text());rows=[]
 for p in sorted((RUN/'cases').glob('*.json')):
  d=json.loads(p.read_text());g=gold[d['id']];old=json.loads((SOURCE/'cases'/p.name).read_text())
  scores={k:iou(v['box'],g['box']) for k,v in d['arms'].items()};scores['predicted_name']=iou(old['arms']['name']['box'],g['box'])
  rows.append(dict(id=d['id'],transition=g['transition'],scores=scores,valid={k:v['box'] is not None for k,v in d['arms'].items()}))
 summary={}
 for group in ['all','nonaligned','aligned_MRI']:
  sub=[r for r in rows if group=='all' or (r['transition']!='aligned_MRI')==(group=='nonaligned')]
  summary[group]={a:dict(n=len(sub),mean_iou=sum(r['scores'][a] for r in sub)/len(sub),acc50=sum(r['scores'][a]>=.5 for r in sub)) for a in ['predicted_name','crop_name','oracle_source_name']}
 write(RUN/'evaluation.json',dict(summary=summary,rows=rows));print(json.dumps(summary,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run','evaluate']);a=p.parse_args();globals()[a.stage]()
