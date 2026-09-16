import json,time
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoModelForCausalLM,AutoProcessor
p='/home/dbw/models/Hulu-Med-4B'
r=Path('/home/dbw/merit-feddg-relational-pilot/runs/medsg-correspondence-dev-v2')
case=json.loads((r/'manifest.json').read_text())['selected'][0]
m=AutoModelForCausalLM.from_pretrained(p,trust_remote_code=True,torch_dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='sdpa',local_files_only=True).eval()
pr=AutoProcessor.from_pretrained(p,trust_remote_code=True,local_files_only=True);pr.image_processor.max_tokens=512
records=[]
for keys,prompt in [(['reference'],'Name the anatomical structure within the red box. Return only a short name.'),(['reference'],'Locate the visible red rectangle. Return only JSON [x1,y1,x2,y2] using original image pixel coordinates. Width336 height336.'),(['reference','target'],'The first image is the reference; the second is the target. Locate in the target the anatomical region corresponding to the red-boxed reference region. Return only JSON [x1,y1,x2,y2] in original target pixel coordinates. Width336 height336.')]:
 imgs=[Image.open(case[k]).convert('RGB') for k in keys]
 content=[{'type':'image'} for _ in imgs]+[{'type':'text','text':prompt}]
 x=pr(images=imgs,conversation=[{'role':'user','content':content}],add_system_prompt=False,add_generation_prompt=True,return_tensors='pt')
 for k,v in x.items():
  if torch.is_tensor(v):x[k]=v.to(device='cuda',dtype=m.dtype if k=='pixel_values' else v.dtype)
 start=time.time()
 with torch.inference_mode():o=m.generate(**x,max_new_tokens=128,do_sample=False,use_cache=True,pad_token_id=pr.tokenizer.eos_token_id)
 text=pr.batch_decode(o,skip_special_tokens=True,use_think=False)[0].strip();records.append(dict(keys=keys,prompt=prompt,text=text,seconds=time.time()-start));print(text,flush=True)
out=r.parent/'hulu-correspondence-interface-canary';out.mkdir(exist_ok=True);(out/'outputs.json').write_text(json.dumps({'id':case['id'],'model':p,'records':records},indent=2))
