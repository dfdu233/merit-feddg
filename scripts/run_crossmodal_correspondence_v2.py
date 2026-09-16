"""Reference-to-target localization using fixed tool descriptions; gold offline only."""
import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
import subprocess
import time
import zipfile
from run_relational_pilot import MODEL,ROOT,digest,write

DATA=Path('/home/dbw/datasets/public/MedSG-Bench')
RUN=ROOT/'runs/medsg-correspondence-dev-v2'
GROUPS=['aligned_MRI','CT_MRI','MRI_CT','US_MRI','MRI_US','CT_xray','xray_CT']
OBSERVE='''The red box in this reference image indicates a region to recognize in another medical image. Describe ONLY that region and its anatomical context. Return exactly JSON with keys: "entity" (short anatomical or finding name), "appearance" (list of short visible features), "relations" (list of objects with "relation" and "landmark" strings). At most six relations. Relate the marked entity to visible anatomical landmarks where discernible. Do not equate image-side with patient-side. Do not invent absent relations. No extra fields or prose.'''


def modality(name):
    return next((m for m in ('MRI','CT','US','xray') if '_'+m+'_' in name),'other')


def case_id(name):
    name=Path(name).stem;dataset=name.split('_')[0]
    match=re.search(r'casenum_(.*?)_sliceid_',name)
    if match:return dataset+':'+match[1]
    match=re.search(r'volume-(\d+)',name)
    if match:return dataset+':volume'+match[1]
    if modality(name)=='xray' and '_imgs_' in name:return dataset+':image:'+name.split('_imgs_',1)[1]
    return None


def target_index(question):
    return next(i for word,i in [('second',1),('third',2),('fourth',3)] if 'in the '+word+' image' in question)


def prepare():
    from PIL import Image
    assert not RUN.exists()
    rows=json.loads((DATA/'Task7.json').read_text());candidates=[]
    for idx,r in enumerate(rows):
        t=target_index(r['question']);sname,tname=r['images'][0],r['images'][t]
        sid,tid=case_id(sname),case_id(tname)
        if sid is None or tid is None:continue
        g='aligned_MRI' if sname.split('/')[-1].startswith('BraTS') and sid==tid else modality(sname)+'_'+modality(tname)
        if g not in GROUPS or (g!='aligned_MRI' and len(r['images'])!=2):continue
        key=hashlib.sha256((str(idx)+sname+tname).encode()).hexdigest()
        candidates.append((key,idx,r,t,g,sid,tid))
    selected=[];gold={};seen=set();counts={g:0 for g in GROUPS}
    with zipfile.ZipFile(DATA/'Task7.zip') as z:
        for _,idx,r,t,g,sid,tid in sorted(candidates):
            if counts[g]>=4 or sid in seen or tid in seen:continue
            key=f'case-{idx:04d}';paths=[];hashes=[]
            for role,name in [('reference',r['images'][0]),('target',r['images'][t])]:
                dest=RUN/'images'/f'{key}-{role}.png';dest.parent.mkdir(parents=True,exist_ok=True)
                with Image.open(BytesIO(z.read('Task7/'+Path(name).name))) as im:
                    im=im.convert('RGB');im.save(dest);hashes.append(digest(im));paths.append(str(dest))
            selected.append(dict(id=key,reference=paths[0],target=paths[1],hashes=hashes))
            gold[key]=dict(box=r['answer'],transition=g,source_case=sid,target_case=tid,
                           original_index=idx,original_image_count=len(r['images']),target_ordinal=t+1)
            seen.update((sid,tid));counts[g]+=1
    cfg=dict(experiment='medsg-correspondence-dev-v2',selected=selected,model=MODEL,observer_prompt=OBSERVE,
             observer_tokens=384,target_tokens=128,counts=counts,seed=197,
             code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             dependency_sha256=hashlib.sha256((ROOT/'scripts/run_relational_pilot.py').read_bytes()).hexdigest(),
             data_sha256=hashlib.sha256((DATA/'Task7.json').read_bytes()).hexdigest(),
             commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
             gpu_uuid='GPU-3846413a-4238-d307-b1f3-10c2dfbe002c')
    write(RUN/'manifest.json',cfg);write(RUN/'evaluation_only.json',gold)
    print(json.dumps({'selected':len(selected),'counts':counts},indent=2),flush=True)


def red_box(im):
    import numpy as np
    a=np.asarray(im);mask=(a[:,:,0]>200)&(a[:,:,1]<80)&(a[:,:,2]<80)
    y,x=np.where(mask)
    return ([int(x.min()),int(y.min()),int(x.max()),int(y.max())] if len(x)>=16 else None),int(len(x))


def parse_description(text):
    d,_=json.JSONDecoder().raw_decode(text[text.index('{'):])
    assert set(d)=={'entity','appearance','relations'} and isinstance(d['entity'],str)
    assert isinstance(d['appearance'],list) and all(isinstance(a,str) for a in d['appearance'])
    assert isinstance(d['relations'],list) and len(d['relations'])<=6
    assert all(set(r)=={'relation','landmark'} and all(isinstance(v,str) for v in r.values()) for r in d['relations'])
    return d


def box_parse(text,size):
    try:
        b,_=json.JSONDecoder().raw_decode(text[text.index('['):])
        if isinstance(b,list) and len(b)==1 and isinstance(b[0],dict):b=b[0].get('bbox_2d')
        assert isinstance(b,list) and len(b)==4 and all(isinstance(v,(int,float)) and not isinstance(v,bool) for v in b)
        assert 0<=b[0]<b[2]<=size[0] and 0<=b[1]<b[3]<=size[1]
        return b
    except (ValueError,AssertionError,TypeError):return None


def run(limit):
    import torch
    from PIL import Image
    from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
    cfg=json.loads((RUN/'manifest.json').read_text())
    assert cfg['code_sha256']==hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    assert cfg['dependency_sha256']==hashlib.sha256((ROOT/'scripts/run_relational_pilot.py').read_bytes()).hexdigest()
    assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==cfg['gpu_uuid']
    torch.manual_seed(197);started=time.perf_counter()
    processor=AutoProcessor.from_pretrained(MODEL,min_pixels=256*28*28,max_pixels=512*28*28,local_files_only=True)
    model=Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL,torch_dtype=torch.bfloat16,
             device_map={'':'cuda:0'},attn_implementation='sdpa',local_files_only=True).eval()
    print('LOADED',round(time.perf_counter()-started,2),flush=True)
    def generate(images,labels,prompt,tokens):
        content=[]
        for im,label in zip(images,labels):content.extend([{'type':'text','text':label},{'type':'image','image':im}])
        content.append({'type':'text','text':prompt})
        text=processor.apply_chat_template([{'role':'user','content':content}],tokenize=False,add_generation_prompt=True)
        inputs=processor(text=[text],images=images,return_tensors='pt').to('cuda');n=inputs['input_ids'].shape[1]
        torch.cuda.synchronize();t=time.perf_counter()
        with torch.inference_mode():out=model.generate(**inputs,max_new_tokens=tokens,do_sample=False,use_cache=True)
        torch.cuda.synchronize()
        return dict(text=processor.decode(out[0,n:],skip_special_tokens=True),input_tokens=n,output_tokens=int(out.shape[1]-n),seconds=time.perf_counter()-t)
    descriptions={}
    for r in cfg['selected']:
        path=RUN/'observations'/f"{r['id']}.json"
        if path.exists():obs=json.loads(path.read_text())
        else:
            im=Image.open(r['reference']).convert('RGB');assert digest(im)==r['hashes'][0]
            obs=generate([im],['REFERENCE image, with marked region.'],cfg['observer_prompt'],384)
            try:obs.update(valid=True,description=parse_description(obs['text']))
            except (ValueError,AssertionError,TypeError,KeyError):obs.update(valid=False)
            write(path,obs);print('OBSERVED',r['id'],obs['valid'],flush=True)
        descriptions[r['id']]=obs
    valid=[r['id'] for r in cfg['selected'] if descriptions[r['id']]['valid']]
    done=0
    for r in cfg['selected']:
        dest=RUN/'cases'/f"{r['id']}.json"
        if dest.exists():continue
        src=Image.open(r['reference']).convert('RGB');tgt=Image.open(r['target']).convert('RGB')
        assert digest(src)==r['hashes'][0] and digest(tgt)==r['hashes'][1]
        rb,rn=red_box(src);_,tn=red_box(tgt)
        copied=[rb[i]*tgt.size[i%2]/src.size[i%2] for i in range(4)] if rb else None
        original_size=tgt.size
        def resize_input(im):
            h,w=smart_resize(im.height,im.width,factor=28,min_pixels=256*28*28,max_pixels=512*28*28)
            return im.resize((w,h),Image.Resampling.BICUBIC)
        src,tgt=resize_input(src),resize_input(tgt)
        suffix=f' Return a JSON list with exactly one object, containing only bbox_2d: [x1,y1,x2,y2]. Use TARGET PIXEL coordinates; target width {tgt.width}, height {tgt.height}. No label or explanation. Do not use normalized coordinates.'
        arms=[('joint',[src,tgt],['REFERENCE image.','TARGET image.'],'Locate in the TARGET the anatomical region corresponding to the red-boxed region in the REFERENCE.'),
              ('target_only',[tgt],['TARGET image.'],'Locate the corresponding anatomical region in the TARGET. The reference image and region identity are unavailable.')]
        donor=None
        if descriptions[r['id']]['valid']:
            d=descriptions[r['id']]['description'];pos=valid.index(r['id']);donor=valid[(pos+1)%len(valid)]
            for name,desc in [('name',{'entity':d['entity']}),('relations',{'entity':d['entity'],'relations':d['relations']}),
                              ('full',d),('wrong_reference',descriptions[donor]['description'])]:
                arms.append((name,[tgt],['TARGET image.'],'Locate the region described by this fallible REFERENCE observation tool output: '+json.dumps(desc,ensure_ascii=False)+'. Use TARGET pixels to determine location.'))
        result=dict(id=r['id'],source_size=list(src.size),target_size=list(tgt.size),source_red_pixels=rn,target_red_pixels=tn,
                    geometry_box=copied,source_parse_valid=descriptions[r['id']]['valid'],wrong_reference_donor=donor,arms={})
        for name,images,labels,prompt in arms:
            out=generate(images,labels,prompt+suffix,cfg['target_tokens']);b=box_parse(out['text'],tgt.size)
            out['processed_target_size']=list(tgt.size);out['processed_box']=b
            out['box']=[v*original_size[i%2]/tgt.size[i%2] for i,v in enumerate(b)] if b else None
            result['arms'][name]=out
        write(dest,result);print('DONE',r['id'],'red',rn,tn,'boxes',{k:v['box'] for k,v in result['arms'].items()},flush=True)
        done+=1
        if limit and done>=limit:break
    print('RUN_FINISHED',done,'seconds',round(time.perf_counter()-started,2),flush=True)


def iou(a,b):
    if a is None:return 0.
    intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/union if union>0 else 0.


def evaluate():
    gold=json.loads((RUN/'evaluation_only.json').read_text());cfg=json.loads((RUN/'manifest.json').read_text())
    rows=[]
    for p in sorted((RUN/'cases').glob('*.json')):
        r=json.loads(p.read_text());g=gold[r['id']]
        scores={k:iou(v['box'],g['box']) for k,v in r['arms'].items()};scores['geometry']=iou(r['geometry_box'],g['box'])
        rows.append(dict(id=r['id'],transition=g['transition'],scores=scores,
                         valid={k:v['box'] is not None for k,v in r['arms'].items()},
                         target_red_pixels=r['target_red_pixels'],source_parse_valid=r['source_parse_valid']))
    summary=dict(completed=len(rows),planned=len(cfg['selected']),source_parse_valid=sum(r['source_parse_valid'] for r in rows),
                 target_red_marked=sum(r['target_red_pixels']>=16 for r in rows),groups={})
    for group_name in ['all','nonaligned']+GROUPS:
        subset=[r for r in rows if group_name=='all' or (group_name=='nonaligned' and r['transition']!='aligned_MRI') or r['transition']==group_name]
        group_result={}
        for arm in ['joint','target_only','name','relations','full','wrong_reference','geometry']:
            usable=[r for r in subset if arm in r['scores']]
            group_result[arm]=dict(n=len(usable),mean_iou=sum(r['scores'][arm] for r in usable)/len(usable) if usable else None,
                                   acc50=sum(r['scores'][arm]>=.5 for r in usable),
                                   valid=sum(r['valid'].get(arm,True) for r in usable))
        summary['groups'][group_name]=group_result
    write(RUN/'evaluation.json',dict(summary=summary,rows=rows))
    print(json.dumps({k:v for k,v in summary.items() if k!='groups'},indent=2))
    print(json.dumps({'nonaligned':summary['groups']['nonaligned'],'aligned_MRI':summary['groups']['aligned_MRI']},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run','evaluate']);p.add_argument('--limit',type=int,default=0);a=p.parse_args()
    if a.stage=='prepare':prepare()
    elif a.stage=='run':run(a.limit)
    else:evaluate()
