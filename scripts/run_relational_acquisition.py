"""Exhaustive frozen crop-bank diagnostic; all policy oracles are offline."""
import argparse
from io import BytesIO
import itertools
import json
import hashlib
from pathlib import Path
import re
import subprocess
import time
import zipfile
from run_relational_pilot import DATA, MODEL, ROOT, RUN as PREVIOUS, STRATA, digest, group, normalize, write

RUN = ROOT/'runs/slake-relational-acquisition-dev-v1'
PATTERN = re.compile(r'\b(bigger|smaller|relative to|left of|right of|between)\b',re.I)
PROMPT = 'Answer the medical image question with only a short answer, without explanation. Image coordinates refer to the displayed image, not patient laterality. Additional crops, if supplied, are from the same image.\n'
PLAN = 'Select exactly two quadrants whose full-resolution observations would help answer the question. You currently see only a low-resolution overview. Quadrants: 0=image top-left, 1=top-right, 2=bottom-left, 3=bottom-right. Output exactly two distinct digits separated by a comma.\nQuestion: '


def prepare():
    from PIL import Image
    assert not RUN.exists()
    previous=json.loads((PREVIOUS/'manifest.json').read_text())
    splits={s:json.loads((DATA/f'{s}.json').read_text()) for s in ('train','validation','test')}
    hashes={}
    with zipfile.ZipFile(DATA/'imgs.zip') as z:
        for name in sorted({r['img_name'] for rows in splits.values() for r in rows}):
            with Image.open(BytesIO(z.read('imgs/'+name))) as im: hashes[name]=digest(im)
        forbidden={hashes[r['img_name']] for s in ('validation','test') for r in splits[s]}
        forbidden.update(r['image_hash'] for r in previous['selected'])
        rows=[r for r in splits['train'] if r['q_lang']=='en' and r['base_type']=='vqa' and PATTERN.search(r['question'])
              and group(r) in STRATA and hashes[r['img_name']] not in forbidden]
        rows.sort(key=lambda r:hashlib.sha256(f"{r['qid']}:{r['img_name']}".encode()).hexdigest())
        counts={g:0 for g in STRATA};seen=set();selected=[];gold={}
        for r in rows:
            g,h=group(r),hashes[r['img_name']]
            if counts[g]>=3 or h in seen: continue
            key=f"slake-{r['qid']}";path=RUN/'images'/f'{key}.png';path.parent.mkdir(parents=True,exist_ok=True)
            with Image.open(BytesIO(z.read('imgs/'+r['img_name']))) as im: im.convert('RGB').save(path)
            selected.append(dict(id=key,image=str(path),image_hash=h,question=r['question']))
            gold[key]=dict(answer=r['answer'],stratum=list(g))
            seen.add(h);counts[g]+=1
    cfg=dict(experiment='slake-relational-acquisition-dev-v1',selected=selected,model=MODEL,seed=197,
             prompt=PROMPT,plan_prompt=PLAN,answer_tokens=32,plan_tokens=16,overview_side=112,
             code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             dependency_sha256=hashlib.sha256((ROOT/'scripts/run_relational_pilot.py').read_bytes()).hexdigest(),
             selection_counts={str(k):v for k,v in counts.items()},eligible=len(rows),
             source_manifest=str(PREVIOUS/'manifest.json'),
             source_manifest_sha256=hashlib.sha256((PREVIOUS/'manifest.json').read_bytes()).hexdigest(),
             gpu_uuid='GPU-3846413a-4238-d307-b1f3-10c2dfbe002c',
             commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    write(RUN/'manifest.json',cfg);write(RUN/'evaluation_only.json',gold)
    print(json.dumps({'selected':len(selected),'counts':cfg['selection_counts'],'eligible':len(rows)},indent=2),flush=True)


def run(limit):
    import torch
    from PIL import Image
    from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
    cfg=json.loads((RUN/'manifest.json').read_text())
    assert cfg['code_sha256']==hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    assert cfg['dependency_sha256']==hashlib.sha256((ROOT/'scripts/run_relational_pilot.py').read_bytes()).hexdigest()
    assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==cfg['gpu_uuid']
    torch.manual_seed(197);started=time.perf_counter()
    processor=AutoProcessor.from_pretrained(MODEL,min_pixels=256*28*28,max_pixels=512*28*28,local_files_only=True)
    model=Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL,torch_dtype=torch.bfloat16,
             device_map={'':'cuda:0'},attn_implementation='sdpa',local_files_only=True).eval()
    print('LOADED',round(time.perf_counter()-started,2),flush=True)

    def generate(images,labels,prompt,budget):
        content=[]
        for im,label in zip(images,labels):
            content.extend([{'type':'text','text':label},{'type':'image','image':im}])
        content.append({'type':'text','text':prompt})
        rendered=processor.apply_chat_template([{'role':'user','content':content}],tokenize=False,add_generation_prompt=True)
        inputs=processor(text=[rendered],images=images or None,return_tensors='pt').to('cuda')
        n=inputs['input_ids'].shape[1];torch.cuda.synchronize();t=time.perf_counter()
        with torch.inference_mode():
            out=model.generate(**inputs,max_new_tokens=budget,do_sample=False,use_cache=True,
                               return_dict_in_generate=True,output_scores=True)
        torch.cuda.synchronize();elapsed=time.perf_counter()-t
        generated=out.sequences[0,n:]
        confidence=sum(float(torch.log_softmax(s[0].float(),dim=-1)[token]) for s,token in zip(out.scores,generated))/max(len(generated),1)
        return dict(text=processor.decode(generated,skip_special_tokens=True),input_tokens=n,
                    output_tokens=len(generated),seconds=elapsed,mean_generated_logp=confidence)

    done=0
    for r in cfg['selected']:
        dest=RUN/'cases'/f"{r['id']}.json"
        if dest.exists(): continue
        im=Image.open(r['image']).convert('RGB');assert digest(im)==r['image_hash']
        overview=im.copy();overview.thumbnail((112,112),Image.Resampling.LANCZOS)
        w,h=im.size;boxes=[(0,0,w//2,h//2),(w//2,0,w,h//2),(0,h//2,w//2,h),(w//2,h//2,w,h)]
        crops=[im.crop(b) for b in boxes]
        labels=[f'Crop {i}, original pixel box {list(b)}, original image size {[w,h]}.' for i,b in enumerate(boxes)]
        question=cfg['prompt']+'Question: '+r['question']
        result=dict(id=r['id'],original_size=[w,h],overview_size=list(overview.size),boxes=boxes,arms={})
        plan=generate([overview],['Low-resolution overview.'],cfg['plan_prompt']+r['question'],cfg['plan_tokens'])
        digits=re.findall(r'\b[0-3]\b',plan['text'])
        result['plan']=plan
        result['planned_pair']=[int(x) for x in digits] if len(digits)==2 and len(set(digits))==2 else None
        result['arms']['overview']=generate([overview],['Low-resolution overview.'],question,32)
        result['arms']['question_only']=generate([],[],question,32)
        result['arms']['full_original']=generate([im],['Full original image.'],question,32)
        subsets=[(i,) for i in range(4)]+[p for a,b in itertools.combinations(range(4),2) for p in ((a,b),(b,a))]+[(0,1,2,3)]
        for subset in subsets:
            key='_'.join(map(str,subset))
            result['arms'][key]=generate([overview]+[crops[i] for i in subset],['Low-resolution overview.']+[labels[i] for i in subset],question,32)
        write(dest,result)
        print('DONE',r['id'],'plan',result['planned_pair'],'answers',{k:v['text'] for k,v in result['arms'].items()},flush=True)
        done+=1
        if limit and done>=limit: break
    print('RUN_FINISHED',done,'seconds',round(time.perf_counter()-started,2),flush=True)


def evaluate():
    cfg=json.loads((RUN/'manifest.json').read_text());gold=json.loads((RUN/'evaluation_only.json').read_text())
    rows=[];pairs=list(itertools.combinations(range(4),2))
    for path in sorted((RUN/'cases').glob('*.json')):
        r=json.loads(path.read_text());truth=normalize(gold[r['id']]['answer'])
        correct={k:normalize(v['text'])==truth for k,v in r['arms'].items()}
        stable=[(a,b) for a,b in pairs if correct[f'{a}_{b}'] and correct[f'{b}_{a}']]
        best_single=any(correct[str(i)] for i in range(4))
        planned=r['planned_pair'];planned_key='_'.join(map(str,planned)) if planned else None
        ranked=sorted(range(4),key=lambda i:(-r['arms'][str(i)]['mean_generated_logp'],i))[:2]
        confidence_key='_'.join(map(str,ranked))
        rows.append(dict(id=r['id'],stratum=gold[r['id']]['stratum'],gold=gold[r['id']]['answer'],correct=correct,
                         best_single=best_single,best_stable_pair=bool(stable),stable_pairs=stable,
                         strict_complementarity=not correct['overview'] and not best_single and bool(stable),
                         planned_pair=planned,planned_correct=correct.get(planned_key,False),
                         confidence_pair=ranked,confidence_correct=correct[confidence_key],
                         order_sensitive_pairs=sum(correct[f'{a}_{b}']!=correct[f'{b}_{a}'] for a,b in pairs),
                         total_answer_seconds=sum(v['seconds'] for v in r['arms'].values()),plan_seconds=r['plan']['seconds']))
    summary=dict(experiment=cfg['experiment'],completed=len(rows),planned=len(cfg['selected']),
                 strict_complementarity_cases=sum(r['strict_complementarity'] for r in rows),
                 best_single_correct=sum(r['best_single'] for r in rows),
                 best_stable_pair_correct=sum(r['best_stable_pair'] for r in rows),
                 generic_planner_correct=sum(r['planned_correct'] for r in rows),
                 invalid_plans=sum(r['planned_pair'] is None for r in rows),
                 expensive_confidence_pair_correct=sum(r['confidence_correct'] for r in rows),
                 order_sensitive_pairs=sum(r['order_sensitive_pairs'] for r in rows),
                 total_answer_seconds=sum(r['total_answer_seconds'] for r in rows),
                 total_plan_seconds=sum(r['plan_seconds'] for r in rows))
    for key in ('overview','question_only','full_original','0_1_2_3'):
        summary[key+'_correct']=sum(r['correct'][key] for r in rows)
    write(RUN/'evaluation.json',dict(summary=summary,rows=rows));print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run','evaluate']);p.add_argument('--limit',type=int,default=0);a=p.parse_args()
    if a.stage=='prepare':prepare()
    elif a.stage=='run':run(a.limit)
    else:evaluate()
