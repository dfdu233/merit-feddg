"""Frozen real-image development pilot. No gold labels read by inference."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DATA = Path('/home/dbw/data/SLAKE')
RUN = ROOT / 'runs/slake-relational-evidence-dev-v1'
MODEL = '/home/dbw/models/Qwen2.5-VL-7B-Instruct'
WORDS = re.compile(r'\b(left|right|above|below|behind|front|between|largest|biggest|bigger|smaller|closest|next to|relative)\b', re.I)
STRATA = [('CT', 'abdomen'), ('CT', 'chest'), ('CT', 'brain'), ('MRI', 'abdomen'), ('MRI', 'brain'), ('X-Ray', 'chest')]
OBSERVE = '''Describe only visible image evidence. Return one JSON object, no prose or markdown.
Schema: {"entities":[{"id":"e1","name":"organ or finding","attributes":["short visible property"]}],"relations":[{"head":"e1","relation":"short spatial or relative-size relation","tail":"e2"}]}.
Use at most 6 entities and 6 relations. Use only declared entity IDs. Include visible spatial and relative-size relationships when discernible. Distinguish image-left from patient-left. Do not invent unobservable relationships. Do not diagnose from general knowledge. Attributes should be short. No extra fields.'''
ANSWER = 'Answer the medical image question with only a short answer, without explanation. Observations, when provided, are fallible tool outputs. Use the image when available.\n'


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def digest(image):
    im = image.convert('RGB')
    return hashlib.sha256(str(im.size).encode() + im.tobytes()).hexdigest()


def group(row):
    location = row['location'].lower()
    anatomy = ('brain' if 'brain' in location else 'chest' if 'lung' in location or 'chest' in location else 'abdomen' if 'abdomen' in location else location)
    return row['modality'], anatomy


def prepare():
    from PIL import Image
    from io import BytesIO
    assert not RUN.exists(), 'fresh run required'
    splits = {s: json.loads((DATA / f'{s}.json').read_text()) for s in ('train', 'validation', 'test')}
    hashes = {}
    with zipfile.ZipFile(DATA / 'imgs.zip') as z:
        names = {r['img_name'] for rows in splits.values() for r in rows}
        for name in sorted(names):
            with Image.open(BytesIO(z.read('imgs/' + name))) as im:
                hashes[name] = digest(im)
        forbidden = {hashes[r['img_name']] for s in ('validation', 'test') for r in splits[s]}
        eligible = [r for r in splits['train'] if r['q_lang'] == 'en' and r['base_type'] == 'vqa'
                    and WORDS.search(r['question']) and group(r) in STRATA and hashes[r['img_name']] not in forbidden]
        eligible.sort(key=lambda r: hashlib.sha256(f"{r['qid']}:{r['img_name']}".encode()).hexdigest())
        selected, seen, counts, evaluation = [], set(), {g: 0 for g in STRATA}, {}
        for r in eligible:
            g, h = group(r), hashes[r['img_name']]
            if h in seen or counts[g] >= 4:
                continue
            key = f"slake-{r['qid']}"
            img = RUN / 'images' / f'{key}.png'
            img.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(BytesIO(z.read('imgs/' + r['img_name']))) as im:
                im.convert('RGB').save(img)
            selected.append(dict(id=key, image=str(img), question=r['question'], image_hash=h))
            evaluation[key] = dict(answer=r['answer'], stratum=list(g), content_type=r['content_type'])
            seen.add(h); counts[g] += 1
    cfg = dict(experiment='slake-relational-evidence-dev-v1', model=MODEL, observer_prompt=OBSERVE,
               answer_prompt=ANSWER, observer_tokens=512, answer_tokens=32, seed=197,
               code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
               gpu_uuid='GPU-3846413a-4238-d307-b1f3-10c2dfbe002c',
               selection_counts={str(k): v for k, v in counts.items()}, selected=selected,
               excluded_eval_image_hashes=len(forbidden), eligible_questions=len(eligible),
               split_hashes={s: hashlib.sha256((DATA/f'{s}.json').read_bytes()).hexdigest() for s in splits})
    write(RUN / 'manifest.json', cfg)
    write(RUN / 'evaluation_only.json', evaluation)
    print(json.dumps({'selected':len(selected), 'counts':cfg['selection_counts'], 'eligible':len(eligible)},indent=2),flush=True)


def parse_graph(text):
    start = text.find('{')
    if start < 0:
        raise ValueError('no JSON object')
    g, _ = json.JSONDecoder().raw_decode(text[start:])
    assert set(g) == {'entities', 'relations'}
    assert isinstance(g['entities'], list) and 0 < len(g['entities']) <= 8
    assert isinstance(g['relations'], list) and len(g['relations']) <= 12
    ids = set()
    for e in g['entities']:
        assert set(e) == {'id', 'name', 'attributes'}
        assert isinstance(e['id'], str) and e['id'] not in ids
        assert isinstance(e['name'], str) and isinstance(e['attributes'], list)
        assert all(isinstance(a, str) for a in e['attributes'])
        ids.add(e['id'])
    for r in g['relations']:
        assert set(r) == {'head', 'relation', 'tail'}
        assert r['head'] in ids and r['tail'] in ids and isinstance(r['relation'], str)
    return g


def table(g):
    # JSON-escaped cells preserve all string content and ID binding exactly.
    return '\n'.join(['ENTITY ' + json.dumps([e['id'], e['name'], e['attributes']],ensure_ascii=False) for e in g['entities']]
                     + ['RELATION ' + json.dumps([r['head'],r['relation'],r['tail']],ensure_ascii=False) for r in g['relations']])


def table_roundtrip(text):
    out = {'entities':[], 'relations':[]}
    for line in text.splitlines():
        kind, value = line.split(' ',1)
        a,b,c = json.loads(value)
        if kind == 'ENTITY': out['entities'].append(dict(id=a,name=b,attributes=c))
        else: out['relations'].append(dict(head=a,relation=b,tail=c))
    return out


def corrupt(g):
    out = copy.deepcopy(g)
    tails = [r['tail'] for r in g['relations']]
    if len(set(tails)) < 2: return out, 0
    shifted = tails[1:]+tails[:1]
    for r,t in zip(out['relations'],shifted): r['tail'] = t
    return out, sum(a != b for a,b in zip(tails,shifted))


def run(limit):
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    cfg = json.loads((RUN / 'manifest.json').read_text())
    assert cfg['code_sha256'] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    uuid = subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    assert uuid == cfg['gpu_uuid']
    torch.manual_seed(cfg['seed'])
    started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(MODEL, min_pixels=256*28*28, max_pixels=512*28*28, local_files_only=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                device_map={'':'cuda:0'}, attn_implementation='sdpa', local_files_only=True).eval()
    print('LOADED', round(time.perf_counter()-started,2),flush=True)

    def generate(prompt, image, tokens):
        content = ([{'type':'image','image':image}] if image is not None else []) + [{'type':'text','text':prompt}]
        rendered = processor.apply_chat_template([{'role':'user','content':content}], tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[rendered], images=[image] if image is not None else None, return_tensors='pt').to('cuda')
        n = inputs['input_ids'].shape[1]
        torch.cuda.synchronize(); t = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(**inputs,max_new_tokens=tokens,do_sample=False,use_cache=True)
        torch.cuda.synchronize()
        return dict(text=processor.batch_decode(output[:,n:],skip_special_tokens=True)[0],
                    input_tokens=n, output_tokens=int(output.shape[1]-n), seconds=time.perf_counter()-t)

    done = 0
    for row in cfg['selected']:
        path = RUN / 'cases' / (row['id']+'.json')
        if path.exists(): continue
        image = Image.open(row['image']).convert('RGB')
        assert digest(image) == row['image_hash']
        observed = generate(cfg['observer_prompt'],image,cfg['observer_tokens'])
        result = dict(id=row['id'],observer=observed,arms={})
        try:
            graph = parse_graph(observed['text'])
            flat = table(graph)
            assert table_roundtrip(flat) == graph
            wrong, changed = corrupt(graph)
            assert table_roundtrip(table(wrong)) == wrong
            result.update(graph=graph,changed_relations=changed,parse_valid=True)
        except (AssertionError,ValueError,KeyError,TypeError) as exc:
            graph=None
            result.update(parse_valid=False,parse_error=type(exc).__name__+': '+str(exc))
        variants = [('direct',None,image)]
        if graph is not None:
            variants += [('graph',json.dumps(graph,ensure_ascii=False),image),('table',flat,image),
                         ('wrong_graph',json.dumps(wrong,ensure_ascii=False),image),('wrong_table',table(wrong),image),
                         ('graph_no_image',json.dumps(graph,ensure_ascii=False),None)]
        for name,evidence,im in variants:
            prompt = cfg['answer_prompt']
            if evidence is not None: prompt += 'Tool observations:\n'+evidence+'\n'
            prompt += 'Question: '+row['question']
            result['arms'][name] = generate(prompt,im,cfg['answer_tokens'])
        write(path,result)
        print('DONE',row['id'],'parse',result['parse_valid'],'changed_edges',result.get('changed_relations'),
              {k:v['text'] for k,v in result['arms'].items()},flush=True)
        done += 1
        if limit and done >= limit: break
    print('RUN_FINISHED',done,'seconds',round(time.perf_counter()-started,2),flush=True)


def normalize(text):
    text = re.sub(r'[^\w\s]',' ',text.lower())
    text = re.sub(r'\b(a|an|the)\b',' ',text)
    return ' '.join(text.split())


def evaluate():
    cfg=json.loads((RUN/'manifest.json').read_text())
    gold=json.loads((RUN/'evaluation_only.json').read_text())
    results=[json.loads(p.read_text()) for p in sorted((RUN/'cases').glob('*.json'))]
    arms=('direct','graph','table','wrong_graph','wrong_table','graph_no_image')
    scored=[]
    for r in results:
        truth=normalize(gold[r['id']]['answer'])
        scored.append(dict(id=r['id'],stratum=gold[r['id']]['stratum'],parse_valid=r['parse_valid'],
                           changed_relations=r.get('changed_relations',0),
                           correct={a:normalize(v['text'])==truth for a,v in r['arms'].items()},
                           answers={a:v['text'] for a,v in r['arms'].items()},gold=gold[r['id']]['answer']))
    summary=dict(experiment=cfg['experiment'],completed=len(results),planned=len(cfg['selected']),
                 parse_valid=sum(r['parse_valid'] for r in results),
                 intervention_coverage=sum(r.get('changed_relations',0)>0 for r in results),arms={},pairs={})
    for a in arms:
        valid=[r for r in scored if a in r['correct']]
        summary['arms'][a]=dict(correct=sum(r['correct'][a] for r in valid),n=len(valid),
                               generation_seconds=sum(r['arms'].get(a,{}).get('seconds',0) for r in results))
    for a,b in [('graph','direct'),('graph','table'),('graph','wrong_graph'),('table','wrong_table')]:
        valid=[r for r in scored if a in r['correct'] and b in r['correct']]
        if 'wrong' in b: valid=[r for r in valid if r['changed_relations']>0]
        summary['pairs'][a+' vs '+b]=dict(n=len(valid),rescues=sum(r['correct'][a] and not r['correct'][b] for r in valid),
            harms=sum(not r['correct'][a] and r['correct'][b] for r in valid),
            answer_changes=sum(normalize(r['answers'][a])!=normalize(r['answers'][b]) for r in valid))
    summary['observer_seconds']=sum(r['observer']['seconds'] for r in results)
    write(RUN/'evaluation.json',dict(summary=summary,rows=scored))
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('stage',choices=['prepare','run','evaluate']);parser.add_argument('--limit',type=int,default=0)
    args=parser.parse_args()
    if args.stage=='prepare': prepare()
    elif args.stage=='run': run(args.limit)
    else: evaluate()
