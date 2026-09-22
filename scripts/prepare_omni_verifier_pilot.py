"""Fixed SHA-ordered Huatuo Omni pilot, reusing frozen raw generations."""
import concurrent.futures
import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

old = Path('/home/dbw/merit-feddg-huatuo-spatial')
parser=argparse.ArgumentParser()
parser.add_argument('--limit',type=int,default=256)
parser.add_argument('--run-dir',default='runs/test-omni256')
parser.add_argument('--reuse-pilot',action='store_true')
args=parser.parse_args()
assert args.limit>=0
manifest = [json.loads(x) for x in (old/'runs/llava-test-v1/omnimedvqa-manifest.jsonl').read_text().splitlines()]
snapshot = json.loads((old/'reports/omnimedvqa-matched-indices-v14-20260922.json').read_text())
owners = {manifest[i]['id']:port for port, entries in snapshot.items() for entry in entries if entry['model']=='huatuo' for i in entry['indices']}
rows = sorted((x for x in manifest if x['id'] in owners), key=lambda x:hashlib.sha256(x['id'].encode()).hexdigest())
if args.limit:rows=rows[:args.limit]
connections = {'40297':['ssh','anchor-autodl'], '44450':['ssh','-i','/root/.ssh/anchor_autodl_ed25519','-p','44450','root@connect.nmb1.seetacloud.com'], '42865':['ssh','-i','/root/.ssh/anchor_autodl_ed25519','-p','42865','root@connect.nmb1.seetacloud.com'], '51493':['ssh','-i','/root/.ssh/merit_5090_ed25519','-p','51493','root@connect.weste.seetacloud.com']}
def fetch(port):
    ids = [x['id'] for x in rows if owners[x['id']]==port]
    code = '''import pathlib,gzip,json
r=pathlib.Path('/home/dbw/merit-feddg-huatuo-spatial/runs/huatuo-test-v1')
def read(p):
 f=p.with_suffix('.json.gz');return json.loads(gzip.decompress(f.read_bytes())) if f.exists() else json.loads(p.with_suffix('.json').read_text())
out={}
for sid in IDS:
 paths=[p for p in r.glob('omnimedvqa-cloudPORT*/'+sid) if 'repair' not in str(p)]
 assert len(paths)==1,(sid,paths)
 p=paths[0];v=read(p/'provenance');a=read(p/'bard')
 out[sid]={'text':a['text'],'row':v['row'],'prompt':v['prompt'],'source':str(p)}
print(json.dumps(out))
'''.replace('IDS',repr(ids)).replace('PORT',port)
    python='/root/autodl-tmp/merit-env/bin/python' if port=='51493' else '/home/dbw/.runtime/miniconda3/envs/huatuo/bin/python'
    return json.loads(subprocess.check_output(connections[port]+[python+' -'],input=code,text=True,timeout=120))
with concurrent.futures.ThreadPoolExecutor(4) as pool:
    candidates={k:v for group in pool.map(fetch,connections) for k,v in group.items()}
bp=Path('/home/dbw/ANCHOR/corrected_runs/paper_baselines_v1/huatuo_final_v1/full/omnimedvqa/greedy/answers.jsonl')
base={x['question_id']:x for x in map(json.loads,bp.read_text().splitlines()) if x['question_id'] in candidates}
for row in rows:
    sid=row['id'];c=candidates[sid];b=base[sid]
    assert c['row']['image_sha256']==row['image_sha256'] and c['prompt']==row['benchmark_prompt']
    assert b['metadata']['effective_img_name']==row['image']
    row.update(modality=c['row']['modality'],group_id=row['image_sha256'],baseline=b['text'],candidate=c['text'])
    options=dict(re.findall(r'^([A-Z])\. (.+)$',row['benchmark_prompt'],re.M))
    for key in ['baseline','candidate']:
        match=re.fullmatch(r'\s*\(?([A-Z])\)?[.。]?\s*',row[key])
        if match and match[1] in options:row['evidence_'+key]=options[match[1]]
        else:row['verification_skip']='answer-not-an-unambiguous-option-label'
out=Path(args.run_dir);out.mkdir(exist_ok=True)
assert not (out/'inputs.json').exists(), 'Do not overwrite an existing frozen pilot'
(out/'inputs.json').write_text(json.dumps(rows))
(out/'input-provenance.json').write_text(json.dumps({'selection':'SHA256(id) over frozen completed Huatuo snapshot; no outcome selection','limit':args.limit,'n':len(rows),'baseline':str(bp),'candidate_sources':{k:v['source'] for k,v in candidates.items()},'routing':'unchanged original model-inferred; known modality errors retained for faithful comparison','unqualified_exploratory':True},indent=2))
if args.reuse_pilot:
    import shutil
    byid={x['id']:x for x in rows};pilot=Path('runs/test-omni256')
    (out/'cases').mkdir(exist_ok=True)
    for x in json.loads((pilot/'inputs.json').read_text()):
        assert x==byid[x['id']],x['id']
        src=pilot/'cases'/(x['id']+'.json');dst=out/'cases'/src.name
        shutil.copyfile(src,dst);assert src.read_bytes()==dst.read_bytes()
print('PREPARED',len(rows),'changed',sum(x['baseline']!=x['candidate'] for x in rows))
