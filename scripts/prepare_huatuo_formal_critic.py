"""Read-only verified reuse of formal VQA-RAD candidates; no model inference."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, '/home/dbw/ANCHOR')
from anchor.corrected_sgta.protocol_v2 import build_prompt


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError('Preserve previous controls')
    root = Path('/home/dbw/ANCHOR')
    baseline = root/'corrected_runs/paper_baselines_v1/huatuo_final_v1/full/vqa_rad/greedy'
    merit = Path('/home/dbw/merit-feddg-expert-coverage/runs/huatuo-merit-full-v1/vqa_rad')
    manifest = root/'corrected_runs/paper_baselines_v1/merit_common_protocol_v1/vqa_rad.jsonl'
    canonical = root/'data/vqa_rad/official_test_full_v1.json'
    read = lambda p: json.loads(p.read_text())
    mp, bp = read(merit/'protocol.json'), read(baseline/'generation_config.json')
    rows = [json.loads(s) for s in manifest.read_text().splitlines()]
    gold = {r['id']:r for r in read(canonical)}
    bs = {r['question_id']:r for r in map(json.loads, (baseline/'answers.jsonl').read_text().splitlines())}
    ms = {p.stem:read(p) for p in (merit/'cases').glob('*.json')}
    ids = {r['id'] for r in rows}
    if len(rows) != 451 or len(ids) != 451 or ids != set(gold) or ids != set(bs) or ids != set(ms):
        raise ValueError('Require identical full451 unique ID sets')
    if read(merit/'actor-complete.json')['identity'] != mp['identity']:
        raise ValueError('Incomplete original MERIT')
    if bp['manifest_sha256'] != sha(canonical) or mp['manifest_sha256'] != sha(manifest):
        raise ValueError('Original manifest changed')
    if bp['max_new_tokens'] != 1024 or mp['generation_config']['max_new_tokens'] != 1024:
        raise ValueError('Official answer budgets differ')
    inputs = [canonical, manifest, baseline/'answers.jsonl', baseline/'generation_config.json',
              merit/'protocol.json', merit/'actor-complete.json', Path(__file__).resolve()]
    sources = {str(p):sha(p) for p in inputs}
    sources.update(mp['entry_sources'])
    for path, digest in (sources | mp['formal_sources'] | mp['base']['huatuo_canary_sources']).items():
        if sha(path) != digest:
            raise ValueError('Frozen input/source changed: '+path)
    for row in rows:
        k = row['id']; b = bs[k]; m = ms[k]; g = gold[k]
        if set(row) & {'answer','answers','label','reference','references'}:
            raise ValueError('References in inference manifest')
        if row['question'] != g['question'] or row['benchmark_prompt'] != build_prompt(g):
            raise ValueError('Formal question/prompt mismatch')
        if sha(row['image']) != row['image_sha256'] or row['image_sha256'] != g['image_sha256']:
            raise ValueError('Image identity mismatch')
        meta = b['metadata']
        if meta['img_name'] != g['img_name'] or meta['effective_img_name'] != g['img_name']:
            raise ValueError('Baseline image differs')
        if meta['fingerprint'] != bp['fingerprint'] or meta['effective_max_new_tokens'] != 1024 or meta.get('input_error'):
            raise ValueError('Invalid original Baseline')
        if m['identity'] != mp['identity'] or m['generation_config'] != mp['generation_config']:
            raise ValueError('Invalid original MERIT')
        if any('runtime_error' in str(t.get('reason','')) for t in m['output']['trace']):
            raise ValueError('Original expert failure')
        sources[str(merit/'cases'/(k+'.json'))] = sha(merit/'cases'/(k+'.json'))
    cfg = {'rows':rows, 'base':mp['base'], 'source':sources,
           'formal_sources':mp['formal_sources'], 'formal_reuse':True,
           'dataset':'vqa_rad', 'split':'official_test',
           'baseline_fingerprint':bp['fingerprint'], 'merit_identity':mp['identity'],
           'inherited_costs':'not available in these raw exports; not zero',
           'no_new_patient_answer_generation':True}
    identity = hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
    (a.output/'cases').mkdir(parents=True)
    def write(p,v):
        p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
    write(a.output/'protocol.json',dict(identity=identity,**cfg))
    for row in rows:
        k=row['id']; b=bs[k]
        write(a.output/'cases'/(k+'.json'), {'id':k,'identity':identity,'complete':True,
            'arms':{'generalist':{'text':b['text'],'token_ids':b['metadata']['generated_token_ids'],
                                  'reuse':'official_baseline','new_answer_calls':0},
                    'compact':dict(ms[k]['output'],reuse='official_merit',new_answer_calls=0)}})
    write(a.output/'references.json',{k:[str(gold[k]['answer'])] for k in ids})
    write(a.output/'complete.json',{'identity':identity,'n':451,'reused_only':True})
    print('FORMAL REUSE VERIFIED',identity,451)


if __name__ == '__main__':
    main()
