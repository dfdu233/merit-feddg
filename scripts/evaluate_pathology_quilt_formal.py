"""Current main ANCHOR scoring, only after exact full PathVQA completion."""
import argparse
import json
import sys
from pathlib import Path

from run_pathology_quilt_formal import read_json, write_new, digest, file_sha, verify, ARMS


def evaluate_native(root):
    """Score the separate native dual-lane run, never the historical five arms."""
    import copy
    from run_pathology_quilt_formal import cached
    from run_pathology_quilt_pilot import scorer_identity
    source=Path('/home/dbw/merit-feddg-pathology-quilt/runs/pathology-quilt-formal-budgeted-v2')
    frozen=read_json(source/'frozen.json')
    assert scorer_identity('/home/dbw/ANCHOR')==frozen['scorer'], 'scorer drift'
    expected=frozen['full_ids'];assert len(expected)==len(set(expected))==6719
    protocols=[];records={};case_hashes={}
    for lane in range(2):
        directory=root/f'shard{lane}'
        p=read_json(directory/'protocol.json');done=read_json(directory/'summary.json')
        scripts=Path(__file__).resolve().parent
        assert p['identity']==digest({'source':p['source'],'specs':p['specs'],'files':{
            name:file_sha(scripts/name) for name in ('canary_native_quilt.py','native_quilt_factory.py','native_quilt_infer.py')}}), 'native source identity drift'
        assert p['full_evaluation'] and p['source']==digest(frozen)
        assert p['shard_count']==2 and p['shard_index']==lane
        assert p['cases']==expected[lane::2]
        assert done['complete'] and done['identity']==p['identity']
        files={v.stem:v for v in (directory/'cases').glob('*.json')}
        assert set(files)==set(p['cases']) and not records.keys() & files.keys()
        for key,path in files.items():
            record=read_json(path)
            assert record['id']==key and record['identity']==p['identity']
            assert 'merit_quilt' in record['outputs']
            records[key]=record;case_hashes[key]=file_sha(path)
        comparable=copy.deepcopy(p)
        for key in ('identity','cases','shard_count','shard_index'):comparable.pop(key)
        for key in ('output','gpu_uuid'):comparable['specs']['quilt_pathology']['factory_kwargs'].pop(key)
        protocols.append(comparable)
    assert protocols[0]==protocols[1], 'scientific configuration differs across shards'
    assert set(records)==set(expected)
    refs=read_json('/home/dbw/ANCHOR/data/pathvqa/official_test_v1.json')
    references={str(r.get('question_id',r.get('qid',r.get('id')))):r for r in refs}
    assert set(references)==set(expected)
    sys.path.insert(0,'/home/dbw/ANCHOR')
    from anchor.medeval.evaluate_mixed_vqa_table import score
    from anchor.medeval.qualify_oe_generation import has_repetition_loop
    outputs=[records[k]['outputs']['merit_quilt'] for k in expected]
    baseline=[]
    for key in expected:
        value,path=cached(key)
        assert file_sha(path)==frozen['native_cache_hashes'][str(path)]
        baseline.append(value)
    metrics={name:score([dict(references[k],text=v['text']) for k,v in zip(expected,values)])
             for name,values in [('incumbent',baseline),('merit_quilt',outputs)]}
    tools=[t for v in outputs for t in v['trace'] if t.get('event')=='tool' and t.get('expert')=='quilt_pathology']
    report={'n':len(expected),'protocol':protocols[0],'metrics':metrics,
        'coverage':{'quilt_called':len(tools),'quilt_executed':sum(bool(t['executed']) for t in tools),
                    'quilt_adopted':sum(bool(t['adopted']) for t in tools),
                    'incumbent_reused_outside_scope':sum(bool(r.get('reused_outside_quilt_scope')) for r in records.values())},
        'quality':{'empty':sum(not v['text'].strip() for v in outputs),
                   'repetition':sum(has_repetition_loop(v['text']) for v in outputs),
                   'actor_at_1024_cap':sum(len(v['token_ids'])>=1024 for v in outputs)},
        'case_sha256':case_hashes,'scorer':scorer_identity('/home/dbw/ANCHOR'),
        'comparison_note':'New method uses8192 input budget vs historical incumbent2048; not a context-matched ablation. Raw results retained including failures; coverage is not quality certification.'}
    write_new(root/'evaluation-native-main.json',report)
    print('NATIVE FULL EVALUATION',json.dumps({'n':6719,'metrics':metrics,'quality':report['quality']}),flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--native',action='store_true')
    args=p.parse_args();root=args.run
    if args.native:
        evaluate_native(root)
        return
    f = read_json(root/'frozen.json')
    verify(f)
    complete = read_json(root/'complete.json')
    assert complete['identity'] == digest(f) and complete['full_dataset_complete']
    from run_pathology_quilt_pilot import scorer_identity
    assert scorer_identity('/home/dbw/ANCHOR') == f['scorer']
    paths = {p.stem: p for p in (root/'cases').glob('*.json')}
    assert set(paths) == set(f['full_ids']) and len(paths) == 6719
    rows = {r['id']: r for r in map(json.loads, Path(f['manifest']).read_text().splitlines())}
    records = {k: read_json(p) for k,p in paths.items()}
    assert all(v['id'] == k and v['identity'] == digest(f) and set(v['arms']) == set(ARMS) for k,v in records.items())
    reference_path = Path('/home/dbw/ANCHOR/data/pathvqa/official_test_v1.json')
    refs = {str(r.get('qid',r.get('id'))): r for r in read_json(reference_path)}
    assert set(refs) == set(rows)
    sys.path.insert(0,'/home/dbw/ANCHOR')
    from anchor.medeval.evaluate_mixed_vqa_table import score, answer_token_recall, source_task_group
    from anchor.corrected_sgta.evaluate_medheval_answers import evaluate_rows, PROTOCOL_VERSION
    from merit_feddg.agent_evaluate import cluster_bootstrap
    scores, metrics = {}, {}
    for arm in ARMS:
        scored = [dict(refs[k], text=records[k]['arms'][arm]['text']) for k in f['full_ids']]
        metrics[arm] = score(scored)
        details = evaluate_rows(scored)['details']
        scores[arm] = {k: float(d['correct']) if source_task_group(r)=='ce' else answer_token_recall(d['text'],d['gt_ans'])
                      for k,r,d in zip(f['full_ids'],scored,details)}
        assert abs(sum(scores[arm].values())/6719-metrics[arm]['primary_unified']['score']) < 1e-12
    def paired(arm, control, keys):
        ds = [scores[arm][k]-scores[control][k] for k in keys]
        return {'n':len(ds),'mean_delta':sum(ds)/len(ds) if ds else None,
                'improved':sum(d>0 for d in ds),'harmed':sum(d<0 for d in ds)}
    active = [k for k in f['full_ids'] if records[k]['status']=='real_candidate']
    eligible = {r['id'] for r in f['rows']}
    report = {'identity':digest(f),'n':6719,'candidate_n':len(active),
        'scorer':PROTOCOL_VERSION,'scorer_hashes':f['scorer'],
        'reference_sha256':file_sha(reference_path),'metric':'mixed CLOSED accuracy / OPEN token recall; not clinical accuracy',
        'arms':{},'per_case_scores':scores,
        'eligible_n':len(eligible), 'matched_delivery_n':len(active),
        'wrong_image_delivery_n':sum(bool(r.get('delivery',{}).get('compact_wrong_image')) for r in records.values()),
        'not_delivered_n':sum(r['status']=='quilt_not_delivered' for r in records.values()),
        'eligible_candidate_coverage':len(active)/len(eligible),
        'full_candidate_coverage':len(active)/6719,
        'transport_policy':f.get('transport_policy','strict'),
        'actor_calls':sum(r['new_actor_calls'] for r in records.values()),
        'actor_row_wall_seconds':sum(r['wall_seconds'] for r in records.values()),
        'inherited_incumbent_seconds':sum(r['arms']['compact']['seconds'] for r in records.values()),
        'actor_load_seconds':sum(read_json(p)['seconds'] for p in root.glob('actor_load_*.json')),
        'quilt_canary_cost':{k:v for k,v in read_json(root/'quilt_canary.json').items() if k not in ('predictions','model')},
        'quilt_cost':{k:v for k,v in read_json(root/'quilt_predictions.json').items() if k not in ('predictions','model')},
        'outside_scope':'incumbent reused for all arms, including quilt_alone; not expert-alone predictions outside eligibility'}
    for arm in ARMS:
        report['arms'][arm] = {'metrics':metrics[arm],'vs_compact':paired(arm,'compact',f['full_ids']),
            'active_vs_compact':paired(arm,'compact',active),
            'vs_wrong_image':paired(arm,'compact_wrong_image',active),
            'image_bootstrap_vs_compact':cluster_bootstrap([scores[arm][k]-scores['compact'][k] for k in f['full_ids']],
                [rows[k]['image_sha256'] for k in f['full_ids']])}
    assert scorer_identity('/home/dbw/ANCHOR') == f['scorer']
    write_new(root/'evaluation-main.json',report)
    print('FULL EVALUATION COMPLETE',root/'evaluation-main.json',flush=True)


if __name__ == '__main__':
    main()
