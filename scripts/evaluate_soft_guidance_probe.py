"""Offline paired probe scoring; cannot publish this as a full-test result."""
import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

from run_soft_guidance_pilot import SOURCES

from merit_feddg.agent_evaluate import cluster_bootstrap

sys.path.insert(0,'/home/dbw/ANCHOR')
from anchor.corrected_sgta.evaluate_medheval_answers import PROTOCOL_VERSION, evaluate_rows
from anchor.medeval.evaluate_mixed_vqa_table import answer_token_recall


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='append',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    records=[]
    for run in args.run:
        for path in Path(run).glob('*test-*.json'):
            record=json.loads(path.read_text())
            if not record['parity']:
                raise ValueError('parity failure is an engineering failure, not a score')
            records.append(record)
    if len({r['id'] for r in records})!=len(records):
        raise ValueError('duplicate cases across probes')
    result={'scorer':PROTOCOL_VERSION,'full_dataset_complete':False,
            'metric':'decoded strict CLOSED + unchanged ANCHOR OPEN answer-token-recall',
            'scorer_sha256':hashlib.sha256(Path('/home/dbw/ANCHOR/anchor/corrected_sgta/evaluate_medheval_answers.py').read_bytes()).hexdigest(),
            'datasets':{}}
    arms=['generalist_cached','compact_rows_cached','other_text_only','text_soft','native_spatial_soft']
    for dataset,(_,data) in SOURCES.items():
        root=Path('/home/dbw/merit-feddg/runs')/data
        manifest={r['id']:r for r in map(json.loads,(root/'manifest.jsonl').read_text().splitlines())}
        refs=json.loads((root/'references.json').read_text())
        subset=[r for r in records if r['dataset']==dataset]
        eligible=[r for r in subset if all(a in r for a in arms)]
        scores={}
        for arm in arms:
            rows=[{'qid':r['id'],'question':manifest[r['id']]['question'],
                   'answer_type':manifest[r['id']]['answer_type'],'answer':refs[r['id']][0],
                   'text':r[arm]['text']} for r in eligible]
            details={d['question_id']:d for d in evaluate_rows(rows)['details']}
            scores[arm]={r['id']:(float(details[r['id']]['correct']) if manifest[r['id']]['answer_type']=='closed'
                         else answer_token_recall(r[arm]['text'],refs[r['id']][0])) for r in eligible}
        report={'n':len(eligible),'selected':len(subset),'unavailable':len(subset)-len(eligible),
                'per_case':scores,'arms':{}}
        for arm in arms:
            deltas=[scores[arm][r['id']]-scores['compact_rows_cached'][r['id']] for r in eligible]
            if not deltas:
                continue
            bad=[r['id'] for r in eligible if scores['generalist_cached'][r['id']]>scores['compact_rows_cached'][r['id']]]
            report['arms'][arm]={'mean':statistics.mean(scores[arm].values()),
                'improved_vs_compact':sum(d>0 for d in deltas),'harmed_vs_compact':sum(d<0 for d in deltas),
                'historical_bad_cases':len(bad),
                'historical_bad_improved':sum(scores[arm][k]>scores['compact_rows_cached'][k] for k in bad),
                'mean_decode_seconds':statistics.mean(r[arm].get('seconds',0) for r in eligible),
                'cached_time_not_remeasured':arm.endswith('_cached'),
                'changed_text':sum(r[arm]['text']!=r['compact_rows_cached']['text'] for r in eligible),
                'image_bootstrap':cluster_bootstrap(deltas,[manifest[r['id']]['image_sha256'] for r in eligible]),
                'ci_not_population_inference':'tiny, enriched diagnostic subset'}
        result['datasets'][dataset]=report
    with Path(args.output).open('x') as f: json.dump(result,f,indent=2)
    print(json.dumps({k:v['arms'] for k,v in result['datasets'].items()},indent=2))


if __name__=='__main__':
    main()
