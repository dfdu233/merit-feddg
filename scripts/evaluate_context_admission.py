"""Frozen ANCHOR offline scoring for complete terminal audits / full TRAIN runs."""
import argparse
import collections
import json
import statistics
import sys
from pathlib import Path

from run_anchored_revision_train import DATA, read, sha

from merit_feddg.agent_evaluate import cluster_bootstrap
from merit_feddg.open_study import atomic_json, fingerprint


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--revision-terminal', action='store_true')
    p.add_argument('--scorer-pin', type=Path,
                   help='Explicit separately recorded rescoring protocol; original run stays immutable')
    a = p.parse_args()
    cfg = read(a.run / 'frozen.json')
    identity = fingerprint(cfg)
    complete = read(a.run / 'complete.json')
    if complete['identity'] != identity:
        raise ValueError('Completion identity mismatch')
    source = cfg['prior'] if a.revision_terminal else cfg['source']
    ids = source['selected'] if a.revision_terminal else cfg['selected']
    expected_arms = source['arms'] if a.revision_terminal else cfg['arms']
    if any(sha(k) != h for k,h in source['input_hashes'].items()):
        raise ValueError('Frozen inputs changed')
    scoring = read(a.scorer_pin)['scorers'] if a.scorer_pin else source['scorers']
    if set(scoring) != set(source['scorers']):
        raise ValueError('Scorer pin must cover the exact scorer set')
    if any(sha(Path('/home/dbw/ANCHOR')/k) != h for k,h in scoring.items()):
        raise ValueError('Frozen primary scorer changed')
    paths = {p.stem:p for p in (a.run/'cases').glob('*.json')}
    if set(paths) != set(ids):
        raise ValueError('Exact selected ID set required; do not impute missing cases')
    cases = {k:read(paths[k]) for k in ids}
    if any(not r['complete'] or r['identity'] != identity or set(r['arms']) != set(expected_arms)
           for r in cases.values()):
        raise ValueError('Incomplete/incompatible case arms')
    rows = {r['id']:r for r in map(json.loads,Path(source['manifest']).read_text().splitlines())}
    references = read(DATA/'train/references.json')
    if set(references) != set(rows):
        raise ValueError('Full TRAIN reference alignment mismatch')
    sys.path.insert(0,'/home/dbw/ANCHOR')
    from anchor.corrected_sgta.evaluate_medheval_answers import PROTOCOL_VERSION, evaluate_rows
    from anchor.medeval.evaluate_mixed_vqa_table import answer_token_recall
    scores, unavailable = {}, {}
    for arm in expected_arms:
        unavailable[arm] = [k for k in ids if cases[k]['arms'][arm]['text'] is None]
        values = evaluate_rows([{'qid':k,'question':rows[k]['question'],
            'answer_type':rows[k]['answer_type'],'answer':references[k][0],
            'text':cases[k]['arms'][arm]['text'] or ''} for k in ids])['details']
        details = {v['question_id']:v for v in values}
        scores[arm] = {k: (0.0 if k in unavailable[arm] else
            float(details[k]['correct']) if rows[k]['answer_type']=='closed' else
            answer_token_recall(cases[k]['arms'][arm]['text'],references[k][0])) for k in ids}

    def paired(arm, ref, keys):
        d = [scores[arm][k]-scores[ref][k] for k in keys]
        return {'n':len(keys),'delta':statistics.mean(d) if d else None,
                'improved':sum(v>0 for v in d),'harmed':sum(v<0 for v in d),
                'image_ci95':cluster_bootstrap(d,[rows[k]['image_sha256'] for k in keys]) if d else None}

    gain = [k for k in ids if scores['compact'][k]>scores['generalist'][k]]
    harm = [k for k in ids if scores['compact'][k]<scores['generalist'][k]]
    report = {'identity':identity,'n':len(ids),'full_train':not a.revision_terminal,
        'test_benchmark':False,'scorer':PROTOCOL_VERSION,'scorers':scoring,
        'original_scorers':source['scorers'],
        'explicit_rescore_pin':str(a.scorer_pin) if a.scorer_pin else None,
        'references_sha256':sha(DATA/'train/references.json'),
        'metric':'Pinned current ANCHOR CLOSED parser + OPEN answer-token-recall; not clinical accuracy',
        'unavailable_scoring':'Explicit engineering-unavailable arms get 0 only in failure-inclusive '
            'task score. No answer is filled in the raw output; available-only paired cohort also reported.',
        'original_compact_gain_cases':len(gain),'original_compact_harm_cases':len(harm),
        'arms':{},'gates':{},'cost':{},'per_case_scores':scores}
    for arm in expected_arms:
        values = [cases[k]['arms'][arm] for k in ids]
        usable = [k for k in ids if k not in unavailable[arm]]
        report['arms'][arm] = {
            'failure_inclusive_score':statistics.mean(scores[arm].values()),
            'unavailable':len(unavailable[arm]),
            'empty':sum(v['text']=='' for v in values),
            'vs_generalist':paired(arm,'generalist',ids),'vs_compact':paired(arm,'compact',ids),
            'available_only_vs_compact':paired(arm,'compact',usable),
            'text_changes_from_generalist':sum(v['text'] != cases[k]['arms']['generalist']['text']
                for k,v in zip(ids,values)),
            'original_gains_retained':sum(scores[arm][k]>=scores['compact'][k] for k in gain),
            'original_harms_recovered':sum(scores[arm][k]>=scores['generalist'][k] for k in harm),
            'new_answer_calls':sum(v['new_calls'] for v in values),
            'new_answer_seconds':sum(v['new_seconds'] for v in values),
            'reuse_counts':dict(collections.Counter(str(v.get('reuse',v.get('reused'))) for v in values)),
        }
    if not a.revision_terminal:
        for policy in ('relevance','scope'):
            judgments = [v for r in cases.values() for v in r['gates'][policy]]
            report['gates'][policy] = {
                'labels':dict(collections.Counter(v['label'] for v in judgments)),
                'reasons':dict(collections.Counter(v['reason'] for v in judgments)),
                'real_model_calls':sum(v['calls'] for v in judgments),
                'seconds':sum(v['seconds'] for v in judgments),
                'packet_count':len(judgments),'correctness_accuracy_measured':False}
    report['cost'] = {
        'new_expert_calls':0,'real_native_cache_reused':True,
        'load_seconds':sum(read(p)['seconds'] for p in a.run.glob('load-*.json')),
        'new_answer_seconds':sum(v['new_answer_seconds'] for v in report['arms'].values()),
        'gate_seconds':sum(v['seconds'] for v in report['gates'].values()),
        'inherited_seconds_by_arm':{arm:sum(cases[k]['arms'][arm].get('inherited_seconds',0)
                                          for k in ids) for arm in expected_arms},
        'canary_parity_seconds':sum(v['seconds'] for r in cases.values() for v in r.get('parity',{}).values()),
        'note':'Shared warm inference with concurrent external jobs; not isolated end-to-end throughput.'}
    if any(sha(Path('/home/dbw/ANCHOR')/k) != h for k,h in scoring.items()):
        raise ValueError('Scorer changed during evaluation')
    target = a.run/('evaluation-current-official.json' if a.scorer_pin else 'evaluation.json')
    if target.exists():
        raise FileExistsError('Do not overwrite prior evaluation')
    atomic_json(target,report)
    print(json.dumps({k:v for k,v in report.items() if k!='per_case_scores'},indent=2))


if __name__ == '__main__':
    main()
