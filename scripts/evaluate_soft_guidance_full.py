"""Offline full-manifest scoring with historical rescue/harm cohorts and fresh controls."""
import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

from run_soft_guidance_full import ARMS

from merit_feddg.agent_evaluate import cluster_bootstrap
from merit_feddg.open_study import atomic_json, fingerprint

SCORERS=('anchor/corrected_sgta/evaluate_medheval_answers.py',
         'anchor/medeval/evaluate_mixed_vqa_table.py')


def scorer_hashes():
    return {p:hashlib.sha256((Path('/home/dbw/ANCHOR')/p).read_bytes()).hexdigest() for p in SCORERS}


def paired(scores, reference, ids):
    deltas=[scores[k]-reference[k] for k in ids]
    return {'n':len(ids),'mean_delta':statistics.mean(deltas) if deltas else None,
            'score_improvements':sum(v>0 for v in deltas),'score_harms':sum(v<0 for v in deltas)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--freeze-scorer',action='store_true',help='Pin code now, do not load references or score')
    args=p.parse_args()
    root=args.run
    frozen=json.loads((root/'frozen.json').read_text())
    pin=root/'scorer-frozen.json'
    current=scorer_hashes()
    if args.freeze_scorer:
        if pin.exists() and json.loads(pin.read_text())!=current:
            raise RuntimeError('existing scorer pin differs; cannot replace')
        if not pin.exists(): atomic_json(pin,current)
        print('Scorer pinned; no references read, no scoring performed')
        return
    if not pin.exists() or json.loads(pin.read_text())!=current:
        raise RuntimeError('frozen scorer missing or changed')
    complete=json.loads((root/'complete.json').read_text())
    identity=fingerprint(frozen)
    if not complete.get('full_dataset_complete') or complete['identity']!=identity:
        raise RuntimeError('full completion required; no partial full-test score')
    if (root/'evaluation.json').exists():
        raise RuntimeError('evaluation already exists; do not overwrite')
    sys.path.insert(0,'/home/dbw/ANCHOR')
    from anchor.corrected_sgta.evaluate_medheval_answers import PROTOCOL_VERSION, evaluate_rows
    from anchor.medeval.evaluate_mixed_vqa_table import answer_token_recall
    report={'identity':identity,'full_dataset_complete':True,'scorer':PROTOCOL_VERSION,
            'metric':'decoded strict CLOSED + ANCHOR OPEN answer-token-recall',
            'scorer_hashes':current,'primary_control':'compact_matched',
            'historical_controls_are_not_current_environment_parity':True,'datasets':{}}
    for dataset,data in frozen['datasets'].items():
        rows=data['rows'];ids=[r['id'] for r in rows]
        files={p.stem:p for p in (root/dataset).glob('*.json')}
        if set(files)!=set(ids): raise RuntimeError('output ID mismatch')
        records={k:json.loads(files[k].read_text()) for k in ids}
        if any(r['identity']!=identity or set(r['arms'])!=set(ARMS) for r in records.values()):
            raise RuntimeError('arm/identity mismatch')
        refs=json.loads((Path(data['manifest']).parent/'references.json').read_text())
        if set(refs)!=set(ids): raise RuntimeError('reference ID mismatch')
        scores={}
        for arm in ARMS:
            evaluated=evaluate_rows([{'qid':r['id'],'question':r['question'],
                'answer_type':r['answer_type'],'answer':refs[r['id']][0],
                'text':records[r['id']]['arms'][arm]['text']} for r in rows])
            details={d['question_id']:d for d in evaluated['details']}
            scores[arm]={r['id']:(float(details[r['id']]['correct']) if r['answer_type']=='closed'
                           else answer_token_recall(records[r['id']]['arms'][arm]['text'],refs[r['id']][0])) for r in rows}
        prior_good=[k for k in ids if scores['historical_compact'][k]>scores['historical_generalist'][k]]
        prior_bad=[k for k in ids if scores['historical_compact'][k]<scores['historical_generalist'][k]]
        closed=[r['id'] for r in rows if r['answer_type']=='closed']
        opened=[r['id'] for r in rows if r['answer_type']=='open']
        ds={'n':len(ids),'applicable':sum(r['applicable'] for r in records.values()),
            'historical_correction_cases':len(prior_good),'historical_harm_cases':len(prior_bad),
            'historical_compact_token_drift':sum(not r['historical_compact_token_parity'] for r in records.values()),
            'historical_generalist_token_drift':sum(not r['historical_generalist_token_parity'] for r in records.values()),
            'new_expert_calls':sum(r['new_expert_calls'] for r in records.values()),
            'all_arm_wall_seconds':sum(r['wall_seconds'] for r in records.values()),'arms':{},
            'per_case_scores':scores,'historical_correction_ids':prior_good,'historical_harm_ids':prior_bad}
        for arm in ARMS:
            values=scores[arm]
            delta=[values[k]-scores['compact_matched'][k] for k in ids]
            ds['arms'][arm]={'mixed_score':statistics.mean(values.values()),
                'closed_accuracy':statistics.mean(values[k] for k in closed) if closed else None,
                'open_token_recall':statistics.mean(values[k] for k in opened) if opened else None,
                'vs_fresh_compact':paired(values,scores['compact_matched'],ids),
                'vs_fresh_generalist':paired(values,scores['generalist_matched'],ids),
                'historical_corrections':{
                    'retained_old_score':sum(values[k]>=scores['historical_compact'][k] for k in prior_good),
                    'lost_old_score':sum(values[k]<scores['historical_compact'][k] for k in prior_good),
                    'vs_fresh_compact_on_same_cases':paired(values,scores['compact_matched'],prior_good)},
                'historical_harms':{
                    'improved_over_old_compact':sum(values[k]>scores['historical_compact'][k] for k in prior_bad),
                    'recovered_to_old_generalist_score':sum(values[k]>=scores['historical_generalist'][k] for k in prior_bad),
                    'vs_fresh_compact_on_same_cases':paired(values,scores['compact_matched'],prior_bad)},
                'changed_text_vs_fresh_compact':sum(records[k]['arms'][arm]['text']!=records[k]['arms']['compact_matched']['text'] for k in ids),
                'guidance_applied_cases':sum(records[k]['arms'][arm].get('guidance_applied',False) for k in ids),
                'mean_arm_seconds':statistics.mean(records[k]['arms'][arm]['seconds'] for k in ids),
                'cost_is_historical':arm.startswith('historical_'),
                'zero_cost_rows_are_explicit_shared_control_reuse':arm in ('other_text_only','text_soft','native_spatial_soft'),
                'score_calls':sum(records[k]['arms'][arm].get('base_score_calls',0)+records[k]['arms'][arm].get('conditioned_score_calls',0) for k in ids),
                'image_paired_bootstrap':cluster_bootstrap(delta,[r['image_sha256'] for r in rows])}
        report['datasets'][dataset]=ds
    if scorer_hashes()!=current: raise RuntimeError('scorer changed during evaluation')
    atomic_json(root/'evaluation.json',report)
    summary={**report,'datasets':{k:{a:b for a,b in v.items() if a not in (
        'per_case_scores','historical_correction_ids','historical_harm_ids')} for k,v in report['datasets'].items()}}
    atomic_json(root/'evaluation-summary.json',summary)
    print('FULL EVALUATION COMPLETE',root/'evaluation-summary.json')


if __name__=='__main__':
    main()
