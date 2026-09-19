"""Offline frozen-ANCHOR scoring; no raw patient text in exported aggregate."""
import collections
import hashlib
import json
import statistics
import sys
from pathlib import Path

from observation_blind_probe import BASE, OUT, ROOT, read, write

sys.path.insert(0,'/home/dbw/ANCHOR')
from anchor.corrected_sgta.evaluate_medheval_answers import evaluate_rows
from anchor.medeval.evaluate_mixed_vqa_table import answer_token_recall
from merit_feddg.agent_evaluate import cluster_bootstrap


def main():
    config = read(OUT/'protocol.json')
    assert read(OUT/'complete.json')['identity'] == config['identity']
    rows = {r['id']:r for r in config['rows']}
    paths = {p.stem:p for p in (OUT/'cases').glob('*.json')}
    assert set(paths) == set(rows)
    cases = {k:read(p) for k,p in paths.items()}
    assert all(c['complete'] and c['identity']==config['identity'] for c in cases.values())
    original = {k:read(BASE/'cases'/(k+'.json')) for k in rows}
    refs = read(Path('/home/dbw/merit-feddg-huatuo-critic/runs/inputs-slake128-confirm/references.json'))
    scorer_paths = {'evaluate_medheval_answers.py':'ad0fb9abfebf1a0185709628835b33a052492fbe5e4828d097141282a59a21d6',
                    'evaluate_mixed_vqa_table.py':'7d5b7675fc0aa6455b1d60fc93dcfbb0fd04fa07e879259e483e38b7f24d99f4'}
    for name,sha in scorer_paths.items():
        folder = 'corrected_sgta' if name.startswith('evaluate_medheval') else 'medeval'
        assert hashlib.sha256((Path('/home/dbw/ANCHOR/anchor')/folder/name).read_bytes()).hexdigest()==sha
    scores = {}
    for arm in ['generalist','compact']+config['arms']:
        texts = {k:original[k]['arms'][arm]['text'] if arm in ('generalist','compact')
                 else cases[k]['conditions'][arm]['text'] for k in rows}
        source = [{'qid':k,'question':r['question'],'answer_type':r['answer_type'],
                   'answer':refs[k][0],'text':texts[k]} for k,r in rows.items()]
        detail = {r['question_id']:r for r in evaluate_rows(source)['details']}
        scores[arm] = {k:float(detail[k]['correct']) if rows[k]['answer_type']=='closed'
                      else answer_token_recall(texts[k],refs[k][0]) for k in rows}
    result = {'identity':config['identity'],'n':len(rows),'pilot_only':True,
        'metric':'Frozen ANCHOR CLOSED parser + OPEN token recall; not clinical accuracy',
        'scorer_hashes':scorer_paths,'arms':{},'calls':{},'coverage':{},
        'limitations':['Previously inspected TRAIN images; not holdout','English exact-name anatomy selection',
            'Full-answer likelihood; claim-level selector not implemented','No classification/text expert efficacy experiment',
            'Original candidate/routing/segmentation costs inherited, not zero']}
    for arm,s in scores.items():
        result['arms'][arm] = {'score':statistics.mean(s.values()),
            'gains_vs_baseline':sum(s[k]>scores['generalist'][k] for k in rows),
            'harms_vs_baseline':sum(s[k]<scores['generalist'][k] for k in rows),
            'ci95_delta_vs_baseline':cluster_bootstrap([s[k]-scores['generalist'][k] for k in rows],
                                                      [r['pixel_sha256'] for r in rows.values()])}
        if arm in config['arms']:
            result['arms'][arm]['selection'] = dict(collections.Counter(c['conditions'][arm]['selected'] for c in cases.values()))
            result['arms'][arm]['sum_mean_selection_disagreement'] = sum(
                ('compact' if c['conditions'][arm]['scores']['compact']['sum'] > c['conditions'][arm]['scores']['generalist']['sum'] else 'generalist')
                != c['conditions'][arm]['selected'] for c in cases.values() if c['conditions'][arm]['scores'])
            result['arms'][arm]['unavailable'] = sum('unavailable' in c['conditions'][arm] for c in cases.values())
        eligible = [k for k in rows if cases[k]['observation']['available']]
        result['arms'][arm]['eligible_score'] = statistics.mean(s[k] for k in eligible) if eligible else None
        result['arms'][arm]['eligible_gains_vs_baseline'] = sum(s[k] > scores['generalist'][k] for k in eligible)
        result['arms'][arm]['eligible_harms_vs_baseline'] = sum(s[k] < scores['generalist'][k] for k in eligible)
    calls = [v for c in cases.values() for v in c['calls']]
    result['calls'] = {'count':len(calls),'seconds':sum(v['seconds'] for v in calls),
        'per_arm_seconds':{a:sum(v['seconds'] for v in calls if v['stage']==a) for a in config['arms']},
        'load_seconds':sum(read(p)['seconds'] for p in OUT.glob('load-*.json')),
        'peak_allocated_bytes':max(c['peak_memory_bytes'] for c in cases.values())}
    result['calls']['inherited_candidate_seconds'] = {
        arm:sum(original[k]['arms'][arm]['seconds'] for k in rows) for arm in ('generalist','compact')}
    result['calls']['inherited_cost_note'] = 'Historical saved runner seconds, not rerun this round; compact and compact_raw describe the same call and are not double-counted. Any unrecorded historical costs remain unknown.'
    result['coverage'] = {'real_crops':sum(c['observation']['available'] for c in cases.values()),'normal_inherited_parity':sum(c['normal_inherited_view_parity'] for c in cases.values()),
        'native_input_parity':sum(c['native_single_image_parity'] for c in cases.values()),
        'different_candidate_texts':sum(original[k]['arms']['generalist']['text'] != original[k]['arms']['compact']['text'] for k in rows),
        'new_answer_generation':0,'raw_candidate_reuse':2*len(cases),
        'crop_vs_fullview_choice_changes':sum(c['conditions']['native_crop']['selected'] != c['conditions']['full_view_control']['selected'] for c in cases.values())}
    write(OUT/'evaluation.json',result)
    write(ROOT/'reports'/(OUT.name+'.json'),result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
