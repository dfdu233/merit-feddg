"""Offline frozen-ANCHOR scoring, only after all fixed pilot cases complete."""
import argparse
import collections
import json
import statistics
import sys
from pathlib import Path

from run_anchored_revision_train import DATA, SCORERS, read, sha

from merit_feddg.agent_evaluate import cluster_bootstrap
from merit_feddg.open_study import atomic_json, fingerprint


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    a = p.parse_args()
    cfg = read(a.run / 'frozen.json')
    identity, ids = fingerprint(cfg), cfg['selected']
    if (a.run / 'evaluation.json').exists():
        raise FileExistsError('Preserve prior scoring')
    files = {p.stem: p for p in (a.run / 'cases').glob('*.json')}
    if set(files) != set(ids):
        raise ValueError('All 24 frozen cases required; no partial pilot scoring')
    if cfg['scorers'] != {p: sha(Path('/home/dbw/ANCHOR') / p) for p in SCORERS}:
        raise ValueError('Frozen primary scoring sources changed')
    if any(sha(p) != h for p, h in cfg['input_hashes'].items()):
        raise ValueError('Frozen input changed')
    records = {k: read(files[k]) for k in ids}
    if any(v['identity'] != identity or set(v['arms']) != set(cfg['arms'])
           for v in records.values()):
        raise ValueError('Case identity or arm coverage mismatch')
    rows = {r['id']: r for r in map(json.loads, Path(cfg['manifest']).read_text().splitlines())}
    refs = read(DATA / 'train/references.json')
    if set(refs) != set(rows):
        raise ValueError('Full TRAIN reference alignment mismatch')
    sys.path.insert(0, '/home/dbw/ANCHOR')
    from anchor.corrected_sgta.evaluate_medheval_answers import PROTOCOL_VERSION, evaluate_rows
    from anchor.medeval.evaluate_mixed_vqa_table import answer_token_recall
    scores, details = {}, {}
    for arm in cfg['arms']:
        values = evaluate_rows([{'qid': k, 'question': rows[k]['question'],
            'answer_type': rows[k]['answer_type'], 'answer': refs[k][0],
            'text': records[k]['arms'][arm]['text']} for k in ids])['details']
        details[arm] = {r['question_id']: r for r in values}
        scores[arm] = {k: float(details[arm][k]['correct'])
            if rows[k]['answer_type'] == 'closed'
            else answer_token_recall(records[k]['arms'][arm]['text'], refs[k][0]) for k in ids}

    def paired(arm, control):
        d = [scores[arm][k] - scores[control][k] for k in ids]
        return {'delta': statistics.mean(d), 'improved': sum(v > 0 for v in d),
                'harmed': sum(v < 0 for v in d), 'image_cluster_ci95':
                cluster_bootstrap(d, [rows[k]['image_sha256'] for k in ids])}

    good = [k for k in ids if scores['compact'][k] > scores['generalist'][k]]
    bad = [k for k in ids if scores['compact'][k] < scores['generalist'][k]]
    report = {'identity': identity, 'pilot_complete': True, 'full_benchmark_complete': False,
        'n': len(ids), 'image_clusters': len({rows[k]['image_sha256'] for k in ids}),
        'metric': 'frozen ANCHOR strict CLOSED + OPEN answer-token-recall; not clinical accuracy',
        'scorer_version': PROTOCOL_VERSION, 'scorer_hashes': cfg['scorers'],
        'reference_sha256': sha(DATA / 'train/references.json'), 'arms': {},
        'answer_types': dict(collections.Counter(rows[k]['answer_type'] for k in ids)),
        'coverage': {'nonempty_revision': sum(bool(v['arms']['revision_evidence']['text'].strip())
            for v in records.values()), 'delivered_cases': sum(bool(v['arms']['compact']['transport']['presented'])
            for v in records.values())},
        'source_gain_cases': len(good), 'source_harm_cases': len(bad),
        'gains_retained': sum(scores['revision_evidence'][k] >= scores['compact'][k] for k in good),
        'harms_recovered': sum(scores['revision_evidence'][k] >= scores['generalist'][k] for k in bad),
        'per_case_scores': scores, 'cost': {}}
    for arm in cfg['arms']:
        report['arms'][arm] = {
            'score': statistics.mean(scores[arm].values()),
            'vs_generalist': paired(arm, 'generalist'), 'vs_compact': paired(arm, 'compact'),
            'vs_revision_no_evidence': paired(arm, 'revision_no_evidence'),
            'changed_from_generalist': sum(records[k]['arms'][arm]['text'] !=
                records[k]['arms']['generalist']['text'] for k in ids),
            'mean_generation_seconds': statistics.mean(records[k]['arms'][arm]['seconds'] for k in ids),
            'calls': sum(records[k]['arms'][arm]['model_calls'] for k in ids),
            'not_eos_finished': sum(not records[k]['arms'][arm]['finished'] for k in ids),
            'parsers': dict(collections.Counter(details[arm][k]['parser'] for k in ids))}
    report['cost'] = {
        'pilot_wall_case_seconds': sum(v['wall_seconds'] for v in records.values()),
        'new_model_calls': sum(x['calls'] for x in report['arms'].values()),
        'expert_calls_new': 0, 'expert_outputs_reused': True,
        'historical_compact_seconds_inherited': sum(v['inherited_compact_seconds'] for v in records.values()),
        'historical_generalist_seconds_inherited': sum(v['inherited_generalist_seconds'] for v in records.values()),
        'model_load_seconds': sum(read(p)['seconds'] for p in a.run.glob('load-*.json')),
        'note': 'Replay/control costs included. Historical compact time includes old acquisition/fusion; '
                'not an independent matched end-to-end deployment benchmark.'}
    atomic_json(a.run / 'evaluation.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'per_case_scores'}, indent=2))


if __name__ == '__main__':
    main()
