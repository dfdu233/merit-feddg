"""Complete-only ANCHOR evaluation of independent classification/text channels."""
import argparse
import json
import statistics
import sys
from pathlib import Path

from evaluate_soft_guidance_full import paired, scorer_hashes
from run_class_text_guidance import ARMS, CHANNELS

from merit_feddg.agent_evaluate import cluster_bootstrap
from merit_feddg.open_study import atomic_json, fingerprint


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True, type=Path)
    args = p.parse_args()
    root = args.run
    frozen = json.loads((root/'frozen.json').read_text())
    complete = json.loads((root/'complete.json').read_text())
    if not complete['full_dataset_complete'] or complete['identity'] != fingerprint(frozen):
        raise RuntimeError('complete matching full run required')
    if scorer_hashes() != frozen['scorer']:
        raise RuntimeError('frozen ANCHOR scorer changed')
    if (root/'evaluation.json').exists():
        raise RuntimeError('refuse overwrite')
    source = Path(frozen['source_run'])
    data = json.loads((source/'frozen.json').read_text())['datasets']
    history = json.loads((source/'evaluation.json').read_text())['datasets']
    sys.path.insert(0, '/home/dbw/ANCHOR')
    from anchor.corrected_sgta.evaluate_medheval_answers import PROTOCOL_VERSION, evaluate_rows
    from anchor.medeval.evaluate_mixed_vqa_table import answer_token_recall
    report = {'identity': root.name, 'full_dataset_complete': True, 'scorer': PROTOCOL_VERSION,
              'native_class_to_token_logits': False, 'datasets': {}}
    for dataset, ds in data.items():
        rows = ds['rows']
        ids = [r['id'] for r in rows]
        refs = json.loads((Path(ds['manifest']).parent/'references.json').read_text())
        if set(refs) != set(ids):
            raise RuntimeError('reference alignment failure')
        report['datasets'][dataset] = {}
        for channel in CHANNELS:
            files = {p.stem: p for p in (root/dataset/channel).glob('*.json')}
            if set(files) != set(ids):
                raise RuntimeError('incomplete channel')
            records = {k: json.loads(files[k].read_text()) for k in ids}
            if any(r['identity'] != root.name or r['id'] != k for k, r in records.items()):
                raise RuntimeError('record identity mismatch')
            active = [k for k in ids if records[k]['status'] == 'real_candidate']
            outputs = {}
            for arm in ARMS:
                field = {'text': 'current_incumbent', 'blend': 'soft'}.get(arm, arm)
                outputs[arm] = {k: records[k][field] if k in active else records[k]['current_incumbent'] for k in ids}
                if any(not r['text'] or not r['token_ids'] for r in outputs[arm].values()):
                    raise RuntimeError('empty arm')
            scores = {}
            for arm in ARMS:
                evaluated = evaluate_rows([{'qid': r['id'], 'question': r['question'],
                    'answer_type': r['answer_type'], 'answer': refs[r['id']][0],
                    'text': outputs[arm][r['id']]['text']} for r in rows])
                details = {r['question_id']: r for r in evaluated['details']}
                scores[arm] = {r['id']: float(details[r['id']]['correct']) if r['answer_type'] == 'closed'
                    else answer_token_recall(outputs[arm][r['id']]['text'], refs[r['id']][0]) for r in rows}
            hs = history[dataset]['per_case_scores']
            good = [k for k in ids if hs['historical_compact'][k] > hs['historical_generalist'][k]]
            bad = [k for k in ids if hs['historical_compact'][k] < hs['historical_generalist'][k]]
            result = {'n': len(ids), 'real_candidate_cases': len(active), 'unavailable_cases': len(ids)-len(active),
                'new_expert_model_calls': 0, 'all_arm_wall_seconds': sum(r['wall_seconds'] for r in records.values()),
                'arms': {}, 'per_case_scores': scores}
            for arm in ARMS:
                values = scores[arm]
                result['arms'][arm] = {
                    'mixed_score': statistics.mean(values.values()),
                    'closed_accuracy': statistics.mean(values[r['id']] for r in rows if r['answer_type'] == 'closed'),
                    'open_token_recall': statistics.mean(values[r['id']] for r in rows if r['answer_type'] == 'open'),
                    'vs_text': paired(values, scores['text'], ids),
                    'vs_without_channel': paired(values, scores['without_channel'], ids),
                    'active_vs_text': paired(values, scores['text'], active),
                    'old_corrections_vs_current_text': paired(values, scores['text'], good),
                    'old_harms_vs_current_text': paired(values, scores['text'], bad),
                    'mean_active_arm_seconds': statistics.mean(outputs[arm][k]['seconds'] for k in active) if active else None,
                    'image_paired_bootstrap': cluster_bootstrap([values[k]-scores['text'][k] for k in ids],
                                                               [r['image_sha256'] for r in rows])}
            report['datasets'][dataset][channel] = result
    if scorer_hashes() != frozen['scorer']:
        raise RuntimeError('scorer changed during evaluation')
    atomic_json(root/'evaluation.json', report)
    for ds in report['datasets'].values():
        for result in ds.values():
            result.pop('per_case_scores')
    atomic_json(root/'evaluation-summary.json', report)
    print('FULL CLASS/TEXT EVALUATION COMPLETE', root/'evaluation-summary.json', flush=True)


if __name__ == '__main__':
    main()
