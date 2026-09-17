"""Complete-only historical ANCHOR scoring; references never enter inference."""
import argparse
import json
from pathlib import Path
import statistics
import sys

from evaluate_soft_guidance_full import paired, scorer_hashes
from run_soft_guidance_full import sha
from merit_feddg.agent_evaluate import cluster_bootstrap
from merit_feddg.open_study import atomic_json, fingerprint


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    root = p.parse_args().run
    frozen = json.loads((root/'frozen.json').read_text())
    complete = json.loads((root/'complete.json').read_text())
    assert complete['full_dataset_complete'] and complete['identity'] == root.name == fingerprint(frozen)
    assert scorer_hashes() == frozen['scorer']
    if (root/'evaluation.json').exists():
        raise RuntimeError('refuse to overwrite evaluation')
    source = Path(frozen['source_run'])
    data = json.loads((source/'frozen.json').read_text())['datasets']['vqarad']
    rows = data['rows']
    ids = [r['id'] for r in rows]
    refs = json.loads((Path(data['manifest']).parent/'references.json').read_text())
    assert set(refs) == set(ids) and len(ids) == 451
    historic = Path(frozen['history'])
    inherited_scores = json.loads((source/'evaluation.json').read_text())['datasets']['vqarad']['per_case_scores']
    good = [k for k in ids if inherited_scores['historical_compact'][k] > inherited_scores['historical_generalist'][k]]
    bad = [k for k in ids if inherited_scores['historical_compact'][k] < inherited_scores['historical_generalist'][k]]
    sys.path.insert(0, '/home/dbw/ANCHOR')
    from anchor.corrected_sgta.evaluate_medheval_answers import PROTOCOL_VERSION, evaluate_rows
    from anchor.medeval.evaluate_mixed_vqa_table import answer_token_recall
    report = {'identity': root.name, 'n': 451, 'scorer': PROTOCOL_VERSION,
              'evaluator_sha256': sha(__file__), 'channels': {}}
    for channel in frozen['channels']:
        assert {p.stem for p in (root/channel).glob('*.json')} == set(ids)
        records = {k: json.loads((root/channel/(k+'.json')).read_text()) for k in ids}
        history = {k: json.loads((historic/'vqarad'/channel/(k+'.json')).read_text()) for k in ids}
        active = [k for k in ids if records[k]['status'] == 'real_candidate']
        for k in ids:
            assert records[k]['id'] == k and records[k]['identity'] == root.name
            assert sha(historic/'vqarad'/channel/(k+'.json')) == frozen['history_records'][channel][k]
        outputs = {'acd': {k: records[k]['acd'] for k in ids}}
        for arm, field in [('text', 'current_incumbent'), ('without_channel', 'without_channel'), ('blend', 'soft'), ('cad', 'cad')]:
            outputs[arm] = {k: history[k][field] if k in active else history[k]['current_incumbent'] for k in ids}
        scores = {}
        for arm, out in outputs.items():
            assert all(x['text'] and x['token_ids'] for x in out.values())
            result = evaluate_rows([{'qid': r['id'], 'question': r['question'], 'answer_type': r['answer_type'],
                'answer': refs[r['id']][0], 'text': out[r['id']]['text']} for r in rows])
            details = {x['question_id']: x for x in result['details']}
            scores[arm] = {r['id']: float(details[r['id']]['correct']) if r['answer_type'] == 'closed'
                else answer_token_recall(out[r['id']]['text'], refs[r['id']][0]) for r in rows}
        arms = {}
        for arm, values in scores.items():
            arms[arm] = {'mixed_score': statistics.mean(values.values()),
                'closed_accuracy': statistics.mean(values[r['id']] for r in rows if r['answer_type'] == 'closed'),
                'open_token_recall': statistics.mean(values[r['id']] for r in rows if r['answer_type'] == 'open'),
                'vs_text': paired(values, scores['text'], ids),
                'vs_cad': paired(values, scores['cad'], ids),
                'old_corrections_vs_text': paired(values, scores['text'], good),
                'old_harms_vs_text': paired(values, scores['text'], bad),
                'image_paired_bootstrap': cluster_bootstrap([values[k]-scores['text'][k] for k in ids], [r['image_sha256'] for r in rows])}
        report['channels'][channel] = {'n': len(ids), 'real_candidate_cases': len(active),
            'candidate_coverage': len(active)/len(ids), 'arms': arms, 'per_case_scores': scores,
            'text_changed': sum(outputs['acd'][k]['text'] != outputs['text'][k]['text'] for k in ids),
            'new_wall_seconds': sum(r['wall_seconds'] for r in records.values()),
            'acd_seconds': sum(records[k]['acd']['seconds'] for k in active),
            'score_calls': sum(records[k]['acd']['score_calls'] for k in active),
            'inherited_control_wall_seconds': sum(h['wall_seconds'] for h in history.values()),
            'effective_weights': [s['effective_weight'] for k in active for s in records[k]['acd']['steps']],
            'source_modes': frozen['source_modes']}
    assert scorer_hashes() == frozen['scorer']
    atomic_json(root/'evaluation.json', report)
    print('EVALUATION COMPLETE', root/'evaluation.json', flush=True)


if __name__ == '__main__':
    main()
