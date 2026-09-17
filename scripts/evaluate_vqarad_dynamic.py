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
    p.add_argument('--current-scorer', action='store_true', help='Explicit matched re-score of every saved arm; preserves original scorer pin')
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    root = args.run
    frozen = json.loads((root/'frozen.json').read_text())
    complete = json.loads((root/'complete.json').read_text())
    assert complete['full_dataset_complete'] and complete['identity'] == root.name == fingerprint(frozen)
    current_hashes = scorer_hashes()
    if not args.current_scorer:
        assert current_hashes == frozen['scorer']
    elif args.output is None:
        raise ValueError('current-scorer requires a separate --output')
    destination = args.output or root/'evaluation.json'
    if args.current_scorer and destination.resolve() == (root/'evaluation.json').resolve():
        raise ValueError('preserve historical evaluation destination')
    if destination.exists():
        raise RuntimeError('refuse to overwrite evaluation')
    source = Path(frozen['source_run'])
    data = json.loads((source/'frozen.json').read_text())['datasets']['vqarad']
    rows = data['rows']
    ids = [r['id'] for r in rows]
    refs = json.loads((Path(data['manifest']).parent/'references.json').read_text())
    assert set(refs) == set(ids) and len(ids) == 451
    historic = Path(frozen['history'])
    inherited = {k: json.loads((source/'vqarad'/(k+'.json')).read_text())['arms'] for k in ids}
    sys.path.insert(0, '/home/dbw/ANCHOR')
    from anchor.corrected_sgta.evaluate_medheval_answers import PROTOCOL_VERSION, evaluate_rows
    from anchor.medeval.evaluate_mixed_vqa_table import answer_token_recall, score, VERSION
    dependencies = {str(path): sha(path) for path in (
        Path('/home/dbw/ANCHOR/anchor/corrected_sgta/protocol.py'),
        Path('/home/dbw/ANCHOR/anchor/corrected_sgta/protocol_v2.py'))}
    report = {'identity': root.name, 'n': 451, 'scorer': PROTOCOL_VERSION,
              'table_scorer': VERSION, 'scorer_hashes': current_hashes,
              'scorer_dependencies': dependencies,
              'original_scorer_hashes': frozen['scorer'], 'all_arms_rescored': True,
              'manifest_sha256': sha(data['manifest']),
              'references_sha256': sha(Path(data['manifest']).parent/'references.json'),
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
        for arm in ('historical_generalist', 'historical_compact'):
            outputs[arm] = {k: inherited[k][arm] for k in ids}
        for arm, field in [('text', 'current_incumbent'), ('without_channel', 'without_channel'), ('blend', 'soft'), ('cad', 'cad')]:
            outputs[arm] = {k: history[k][field] if k in active else history[k]['current_incumbent'] for k in ids}
        scores, primary = {}, {}
        for arm, out in outputs.items():
            assert all(x['text'] and x['token_ids'] for x in out.values())
            scoring_rows = [{'qid': r['id'], 'question': r['question'], 'answer_type': r['answer_type'],
                'answer': refs[r['id']][0], 'text': out[r['id']]['text']} for r in rows]
            result = evaluate_rows(scoring_rows)
            primary[arm] = score(scoring_rows)
            details = {x['question_id']: x for x in result['details']}
            scores[arm] = {r['id']: float(details[r['id']]['correct']) if r['answer_type'] == 'closed'
                else answer_token_recall(out[r['id']]['text'], refs[r['id']][0]) for r in rows}
            assert abs(statistics.mean(scores[arm].values())-primary[arm]['primary_unified']['score']) < 1e-12
        good = [k for k in ids if scores['historical_compact'][k] > scores['historical_generalist'][k]]
        bad = [k for k in ids if scores['historical_compact'][k] < scores['historical_generalist'][k]]
        arms = {}
        for arm, values in scores.items():
            arms[arm] = {'mixed_score': statistics.mean(values.values()),
                'main_pipeline_metrics': primary[arm],
                'closed_accuracy': statistics.mean(values[r['id']] for r in rows if r['answer_type'] == 'closed'),
                'open_token_recall': statistics.mean(values[r['id']] for r in rows if r['answer_type'] == 'open'),
                'vs_text': paired(values, scores['text'], ids),
                'vs_cad': paired(values, scores['cad'], ids),
                'old_corrections_vs_text': paired(values, scores['text'], good),
                'old_harms_vs_text': paired(values, scores['text'], bad),
                'image_paired_bootstrap': cluster_bootstrap([values[k]-scores['text'][k] for k in ids], [r['image_sha256'] for r in rows]),
                'image_paired_bootstrap_vs_cad': cluster_bootstrap([values[k]-scores['cad'][k] for k in ids], [r['image_sha256'] for r in rows])}
        report['channels'][channel] = {'n': len(ids), 'real_candidate_cases': len(active),
            'candidate_coverage': len(active)/len(ids), 'arms': arms, 'per_case_scores': scores,
            'text_changed': sum(outputs['acd'][k]['text'] != outputs['text'][k]['text'] for k in ids),
            'new_wall_seconds': sum(r['wall_seconds'] for r in records.values()),
            'acd_seconds': sum(records[k]['acd']['seconds'] for k in active),
            'score_calls': sum(records[k]['acd']['score_calls'] for k in active),
            'inherited_control_wall_seconds': sum(h['wall_seconds'] for h in history.values()),
            'effective_weights': [s['effective_weight'] for k in active for s in records[k]['acd']['steps']],
            'source_modes': frozen['source_modes']}
    assert scorer_hashes() == current_hashes
    assert all(sha(path) == digest for path, digest in dependencies.items())
    atomic_json(destination, report)
    print('EVALUATION COMPLETE', destination, flush=True)


if __name__ == '__main__':
    main()
