"""Offline complete TRAIN probe scoring, using current ANCHOR; no raw text export."""
import argparse
import collections
import hashlib
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, '/home/dbw/ANCHOR')
from anchor.corrected_sgta.evaluate_medheval_answers import PROTOCOL_VERSION, evaluate_rows
from anchor.medeval.evaluate_mixed_vqa_table import answer_token_recall


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    read = lambda p: json.loads(p.read_text())
    protocol = read(args.run / 'protocol.json')
    complete = read(args.run / 'complete.json')
    rows = {r['id']: r for r in protocol['rows']}
    files = {p.stem: p for p in (args.run / 'cases').glob('*.json')}
    if complete['identity'] != protocol['identity'] or set(files) != set(rows):
        raise ValueError('Exact completed identity and ID set required')
    cases = {k: read(v) for k, v in files.items()}
    if any(not v['complete'] or v['identity'] != protocol['identity'] for v in cases.values()):
        raise ValueError('Incomplete cases')
    ref_path = Path('/home/dbw/merit-feddg/runs/vqarad-official-protocol-v3/data/train/references.json')
    refs = read(ref_path)
    scorer_paths = [Path('/home/dbw/ANCHOR/anchor/corrected_sgta') / 'evaluate_medheval_answers.py',
                    Path('/home/dbw/ANCHOR/anchor/medeval/evaluate_mixed_vqa_table.py')]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in scorer_paths}
    scores = {}
    for arm in ('generalist', 'compact', 'relevance', 'scope'):
        source = [{'qid': k, 'question': r['question'], 'answer_type': r['answer_type'],
                   'answer': refs[k][0], 'text': cases[k]['arms'][arm]['text']} for k, r in rows.items()]
        details = {r['question_id']: r for r in evaluate_rows(source)['details']}
        scores[arm] = {k: float(details[k]['correct']) if r['answer_type'] == 'closed' else
                      answer_token_recall(cases[k]['arms'][arm]['text'], refs[k][0]) for k, r in rows.items()}
    result = {'identity': protocol['identity'], 'n': len(rows), 'pilot_only': True,
        'metric': 'Current pinned ANCHOR CLOSED parser + OPEN token recall, not clinical accuracy',
        'scorer_version': PROTOCOL_VERSION, 'scorer_hashes': hashes,
        'reference_hash': hashlib.sha256(ref_path.read_bytes()).hexdigest(),
        'arms': {}, 'gates': {}, 'per_case_scores': scores,
        'cost_limit': 'Successful v5 calls only; earlier failed-run wall/call costs not fully captured, '
                      'so these are not total project costs or isolated throughput.',
        'privacy': 'No patient questions, answers, images, native packets or token IDs.'}
    for arm, values in scores.items():
        result['arms'][arm] = {'score': statistics.mean(values.values()),
            'improved_vs_compact': sum(v > scores['compact'][k] for k, v in values.items()),
            'harmed_vs_compact': sum(v < scores['compact'][k] for k, v in values.items()),
            'empty': sum(not r['arms'][arm]['text'] for r in cases.values()),
            'new_candidate_calls': sum(r['arms'][arm].get('new_answer_calls', 0) for r in cases.values()),
            'reuse': dict(collections.Counter(r['arms'][arm].get('reuse', 'generated') for r in cases.values()))}
    for policy in ('relevance', 'scope'):
        judgments = [v for r in cases.values() for v in r['gates'][policy]]
        result['gates'][policy] = {'labels': dict(collections.Counter(v['label'] for v in judgments)),
            'real_calls': len(judgments), 'seconds': sum(v['seconds'] for v in judgments)}
    result['successful_run_cost'] = {
        'generalist_seconds': sum(r['arms']['generalist']['seconds'] for r in cases.values()),
        'compact_outer_seconds': sum(r['compact_raw']['wall_seconds'] for r in cases.values()),
        'new_candidate_seconds': sum(r['arms'][arm]['seconds'] for r in cases.values()
                                    for arm in ('relevance', 'scope')
                                    if r['arms'][arm]['new_answer_calls']),
        'actor_model_load_seconds': read(args.run / 'model-load.json')['seconds'],
        'parity_calls': sum('compact_parity' in r for r in cases.values()),
        'parity_passed': sum(r.get('compact_parity', False) for r in cases.values()),
        'parity_seconds': None,
        'native_expert_prefetch_cache_records': len(list((args.run / 'expert-cache').rglob('*.json'))),
        'native_expert_prefetch_seconds': None,
        'missing_costs': 'Route/model-load/expert-prefetch/parity phase wall times not fully instrumented; '
                         'not zero and not included in a total speedup claim.',
    }
    if any(hashlib.sha256(p.read_bytes()).hexdigest() != hashes[str(p)] for p in scorer_paths):
        raise ValueError('Scorer changed during evaluation')
    if args.output.exists():
        raise FileExistsError('Never overwrite prior results')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
