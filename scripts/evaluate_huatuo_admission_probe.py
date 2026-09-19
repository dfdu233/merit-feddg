"""Offline complete TRAIN probe scoring, using current ANCHOR; no raw text export."""
import argparse
import collections
import hashlib
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, '/home/dbw/ANCHOR')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anchor.corrected_sgta.evaluate_medheval_answers import PROTOCOL_VERSION, evaluate_rows
from anchor.medeval.evaluate_mixed_vqa_table import answer_token_recall

from merit_feddg.agent_evaluate import cluster_bootstrap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--candidate-run', type=Path)
    parser.add_argument('--references', type=Path,
        default=Path('/home/dbw/merit-feddg/runs/vqarad-official-protocol-v3/data/train/references.json'))
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
    arms = ['generalist', 'compact', 'relevance', 'scope']
    candidate_cost = None
    if args.candidate_run:
        candidate_protocol = read(args.candidate_run / 'protocol.json')
        if candidate_protocol.get('image_control', 'original') != 'original':
            raise ValueError('Mismatched-image diagnostic must not be scored as patient predictions')
        candidate_complete = read(args.candidate_run / 'complete.json')
        if candidate_protocol['base_identity'] != protocol['identity'] or candidate_complete['identity'] != candidate_protocol['identity']:
            raise ValueError('Candidate/control identity mismatch')
        candidate_files = {p.stem:p for p in (args.candidate_run/'cases').glob('*.json')}
        if set(candidate_files) != set(rows):
            raise ValueError('Candidate ID set differs')
        calls = []
        for key, path in candidate_files.items():
            candidate = read(path)
            if not candidate['complete'] or candidate['identity'] != candidate_protocol['identity']:
                raise ValueError('Incomplete candidate')
            for arm, value in candidate['arms'].items():
                if arm in cases[key]['arms']:
                    raise ValueError('Candidate must not overwrite old arms')
                cases[key]['arms'][arm] = dict(value, new_answer_calls=value.get(
                    'new_answer_calls', int(not value.get('reused_generalist', False))))
            calls.extend(candidate['calls'])
        candidate_arms = candidate_protocol.get('arms', sorted(read(next(iter(candidate_files.values())))['arms']))
        if any(set(read(p)['arms']) != set(candidate_arms) for p in candidate_files.values()):
            raise ValueError('Candidate arm set differs between cases')
        arms = ['generalist', 'compact'] + candidate_arms
        candidate_cost = {'identity':candidate_protocol['identity'], 'calls':len(calls),
                          'seconds':sum(v['seconds'] for v in calls),
                          'stages':dict(collections.Counter(v.get('stage', 'pairwise_judge') for v in calls))}
        if 'decision_channel' in candidate_protocol:
            candidate_cost['selection_audit'] = {
                'channel': candidate_protocol['decision_channel'],
                'reasons': dict(collections.Counter(read(p)['reason'] for p in candidate_files.values())),
                'selected': dict(collections.Counter(read(p)['selected'] for p in candidate_files.values())),
                'labels': dict(collections.Counter(v for p in candidate_files.values() for v in read(p)['verdicts'])),
                'medical_confidence_calibrated': False,
            }
    ref_path = args.references
    refs = read(ref_path)
    scorer_paths = [Path('/home/dbw/ANCHOR/anchor/corrected_sgta') / 'evaluate_medheval_answers.py',
                    Path('/home/dbw/ANCHOR/anchor/medeval/evaluate_mixed_vqa_table.py')]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in scorer_paths}
    scores = {}
    for arm in arms:
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
        'cost_limit': 'Recorded completed-run calls only; earlier failed-run wall/call costs not fully captured, '
                      'so these are not total project costs or isolated throughput.',
        'privacy': 'No patient questions, answers, images, native packets or token IDs.'}
    result['candidate_cost'] = candidate_cost
    for arm, values in scores.items():
        result['arms'][arm] = {'score': statistics.mean(values.values()),
            'improved_vs_compact': sum(v > scores['compact'][k] for k, v in values.items()),
            'harmed_vs_compact': sum(v < scores['compact'][k] for k, v in values.items()),
            'improved_vs_generalist': sum(v > scores['generalist'][k] for k, v in values.items()),
            'harmed_vs_generalist': sum(v < scores['generalist'][k] for k, v in values.items()),
            'image_ci95_vs_generalist': cluster_bootstrap(
                [v-scores['generalist'][k] for k,v in values.items()],
                [rows[k]['image_sha256'] for k in values]),
            'image_ci95_vs_compact': cluster_bootstrap(
                [v-scores['compact'][k] for k,v in values.items()],
                [rows[k]['image_sha256'] for k in values]),
            'empty': sum(not r['arms'][arm]['text'] for r in cases.values()),
            'new_candidate_calls': sum(r['arms'][arm].get('new_answer_calls', 0) for r in cases.values()),
            'reuse': dict(collections.Counter(r['arms'][arm].get('reuse', 'generated') for r in cases.values()))}
        if 'anchored_blind_editor' in scores:
            blind = scores['anchored_blind_editor']
            result['arms'][arm]['improved_vs_blind_editor'] = sum(v > blind[k] for k, v in values.items())
            result['arms'][arm]['harmed_vs_blind_editor'] = sum(v < blind[k] for k, v in values.items())
            result['arms'][arm]['image_ci95_vs_blind_editor'] = cluster_bootstrap(
                [v-blind[k] for k, v in values.items()],
                [rows[k].get('pixel_sha256', rows[k]['image_sha256']) for k in values])
    for policy in ('relevance', 'scope'):
        judgments = [v for r in cases.values() for v in r.get('gates', {}).get(policy, [])]
        result['gates'][policy] = {'labels': dict(collections.Counter(v['label'] for v in judgments)),
            'real_calls': len(judgments), 'seconds': sum(v['seconds'] for v in judgments)}
    result['successful_run_cost'] = {
        'generalist_seconds': sum(r['arms']['generalist']['seconds'] for r in cases.values()),
        'compact_outer_seconds': sum(r['compact_raw']['wall_seconds'] for r in cases.values()),
        'new_candidate_seconds': sum(r['arms'][arm]['seconds'] for r in cases.values()
                                    for arm in ('relevance', 'scope')
                                    if arm in r['arms'] and r['arms'][arm]['new_answer_calls']),
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
