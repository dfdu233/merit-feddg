"""Offline paired utility diagnosis. Never imported by generation or routing.

Scores must come from a declared frozen evaluator, not a Yes/No heuristic here.
Optional natural-domain metadata reports cross-domain performance without
creating splits, fitting a gate, or using target labels during inference.
"""
import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from .capabilities import EvidenceItem
from .compact_evidence import compact_records


def audit_revisions(baseline, candidate, verified, *, scores=None, groups=None):
    ids = set(baseline)
    if not ids or set(candidate) != ids or set(verified) != ids:
        raise ValueError('complete identical case IDs required; no partial intersection')
    if scores is not None and set(scores) != ids:
        raise ValueError('scores must cover the complete run')
    if groups is not None and set(groups) != ids:
        raise ValueError('domain metadata must cover the complete run')
    events, domains, cases = Counter(), defaultdict(list), []
    for key in baseline:
        base, cand, final = baseline[key], candidate[key], verified[key]
        gate = final['answer_arbitration']
        if final.get('candidate_answer', {}).get('text') != cand['text']:
            raise ValueError(f'candidate mismatch for {key}')
        decision = gate['decision']
        if decision not in {'accept', 'reject', 'abstain', 'unchanged'}:
            raise ValueError('unknown gate decision')
        resolution = gate.get('resolution', 'candidate' if decision in {'accept', 'unchanged'} else 'baseline')
        expected = cand if resolution == 'candidate' else base
        if final['text'] != expected['text'] or final['token_ids'] != expected['token_ids']:
            raise ValueError(f'final output does not match recorded resolution for {key}')
        row = {'id': key, 'domain': str(groups[key]) if groups else 'unspecified',
               'baseline': base['text'], 'candidate': cand['text'], 'final': final['text'],
               'decision': decision, 'resolution': resolution, 'reason': gate['reason'],
               'margins': gate.get('margins'), 'score_calls': gate.get('score_calls', 0),
               'expert_ids': [e['expert_id'] for e in cand.get('evidence', [])],
               'evidence': compact_records([EvidenceItem(**e) for e in cand.get('evidence', [])]),
               'transport': cand.get('evidence_transport', {})}
        events[f'reason:{gate["reason"]}'] += 1
        if scores is not None:
            a, b = scores[key]['baseline'], scores[key]['candidate']
            if any(type(x) not in {int, float} or not math.isfinite(x) or not 0 <= x <= 1 for x in (a, b)):
                raise ValueError('frozen evaluator scores must be finite values in [0,1]')
            delta = b-a
            effect = 'beneficial' if delta > 0 else 'harmful' if delta < 0 else 'tied'
            events[f'{effect}:{decision}'] += 1
            final_score = b if resolution == 'candidate' else a
            row.update(effect=effect, candidate_gain=delta, final_gain=final_score-a,
                       gate_gain=final_score-b)
            domains[row['domain']].append(row)
        cases.append(row)
    per_domain = {name: {'n': len(rows),
                        'candidate_gain': sum(r['candidate_gain'] for r in rows)/len(rows),
                        'final_gain': sum(r['final_gain'] for r in rows)/len(rows),
                        'gate_gain': sum(r['gate_gain'] for r in rows)/len(rows)}
                  for name, rows in domains.items()}
    return {'schema': 'revision-utility-audit-v1', 'n': len(ids),
            'offline_only': True, 'probability_calibrated': False,
            'events': dict(events), 'domains': per_domain, 'cases': cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline', 'candidate', 'verified', 'output'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--scores', help='JSON with evaluator_id and cases: {id: {baseline, candidate}}')
    parser.add_argument('--groups', help='JSON {id: existing hospital/domain}; no inferred hospitals')
    args = parser.parse_args()
    read = lambda p: json.loads(Path(p).read_text())
    scored = read(args.scores) if args.scores else None
    if scored is not None and not scored.get('evaluator_id'):
        raise ValueError('declare the frozen evaluator identity')
    result = audit_revisions(read(args.baseline), read(args.candidate), read(args.verified),
        scores=scored['cases'] if scored is not None else None,
        groups=read(args.groups) if args.groups else None)
    result['evaluator_id'] = scored['evaluator_id'] if scored else None
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
