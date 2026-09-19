"""Export aggregates only: never patient text, image paths or raw evidence."""
import argparse
import collections
import json
from pathlib import Path

from merit_feddg.open_study import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--evaluation', default='evaluation.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.run / args.evaluation).read_text())
    scores = report.pop('per_case_scores')
    report['metric'] = ('Pinned current ANCHOR CLOSED parser + OPEN answer-token-recall; '
                        'CLOSED includes explanatory parsing, not leading-token-only strict EM; '
                        'not clinical accuracy')
    cases = [json.loads(p.read_text()) for p in (args.run / 'cases').glob('*.json')]
    if len(cases) != report['n'] or not all(r['complete'] for r in cases):
        raise ValueError('Require complete exact audited run')
    frozen = json.loads((args.run / 'frozen.json').read_text())
    source = frozen.get('source', frozen.get('prior'))
    historical = json.loads((Path(source['base']) / 'compact_rows.json').read_text())
    tools = [v for row in cases for v in historical[row['id']]['trace'] if v.get('event') == 'tool']
    report['inherited_tool_audit'] = {
        'historical_expert_calls': sum(historical[r['id']]['expert_calls'] for r in cases),
        'historical_controller_calls': sum(historical[r['id']]['controller_calls'] for r in cases),
        'tool_trace_seconds': sum(v.get('seconds', 0) for v in tools),
        'tool_reasons': dict(collections.Counter(v.get('reason') for v in tools)),
        'note': 'Historical trace costs, not newly executed. May include native cache reuse; do not sum blindly with outer timings.',
    }
    report['delivery_audit'] = {
        'cases_with_packets': (sum(r['delivered_packets'] > 0 for r in cases)
                               if all('delivered_packets' in r for r in cases) else None),
        'packet_distribution': (dict(collections.Counter(r['delivered_packets'] for r in cases))
                                if all('delivered_packets' in r for r in cases) else None),
        'parity_checks': sum(len(r.get('parity', {})) for r in cases),
        'parity_passed': sum(v['passed'] for r in cases for v in r.get('parity', {}).values()),
        'genuinely_new_candidate_cases': {
            arm: sum(r['arms'][arm]['new_calls'] > 0 for r in cases) for arm in report['arms']
        },
    }
    report['privacy'] = 'Aggregates only. No questions, answers, patient text, images or raw evidence.'
    if all('gates' in r for r in cases):
        groups = collections.defaultdict(list)
        for row in cases:
            experts = sorted({v['expert_id'] for v in row['gates']['scope']})
            groups['+'.join(experts) or 'no_delivered_packets'].append(row['id'])
        report['expert_groups_observational_not_causal'] = {
            group: {
                'n': len(ids),
                'compact_improves_generalist': sum(scores['compact'][k] > scores['generalist'][k] for k in ids),
                'compact_harms_generalist': sum(scores['compact'][k] < scores['generalist'][k] for k in ids),
            } for group, ids in groups.items()
        }
    if args.output.exists():
        raise FileExistsError('Preserve prior reports')
    atomic_json(args.output, report)


if __name__ == '__main__':
    main()
