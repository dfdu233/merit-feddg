"""Offline paired diagnostics; reference labels NEVER enter agent_run.

Diagnostic closed exact-match/open token-recall is NOT clinical accuracy or the
paper's official metric. Optionally invoke the installed frozen ANCHOR evaluator
and keep its complete report separately, without replacing/changing its scorer.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from .agent_protocol import atomic_json, file_hash, read_inputs


def words(value):
    return re.findall(r'\w+', value.casefold())


def diagnostic_score(text, reference, answer_type):
    pred, ref = words(text), words(reference)
    if answer_type == 'closed':
        if ref in (['yes'], ['no']):
            match = re.match(r'^\s*(?:answer\s*:\s*)?(yes|no)\b', text, re.I)
            return float(bool(match and match.group(1).casefold() == ref[0]))
        # Do not coerce every non-yes reference into no (SLAKE is not binary).
        return float(pred == ref)
    if answer_type != 'open':
        raise ValueError('diagnostic scorer supports open/closed VQA, not report generation')
    overlap = sum((Counter(pred) & Counter(ref)).values())
    return overlap / len(ref) if ref else float(not pred)


def cluster_bootstrap(deltas, groups, *, repetitions=2000, seed=0):
    if not deltas or len(deltas) != len(groups) or repetitions < 1:
        raise ValueError('aligned nonempty scores/groups and positive repetitions required')
    unique = sorted(set(groups))
    sums = np.array([sum(d for d, g in zip(deltas, groups) if g == k) for k in unique])
    counts = np.array([sum(g == k for g in groups) for k in unique])
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(repetitions):
        sample = rng.integers(len(unique), size=len(unique))
        means.append(float(sums[sample].sum() / counts[sample].sum()))
    return {'low': float(np.quantile(means, .025)), 'high': float(np.quantile(means, .975)),
            'clusters': len(unique), 'unit': 'image', 'sample_weighted': True,
            'informative_ci': len(unique) > 1}


def evaluate(run, manifest, references, *, repetitions=2000):
    root = Path(run)
    protocol = json.loads((root / 'protocol.json').read_text())
    if protocol.get('shards_complete') is not True:
        raise ValueError('do not evaluate incomplete shards as a completed experiment')
    rows = read_inputs(manifest)
    ids = [r['id'] for r in rows]
    refs = json.loads(Path(references).read_text())
    if set(refs) != set(ids):
        raise ValueError('references must exactly match the complete inference manifest')
    if any(not isinstance(refs[k], list) or not refs[k]
           or any(not isinstance(v, str) for v in refs[k]) for k in ids):
        raise ValueError('reference values must be nonempty lists of strings')
    methods = protocol['methods']
    outputs = {m: json.loads((root / f'{m}.json').read_text()) for m in methods}
    if any(set(v) != set(ids) for v in outputs.values()):
        raise ValueError('method output IDs must match exactly; never impute missing rows')
    scores = {m: {r['id']: diagnostic_score(outputs[m][r['id']]['text'], refs[r['id']][0],
                                           r['answer_type']) for r in rows} for m in methods}
    groups = [r['_pixel_sha256'] for r in rows]
    report = {'metric': 'DIAGNOSTIC closed-EM/open-token-recall, first reference',
              'clinical_accuracy_claim': False, 'reference_sha256': file_hash(references),
              'manifest_sha256': file_hash(manifest), 'identity': protocol['identity'],
              'reference_used_at_inference': False, 'methods': {}, 'per_case': {}}
    for name in methods:
        values = outputs[name]
        delta = [scores[name][k] - scores['incumbent'][k] for k in ids]
        workflows = [values[k].get('agent_workflow', {}) for k in ids]
        report['methods'][name] = {
            'n': len(ids), 'score': statistics.mean(scores[name].values()),
            'delta_to_incumbent': statistics.mean(delta),
            'improved_metric': sum(d > 0 for d in delta), 'harmed_metric': sum(d < 0 for d in delta),
            'changed_text': sum(values[k]['text'] != outputs['incumbent'][k]['text'] for k in ids),
            'accepted': sum(values[k].get('agent_commit', {}).get('accepted', False) for k in ids),
            'mean_reported_seconds': statistics.mean(values[k].get('seconds', 0) for k in ids),
            'region_cases': sum(v.get('region_count', 0) > 0 for v in workflows),
            'candidate_cases': sum(v.get('candidate') is not None for v in workflows),
            'action_failures': sum(len(v.get('execution', {}).get('failed', [])) for v in workflows),
            'model_calls': sum(len(v.get('calls', [])) for v in workflows),
            'paired_cluster_bootstrap_95': cluster_bootstrap(delta, groups, repetitions=repetitions),
            'cost_caveat': 'shared warm assets and incumbent cost; not an independent cold-start benchmark',
        }
        report['per_case'][name] = {k: {'score': scores[name][k],
                                     'delta': scores[name][k] - scores['incumbent'][k]}
                                   for k in ids}
    return report, rows, refs, outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--references', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--anchor-root')
    args = parser.parse_args()
    report, rows, refs, outputs = evaluate(args.run, args.manifest, args.references,
                                          repetitions=args.bootstrap)
    if args.anchor_root:
        sys.path.insert(0, str(Path(args.anchor_root).resolve()))
        from anchor.corrected_sgta.evaluate_medheval_answers import PROTOCOL_VERSION, evaluate_rows
        report['official_evaluator_protocol'] = PROTOCOL_VERSION
        report['official_unchanged_reports'] = {}
        for method, values in outputs.items():
            data = [{'qid': r['id'], 'question': r['question'], 'answer': refs[r['id']][0],
                     'answer_type': r['answer_type'], 'text': values[r['id']]['text'],
                     'image_sha256': r['image_sha256']} for r in rows]
            report['official_unchanged_reports'][method] = evaluate_rows(data)
    atomic_json(args.output, report)
    print(json.dumps(report['methods'], indent=2))


if __name__ == '__main__':
    main()
