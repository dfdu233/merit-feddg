"""Offline frozen formal scoring; requires both full scheduling lanes complete."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from run_coverage_pathvqa import OLD, file_sha, original, read_json, verify, write_new
from run_pathology_quilt_pilot import scorer_identity


def evaluate(root):
    protocol = verify(root)
    ids = protocol['ids']
    if len(ids) != len(set(ids)) or len(ids) != 6719:
        raise ValueError('full unique PathVQA ID set required')
    for lane in (0, 1):
        done = read_json(root / f'actor-complete-{lane}.json')
        if (done['identity'] != protocol['identity'] or not done['complete']
                or done['n'] != len(ids[lane::2])):
            raise ValueError('incomplete/mismatched actor lane')
        canary = read_json(root / f'canary-complete-{lane}.json')
        if canary['identity'] != protocol['identity'] or not canary['complete']:
            raise ValueError('canary incomplete')
    paths = {p.stem: p for p in (root / 'cases').glob('*.json')}
    if set(paths) != set(ids):
        raise ValueError('missing/extra cases; never fill missing rows')
    records = {k: read_json(paths[k]) for k in ids}
    plans = {k: read_json(root / 'plans' / (k + '.json')) for k in ids}
    for k, record in records.items():
        if record['id'] != k or record['identity'] != protocol['identity']:
            raise ValueError('case identity mismatch')
        plan = plans[k]
        if file_sha(plan['incumbent_path']) != plan['incumbent_sha256']:
            raise ValueError('cached native incumbent changed')
        if record['status'] == 'real_candidate' and record['transport']['omitted']:
            raise ValueError('selected evidence was not delivered')
    if scorer_identity('/home/dbw/ANCHOR') != protocol['scorer']:
        raise ValueError('formal scorer drift')
    refs_path = Path('/home/dbw/ANCHOR/data/pathvqa/official_test_v1.json')
    refs = {str(r.get('question_id', r.get('qid', r.get('id')))): r
            for r in read_json(refs_path)}
    if set(refs) != set(ids):
        raise ValueError('reference identity mismatch')
    sys.path.insert(0, '/home/dbw/ANCHOR')
    from anchor.corrected_sgta.evaluate_medheval_answers import PROTOCOL_VERSION, evaluate_rows
    from anchor.medeval.evaluate_mixed_vqa_table import (
        answer_token_recall,
        score,
        source_task_group,
    )

    from merit_feddg.agent_evaluate import cluster_bootstrap

    frozen = read_json(OLD / 'frozen.json')
    values = {
        'expanded_pool': {k: records[k]['output'] for k in ids},
        'native_merit_quilt': {k: read_json(plans[k]['incumbent_path'])['outputs']['merit_quilt']
                              for k in ids},
        'original_compact': {k: original(k, frozen)[0] for k in ids},
    }
    metrics, scores = {}, {}
    for arm, outputs in values.items():
        rows = [dict(refs[k], text=outputs[k]['text']) for k in ids]
        metrics[arm] = score(rows)
        details = evaluate_rows(rows)['details']
        scores[arm] = {k: float(d['correct']) if source_task_group(r) == 'ce'
                      else answer_token_recall(d['text'], d['gt_ans'])
                      for k, r, d in zip(ids, rows, details)}
        if abs(sum(scores[arm].values()) / len(ids)
               - metrics[arm]['primary_unified']['score']) > 1e-12:
            raise ValueError('per-case/main metric inconsistency')
    new_ids = [k for k in ids if plans[k]['new_experts']]
    active = [k for k in ids if records[k]['status'] == 'real_candidate']

    def paired(control, keys):
        deltas = [scores['expanded_pool'][k] - scores[control][k] for k in keys]
        return {'n': len(keys), 'improved': sum(v > 0 for v in deltas),
                'harmed': sum(v < 0 for v in deltas),
                'mean_delta': sum(deltas) / len(deltas) if deltas else None,
                'text_changed': sum(values['expanded_pool'][k]['text'] != values[control][k]['text']
                                    for k in keys),
                'image_cluster_bootstrap': cluster_bootstrap(
                    deltas, [plans[k]['row']['image_sha256'] for k in keys]) if keys else None}

    expert_costs = [read_json(p)['cost'] for p in (root / 'new-evidence').glob('*/*.json')]
    report = {
        'identity': protocol['identity'], 'n': len(ids), 'complete': True,
        'scorer': PROTOCOL_VERSION, 'scorer_hashes': protocol['scorer'],
        'reference_sha256': file_sha(refs_path), 'metrics': metrics,
        'metric_meaning': 'CLOSED formal correctness + OPEN reference-token recall; not clinical accuracy',
        'comparisons': {c: {name: paired(c, keys) for name, keys in
                          [('full', ids), ('real_candidate', active), ('new_expert', new_ids)]}
                        for c in ('native_merit_quilt', 'original_compact')},
        'candidate_n': len(active), 'candidate_coverage': len(active) / len(ids),
        'new_expert_case_n': len(new_ids), 'new_expert_case_coverage': len(new_ids) / len(ids),
        'status_counts': dict(Counter(r['status'] for r in records.values())),
        'selected_counts': dict(Counter(n for r in records.values() for n in r['decision']['selected'])),
        'new_expert_cost_records': expert_costs,
        'fresh_actor_seconds': sum(records[k]['output']['seconds'] for k in active),
        'fresh_actor_calls': len(active),
        'inherited_native_total_seconds': sum(v.get('seconds', 0) for v in values['native_merit_quilt'].values()),
        'inherited_compact_total_seconds': sum(v.get('seconds', 0) for v in values['original_compact'].values()),
        'per_case_scores': scores, 'case_sha256': {k: file_sha(paths[k]) for k in ids},
        'limitations': [
            'Only the expanded-pool arm was newly generated; controls reuse complete caches.',
            'Historical compact input limit was 2048; new/native Quilt limits are 8192.',
            'Selection also changes which old evidence is delivered: total delta is not new-model-only gain.',
            'Source overlap and broad question-parser applicability remain unresolved.',
            'Cached expert costs are inherited, not zero; model-load and canary overhead are additional.',
        ],
    }
    if scorer_identity('/home/dbw/ANCHOR') != protocol['scorer']:
        raise ValueError('scorer changed during evaluation')
    write_new(root / 'evaluation-main.json', report)
    # Only this experiment's completion field changes; all identities stay frozen.
    protocol['shards_complete'] = True
    (root / 'protocol.json').write_text(json.dumps(protocol, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'n': len(ids), 'metrics': metrics, 'candidate_n': len(active),
                      'new_expert_case_n': len(new_ids)}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    evaluate(parser.parse_args().run)
