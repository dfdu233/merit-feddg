"""Rescore expanded native MERIT and its exact cached control with one frozen scorer."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from run_native_expanded import file_sha, read_json, verify, write_new


def evaluate(root):
    p = verify(root)
    ids = p['ids']
    if len(ids) != 6719 or len(set(ids)) != 6719:
        raise ValueError('complete unchanged PathVQA manifest required')
    for lane in (0, 1):
        done = read_json(root / f'actor-complete-{lane}.json')
        canary = read_json(root / f'canary-complete-{lane}.json')
        if (done['identity'] != p['identity'] or not done['complete']
                or done['n'] != len(ids[lane::2]) or canary['identity'] != p['identity']
                or not canary['complete']):
            raise ValueError('incomplete execution')
    paths = {f.stem: f for f in (root / 'cases').glob('*.json')}
    if set(paths) != set(ids):
        raise ValueError('missing or extra predictions')
    records = {}
    plans = {k: read_json(root / 'plans' / (k + '.json')) for k in ids}
    baselines = {}
    for k in ids:
        record = read_json(paths[k])
        if record['identity'] != p['identity'] or record['id'] != k:
            raise ValueError('prediction identity mismatch')
        plan = plans[k]
        if file_sha(plan['incumbent_path']) != plan['incumbent_sha256']:
            raise ValueError('control cache changed')
        baseline = read_json(plan['incumbent_path'])['outputs']['merit_quilt']
        if not plan['added'] and record['output'] != baseline:
            raise ValueError('unaffected case was changed')
        if plan['added'] and not record.get('old_evidence_preserved'):
            raise ValueError('old evidence preservation failed')
        # Native evidence can contain dense masks. Validate the full record, then
        # retain only scoring fields rather than duplicating all masks in RAM.
        baselines[k] = {'text': baseline['text'], 'seconds': baseline.get('seconds', 0)}
        records[k] = {key: record[key] for key in ('status', 'new_delivered', 'new_calls') if key in record}
        records[k]['output'] = {'text': record['output']['text'],
                                'seconds': record['output'].get('seconds', 0)}

    def check_scorer():
        actual = {str(f.relative_to('/home/dbw/ANCHOR')): file_sha(f)
                  for f in Path('/home/dbw/ANCHOR/anchor').rglob('*.py')}
        if actual != p['scorer_at_prepare']:
            raise ValueError('new protocol scorer changed')

    check_scorer()
    ref_path = Path('/home/dbw/ANCHOR/data/pathvqa/official_test_v1.json')
    refs = {str(v.get('question_id', v.get('qid', v.get('id')))): v for v in read_json(ref_path)}
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

    metrics, scores = {}, {}
    outputs = {'native_merit_quilt': baselines,
               'native_merit_expanded': {k: records[k]['output'] for k in ids}}
    for arm, values in outputs.items():
        rows = [dict(refs[k], text=values[k]['text']) for k in ids]
        metrics[arm] = score(rows)
        details = evaluate_rows(rows)['details']
        scores[arm] = {k: float(d['correct']) if source_task_group(r) == 'ce'
                      else answer_token_recall(d['text'], d['gt_ans'])
                      for k, r, d in zip(ids, rows, details)}
        if abs(sum(scores[arm].values()) / len(ids) - metrics[arm]['primary_unified']['score']) > 1e-12:
            raise ValueError('case metrics mismatch')
    # A changed unused quality-diagnostic file is explicitly recorded. Any loaded
    # scoring module differing from the historical scorer would block comparison.
    for module in tuple(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if not filename:
            continue
        try:
            relative = str(Path(filename).resolve().relative_to('/home/dbw/ANCHOR'))
        except ValueError:
            continue
        if relative in p['old_scorer'] and file_sha(filename) != p['old_scorer'][relative]:
            raise ValueError('a loaded scoring dependency changed from historical protocol')
    delta = {k: scores['native_merit_expanded'][k] - scores['native_merit_quilt'][k] for k in ids}
    affected = [k for k in ids if plans[k]['added']]

    def paired(keys):
        return {'n': len(keys), 'improved': sum(delta[k] > 0 for k in keys),
                'harmed': sum(delta[k] < 0 for k in keys),
                'mean_delta': sum(delta[k] for k in keys) / len(keys) if keys else None,
                'text_changed': sum(outputs['native_merit_expanded'][k]['text'] != baselines[k]['text']
                                    for k in keys)}

    report = {'identity': p['identity'], 'n': len(ids), 'complete': True,
              'evaluation_script_sha256': file_sha(__file__),
              'metrics': metrics, 'vs_native_merit_quilt': paired(ids), 'eligible_cases': paired(affected),
              'image_cluster_bootstrap': cluster_bootstrap(
                  [delta[k] for k in ids], [plans[k]['row']['image_sha256'] for k in ids]),
              'status_counts': dict(Counter(r['status'] for r in records.values())),
              'new_delivered_cases': sum(bool(r.get('new_delivered')) for r in records.values()),
              'new_runtime_calls': sum(r.get('new_calls', 0) for r in records.values()),
              'fresh_actor_calls': len(affected),
              'fresh_runtime_seconds': sum(records[k]['output']['seconds'] for k in affected),
              'inherited_incumbent_seconds': sum(v.get('seconds', 0) for v in baselines.values()),
              'new_prefetch_costs': [read_json(f)['cost'] for f in (root / 'prefetch').glob('*/*.json')],
              'scorer': PROTOCOL_VERSION, 'references_sha256': file_sha(ref_path),
              'scorer_changes_since_control': [k for k, v in p['scorer_at_prepare'].items()
                                               if p['old_scorer'].get(k) != v],
              'scoring_note': 'Both arms rescored together; all loaded historical scorer dependencies match.',
              'metric_note': 'Mixed CLOSED correctness / OPEN token recall, not clinical accuracy.',
              'per_case_scores': scores,
              'case_sha256': {k: file_sha(paths[k]) for k in ids}}
    check_scorer()
    write_new(root / 'evaluation-main.json', report)
    p['shards_complete'] = True
    (root / 'protocol.json').write_text(json.dumps(p, indent=2, sort_keys=True) + '\n')
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ('per_case_scores', 'case_sha256', 'new_prefetch_costs')}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    evaluate(parser.parse_args().run)
