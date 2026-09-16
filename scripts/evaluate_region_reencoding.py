"""Offline same-case 2x2 contrasts. Freeze the existing scorer BEFORE generation."""
import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

from merit_feddg.region_reencoding import ARMS, factorial_contrasts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--anchor-root', type=Path, required=True)
    p.add_argument('--references', type=Path)
    p.add_argument('--freeze-scorer', action='store_true')
    p.add_argument('--partial-diagnostic', action='store_true')
    a = p.parse_args()
    from run_region_reencoding import sha
    from merit_feddg.agent_evaluate import cluster_bootstrap
    from merit_feddg.control_study import validate_outputs
    from merit_feddg.open_study import atomic_json, fingerprint
    names = ('anchor/corrected_sgta/evaluate_medheval_answers.py', 'anchor/medeval/evaluate_mixed_vqa_table.py')
    current = {n: sha(a.anchor_root/n) for n in names}
    pin = a.run/'scorer-frozen.json'
    frozen = json.loads((a.run/'frozen.json').read_text())
    if current != frozen.get('scorer_hashes'):
        raise ValueError('scorer differs from pre-generation frozen hashes')
    if a.freeze_scorer:
        if pin.exists() and json.loads(pin.read_text()) != current:
            raise ValueError('cannot change scorer pin')
        atomic_json(pin, current)
        print('Pinned existing scorer, no references read')
        return
    if not pin.exists() or json.loads(pin.read_text()) != current or not a.references:
        raise ValueError('unchanged scorer pin and separate references required')
    frozen = json.loads((a.run/'frozen.json').read_text())
    identity = fingerprint(frozen)
    if frozen['schema'] != 'region-reencoding-v1' or frozen['arms'] != list(ARMS):
        raise ValueError('wrong experiment schema')
    if frozen['evaluator_sha256'] != sha(__file__):
        raise ValueError('frozen evaluator changed')
    import fcntl
    with (a.run/'.worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        paths = {f.stem: f for f in (a.run/'cases').glob('*.json')}
        rows = frozen['rows'][:len(paths)] if a.partial_diagnostic else frozen['rows']
        if not rows or set(paths) != {fingerprint(r['id']) for r in rows}:
            raise ValueError('only exact complete set or explicit ordered prefix may be scored')
        if not a.partial_diagnostic:
            complete = json.loads((a.run/'complete.json').read_text())
            if (complete.get('identity') != identity or complete.get('full_manifest_complete') is not True
                    or complete.get('n') != len(rows)):
                raise ValueError('missing valid completion marker')
        output = a.run/('partial-evaluation.json' if a.partial_diagnostic else 'evaluation.json')
        if output.exists():
            raise ValueError('refuse to overwrite evaluation')
        refs = json.loads(a.references.read_text())
        if (set(refs) != {r['id'] for r in frozen['rows']} or any(not isinstance(v, list) or not v
                or any(not isinstance(x, str) or not x for x in v) for v in refs.values())):
            raise ValueError('complete reference alignment required')
        records = {r['id']: json.loads(paths[fingerprint(r['id'])].read_text()) for r in rows}
        for key, record in records.items():
            if record.get('identity') != identity or record.get('id') != key:
                raise ValueError('record identity mismatch')
            validate_outputs(record, ARMS)
        sys.path.insert(0, str(a.anchor_root.resolve()))
        from anchor.corrected_sgta.evaluate_medheval_answers import evaluate_rows, PROTOCOL_VERSION
        from anchor.medeval.evaluate_mixed_vqa_table import answer_token_recall
        scores = {}
        for arm in ARMS:
            ev = evaluate_rows([{'qid': r['id'], 'question': r['question'], 'answer_type': r['answer_type'],
                'answer': refs[r['id']][0], 'text': records[r['id']]['arms'][arm]['text']} for r in rows])
            detail = {d['question_id']: d for d in ev['details']}
            scores[arm] = {r['id']: float(detail[r['id']]['correct']) if r['answer_type']=='closed' else
                answer_token_recall(records[r['id']]['arms'][arm]['text'], refs[r['id']][0]) for r in rows}
        applicable = [r for r in rows if records[r['id']]['applicable']]
        report = {'identity': identity, 'n': len(rows), 'applicable': len(applicable),
            'full_manifest_n': len(frozen['rows']), 'partial_diagnostic': a.partial_diagnostic,
            'selection_warning': 'ordered nonrandom prefix; cluster CI does not correct selection' if a.partial_diagnostic else None,
            'scorer': PROTOCOL_VERSION, 'scorer_hashes': current, 'reference_sha256': sha(a.references),
            'metric': 'ANCHOR CLOSED + OPEN token recall; not clinical correctness proof',
            'arms': {}, 'contrasts': {}, 'per_case_scores': scores,
            'unavailability': dict(Counter(v['region_audit']['reason'] for v in records.values())),
            'startup_seconds': sum(json.loads(f.read_text())['seconds'] for f in a.run.glob('startup-*.json'))}
        for arm in ARMS:
            out = [records[r['id']]['arms'][arm] for r in rows]
            delta = [scores[arm][r['id']]-scores['deletion'][r['id']] for r in rows]
            report['arms'][arm] = {'score': statistics.mean(scores[arm].values()),
                'delta_deletion': statistics.mean(delta), 'gains': sum(d>0 for d in delta), 'harms': sum(d<0 for d in delta),
                'changed_tokens_vs_deletion': sum(v['token_ids']!=records[r['id']]['arms']['deletion']['token_ids'] for r,v in zip(rows,out)),
                'mean_seconds_all': statistics.mean(v['seconds'] for v in out),
                'mean_seconds_applicable': statistics.mean(records[r['id']]['arms'][arm]['seconds'] for r in applicable) if applicable else None,
                'sum_extra_vision_encodes': sum(v.get('extra_vision_encodes',0) for v in out),
                'ci_vs_deletion': cluster_bootstrap(delta,[r['image_sha256'] for r in rows])}
        for subset_name, subset in (('all', rows), ('applicable', applicable)):
            if not subset:
                report['contrasts'][subset_name] = None
                continue
            contrasts = [factorial_contrasts({k:scores[k][r['id']] for k in ARMS}) for r in subset]
            report['contrasts'][subset_name] = {key: {'mean': statistics.mean(v[key] for v in contrasts),
                'ci': cluster_bootstrap([v[key] for v in contrasts], [r['image_sha256'] for r in subset])} for key in contrasts[0]}
        report['wall_seconds_completed'] = sum(r['wall_seconds'] for r in records.values())
        report['interrupted_case_cost_included'] = False
        report['causal_or_clinical_guarantee'] = False
        atomic_json(output, report)
        print('SAVED', output)


if __name__ == '__main__':
    main()
