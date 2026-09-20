"""Offline matched pathway diagnostics using the unchanged source TRAIN scorer."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.append('/home/dbw/ANCHOR')
from anchor.corrected_sgta.evaluate_medheval_answers import PROTOCOL_VERSION, evaluate_rows
from anchor.medeval.evaluate_mixed_vqa_table import answer_token_recall

from merit_feddg.agent_evaluate import cluster_bootstrap

ARMS = ('generalist', 'compact', 'restore_mass', 'restore_vector')


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, value):
    with Path(p).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def observation(arm):
    events = [e for e in arm.get('events', []) if e['branch'] == 'receiver']
    def extent(key):
        values = [e[key] for e in events if e.get(key) is not None]
        return dict(mean=statistics.mean(values), min=min(values), max=max(values)) if values else None
    return dict(patched_steps=sum(e['patched'] for e in events), observed_steps=len(events),
                zero_delta_steps=sum(e['delta_norm'] == 0 for e in events),
                receiver_visual_mass=extent('visual_mass'), reference_visual_mass=extent('reference_visual_mass'),
                delta_norm=extent('delta_norm'),
                failure=arm.get('failure'), restored_mass_interpretation='restore_mass targets reference mass; restored attention is not renormalized',
                seconds=arm.get('seconds'), reference_forwards=arm.get('reference_forwards'),
                receiver_forwards=arm.get('receiver_forwards'), output_tokens=len(arm['token_ids']),
                bypass=arm.get('identical_input_bypass', False), stop_reason=arm.get('stop_reason'),
                peak_allocated_bytes=arm.get('peak_allocated_bytes'),
                peak_reserved_bytes=arm.get('peak_reserved_bytes'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--source-run', type=Path, required=True)
    p.add_argument('--references', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    source, protocol = read(args.source_run/'protocol.json'), read(args.run/'protocol.json')
    rows = source['rows']
    ids = [r['id'] for r in rows]
    if protocol['source_protocol_sha256'] != sha(args.source_run/'protocol.json'):
        raise ValueError('Changed source protocol')
    if read(args.run/'complete.json') != dict(identity=protocol['identity'], n=len(ids), clinical_efficacy_evaluated=False):
        raise ValueError('Complete exact queue required before scoring')
    if {p.stem for p in (args.run/'cases').glob('*.json')} != set(ids):
        raise ValueError('Different case set')
    cases = {k: read(args.run/'cases'/(k+'.json')) for k in ids}
    old = {k: read(args.source_run/'cases'/(k+'.json')) for k in ids}
    for k, c in cases.items():
        if not c['complete'] or c['identity'] != protocol['identity']:
            raise ValueError('Mixed/incomplete generation')
        if sha(args.source_run/'cases'/(k+'.json')) != c['source_case_sha256']:
            raise ValueError('Changed source case')
    refs = read(args.references)
    scorer_paths = [Path(inspect.getfile(evaluate_rows)), Path(inspect.getfile(answer_token_recall))]
    scorer_hashes = {str(p): sha(p) for p in scorer_paths}
    scores = {}
    for arm in ARMS:
        scored = evaluate_rows([dict(qid=k, question=r['question'], answer_type=r['answer_type'],
                                    answer=refs[k][0], text=cases[k]['arms'][arm]['text'])
                                for k, r in zip(ids, rows) if not cases[k]['arms'][arm].get('failure')])
        details = {v['question_id']: v for v in scored['details']}
        scores[arm] = {k: (None if cases[k]['arms'][arm].get('failure') else
                      float(details[k]['correct']) if r['answer_type'] == 'closed' else
                      answer_token_recall(cases[k]['arms'][arm]['text'], refs[k][0])) for k, r in zip(ids, rows)}
    observations = {a: {k: observation(cases[k]['arms'][a]) for k in ids} for a in ARMS[2:]}
    groups = [r.get('pixel_sha256', r['image_sha256']) for r in rows]
    labels = {k: dict(answer_type=r['answer_type'], language=r.get('q_lang', 'unrecorded'),
                      modality=old[k]['route'].get('modality', 'unrecorded'), task=r['task']) for k, r in zip(ids, rows)}
    summary = dict(identity=protocol['identity'], planned_n=len(ids), processed_n=len(ids), technical_failures=0,
                   metric='Frozen ANCHOR CLOSED parser + OPEN token recall (first reference); not clinical accuracy',
                   scorer_version=PROTOCOL_VERSION, scorer_hashes=scorer_hashes,
                   bootstrap=dict(seed=0, repetitions=2000, unit='pixel image', confidence=.95), arms={},
                   failure_table=[], paired={}, splits={}, mechanism={}, costs={},
                   source_damage_n=sum(scores['compact'][k] < scores['generalist'][k] for k in ids),
                   source_gain_n=sum(scores['compact'][k] > scores['generalist'][k] for k in ids))
    common = [k for k in ids if all(scores[a][k] is not None for a in ARMS)]
    summary['all_arms_successful_n'] = len(common)
    summary['common_completed_subset'] = dict(n=len(common), scores={
        a:statistics.mean(scores[a][k] for k in common) if common else None for a in ARMS})
    for arm in ARMS:
        valid = [v for v in scores[arm].values() if v is not None]
        failed = [k for k in ids if scores[arm][k] is None]
        summary['failure_table'].extend(dict(sample_id=hashlib.sha256(('pathway-train-v1:'+k).encode()).hexdigest()[:20],
            arm=arm, error=cases[k]['arms'][arm]['failure']) for k in failed)
        summary['arms'][arm] = dict(planned_n=len(ids), completed_n=len(valid), failed_n=len(failed),
            score=statistics.mean(valid) if valid else None,
            score_scope='completed arm subset; see full planned-denominator bounds',
            planned_score_lower_bound=sum(valid)/len(ids),
            planned_score_upper_bound=(sum(valid)+len(failed))/len(ids),
            full_credit=sum(v == 1 for v in valid),
            empty=sum(cases[k]['arms'][arm]['text'] == '' for k in ids))
        summary['paired'][arm] = {}
        for baseline in ARMS[:2]:
            paired = [k for k in ids if scores[arm][k] is not None]
            ds = [scores[arm][k]-scores[baseline][k] for k in paired]
            gs = [groups[ids.index(k)] for k in paired]
            summary['paired'][arm][baseline] = dict(completed_pair_n=len(paired), planned_n=len(ids),
                unavailable=len(ids)-len(paired), improved=sum(d > 0 for d in ds),
                harmed=sum(d < 0 for d in ds), unchanged=sum(d == 0 for d in ds),
                delta=statistics.mean(ds) if ds else None,
                image_ci95=cluster_bootstrap(ds, gs, repetitions=2000, seed=0) if ds else None)
    summary['technical_failures'] = len(summary['failure_table'])
    for label in ('answer_type', 'language', 'modality', 'task'):
        summary['splits'][label] = {}
        for value in sorted({labels[k][label] for k in ids}):
            subset = [k for k in ids if labels[k][label] == value]
            summary['splits'][label][value] = dict(n=len(subset), scores={a:dict(completed_n=sum(scores[a][k] is not None for k in subset),
                score=statistics.mean(scores[a][k] for k in subset if scores[a][k] is not None)
                if any(scores[a][k] is not None for k in subset) else None) for a in ARMS})
    for arm in ARMS[2:]:
        base, compact, current = (scores[a] for a in ('generalist', 'compact', arm))
        harm = [k for k in ids if compact[k] < base[k] and current[k] is not None]
        gain = [k for k in ids if compact[k] > base[k] and current[k] is not None]
        summary['mechanism'][arm] = dict(
            original_harms_unavailable=summary['source_damage_n']-len(harm),
            original_gains_unavailable=summary['source_gain_n']-len(gain),
            original_harms_fully_repaired=sum(current[k] >= base[k] for k in harm),
            original_harms_partially_or_fully_improved=sum(current[k] > compact[k] for k in harm),
            original_gains_fully_retained=sum(current[k] >= compact[k] for k in gain),
            original_gains_still_above_baseline=sum(current[k] > base[k] for k in gain),
            original_gains_reduced=sum(current[k] < compact[k] for k in gain),
            original_gains_lost_to_baseline_or_worse=sum(current[k] <= base[k] for k in gain),
            new_full_credit_candidates=sum(base[k] < 1 and compact[k] < 1 and current[k] == 1 for k in ids),
            changed_token_sequence_vs_compact=sum(cases[k]['arms'][arm]['token_ids'] != cases[k]['arms']['compact']['token_ids'] for k in ids if current[k] is not None),
            changed_text_vs_compact=sum(cases[k]['arms'][arm]['text'] != cases[k]['arms']['compact']['text'] for k in ids if current[k] is not None))
        obs = list(observations[arm].values())
        events = [e for k in ids for e in cases[k]['arms'][arm]['events'] if e['branch']=='receiver']
        summary['costs'][arm] = dict(seconds=sum(v['seconds'] for v in obs),
            reference_forwards=sum(v['reference_forwards'] for v in obs),
            receiver_forwards=sum(v['receiver_forwards'] for v in obs),
            output_tokens=sum(v['output_tokens'] for v in obs),
            intervention_cases=sum(v['patched_steps'] > 0 for v in obs),
            patched_steps=sum(v['patched_steps'] for v in obs),
            zero_delta_steps=sum(v['zero_delta_steps'] for v in obs),
            bypass_cases=sum(v['bypass'] for v in obs),
            stop_reasons=dict(Counter(v['stop_reason'] for v in obs)),
            peak_allocated_bytes=max(v['peak_allocated_bytes'] for v in obs),
            peak_reserved_bytes=max(v['peak_reserved_bytes'] for v in obs),
            event_means={key:statistics.mean(e[key] for e in events if e.get(key) is not None) if any(e.get(key) is not None for e in events) else None
                         for key in ('visual_mass','reference_visual_mass','delta_norm')})
    summary['costs']['preparation_seconds'] = sum(c['preparation_seconds'] for c in cases.values())
    summary['costs']['model_load_seconds'] = sum(read(p)['seconds'] for p in args.run.glob('load-*.json'))
    summary['costs']['audit_seconds'] = sum(c.get('audit_control', {}).get('seconds', 0) for c in cases.values())
    summary['costs']['off_control_seconds'] = sum(v['seconds'] for c in cases.values() for v in c.get('control_cost', {}).values())
    summary['costs']['native_control_seconds'] = None
    summary['costs']['limits'] = 'Cached Baseline cost is unknown, not zero; no deployment speedup ratio. Native parity timing and initial failed attempts are not included in decode totals.'
    summary['input_parity'] = {field:dict(Counter(c['input_parity'][field] for c in cases.values()))
                               for field in ('historical_prompt_hash','historical_transport')}
    summary['limitations'] = ['TRAIN development diagnostic only; no parameter training or TEST inference.',
        'No nonvisual/random-path control: specific causal mechanism remains unresolved.',
        'OPEN score increases can reflect wording/coverage, not medical factual correction.',
        'No patient-level separation proof; image-level audit is reported separately.',
        'Native eager attention on this environment; no claim of parity across kernels.',
        'No per-head post-intervention normalized attention tensor was recorded.']
    public, private = [], []
    for k in ids:
        anonymous = hashlib.sha256(('pathway-train-v1:'+k).encode()).hexdigest()[:20]
        item = dict(sample_id=anonymous, labels=labels[k], failure={a:cases[k]['arms'][a]['failure'] for a in ARMS if cases[k]['arms'][a].get('failure')},
                    arms={a:dict(prediction=cases[k]['arms'][a]['text'], score=scores[a][k],
                                 statistics=observations.get(a, {}).get(k), reused=a in ARMS[:2]) for a in ARMS})
        public.append(item)
        if any(scores[a][k] is not None and scores[a][k] != scores['compact'][k] for a in ARMS[2:]):
            private.append(dict(id=k, anonymous_id=anonymous, reference=refs[k][0],
                question=next(r['question'] for r in rows if r['id']==k), result=item,
                interpretation='unreviewed wording/coverage change; no clinical correction claim'))
    if args.output.exists():
        raise FileExistsError('Preserve previous evaluation')
    args.output.mkdir(parents=True)
    write(args.output/'summary.json', summary)
    with (args.output/'per-case-results.jsonl').open('x') as stream:
        for item in public:
            stream.write(json.dumps(item,ensure_ascii=False)+'\n')
    diagnostic_path = args.run/'offline-diagnostics.json'
    if diagnostic_path.exists():
        if read(diagnostic_path) != private:
            raise ValueError('Existing offline diagnostics differ; preserve the prior evaluation')
    else:
        write(diagnostic_path, private)
    write(args.output/'provenance.json', dict(source_identity=source['identity'], output_identity=protocol['identity'],
        source_protocol_sha256=sha(args.source_run/'protocol.json'), source_case_sha256={hashlib.sha256(('pathway-train-v1:'+k).encode()).hexdigest()[:20]:v for k,v in protocol['case_sha256'].items()},
        generation=protocol['generation'], code=protocol['code'], import_provenance=protocol['import_provenance'],
        native_generation_config=protocol['native_generation_config'],
        model_config=protocol['model_config'], checkpoint_file_stats_not_content_hashes=protocol['checkpoint_file_stats_not_content_hashes'],
        adapter_source_sha256=protocol['adapter_source_sha256'], torch_version=protocol['torch_version'],
        transformers_version=protocol['transformers_version'], scorer_hashes=scorer_hashes,
        references_sha256=sha(args.references), evaluator_sha256=sha(__file__),
        source_evaluator_sha256=sha('/home/dbw/merit-feddg-huatuo-critic/scripts/evaluate_huatuo_admission_probe.py')))
    if any(sha(p) != scorer_hashes[str(p)] for p in scorer_paths):
        raise ValueError('Scorer changed during evaluation')
    print(json.dumps(dict(n=len(ids),arms=summary['arms'],mechanism=summary['mechanism']),indent=2))


if __name__ == '__main__':
    main()
