"""Offline paired diagnostics. Labels never enter decoding.py.

Bootstrap intervals are empirical decision aids, not clinical guarantees.
Two predeclared confirmation looks use a Bonferroni error budget; no repeated
peeking at intermediate scores and no candidate switching after confirmation.
"""
from __future__ import annotations
import math
from statistics import NormalDist
import numpy as np


def paired_interval(values, groups, *, alpha=.05, repetitions=4000, seed=0):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) != len(groups) or not np.isfinite(values).all():
        raise ValueError('Finite aligned observations and image groups required')
    if not 0 < alpha < 1 or type(repetitions) is not int or repetitions < 100:
        raise ValueError('Invalid frozen bootstrap settings')
    keys = sorted(set(groups))
    if len(keys) < 2:
        return dict(mean=float(values.mean()) if len(values) else None, lower=None, upper=None,
                    groups=len(keys), n=len(values), sufficient=False)
    sums = np.array([values[np.array(groups) == g].sum() for g in keys])
    counts = np.array([sum(x == g for x in groups) for g in keys])
    rng = np.random.default_rng(seed)
    estimates = []
    # Bounded temporary storage rather than reps x cohort size allocation.
    for start in range(0, repetitions, 128):
        indices = rng.integers(0, len(keys), size=(min(128, repetitions-start), len(keys)))
        estimates.extend((sums[indices].sum(1) / counts[indices].sum(1)).tolist())
    lo, hi = np.quantile(estimates, [alpha / 2, 1-alpha / 2])
    return dict(mean=float(values.mean()), lower=float(lo), upper=float(hi), groups=len(keys),
                n=len(values), sufficient=True, alpha=alpha, repetitions=repetitions, seed=seed)


def wilson(successes, n, alpha):
    if type(n) is not int or type(successes) is not int or not 0 <= successes <= n:
        raise ValueError('Invalid count')
    if not n:
        return dict(rate=None, lower=0.0, upper=1.0, n=0)
    z = NormalDist().inv_cdf(1-alpha/2)
    p, denom = successes/n, 1+z*z/n
    center = (p+z*z/(2*n))/denom
    half = z/denom * math.sqrt(p*(1-p)/n+z*z/(4*n*n))
    return dict(rate=p, lower=max(0.0, center-half), upper=min(1.0, center+half), n=n)


def scored_rows(predictions, scores):
    """Reject missing/extra scores, duplicate IDs, nonfinite values and imputation.

    A score document is bound externally to the prediction file SHA. Failed
    predictions require null score; a label cannot turn a failed run into a pass.
    """
    by_id = {}
    for row in scores:
        if row['id'] in by_id:
            raise ValueError('Duplicate score ID')
        by_id[row['id']] = row
    if len({r['id'] for r in predictions}) != len(predictions):
        raise ValueError('Duplicate prediction ID')
    if set(by_id) != {p['id'] for p in predictions}:
        raise ValueError('Scores must retain the exact planned denominator')
    merged = []
    for pred in predictions:
        row = by_id[pred['id']]
        arms = {}
        for name, arm in pred['arms'].items():
            if name not in row['scores']:
                raise ValueError('Missing arm score')
            value = row['scores'][name]
            if arm.get('status') == 'failed':
                if value is not None:
                    raise ValueError('Failed predictions must never receive imputed scores')
            elif type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError('Require frozen bounded [0,1] metric')
            arms[name] = value
        if set(row['scores']) != set(arms):
            raise ValueError('Unexpected score arm')
        merged.append(dict(pred, scores=arms))
    return merged


def quality(rows, *, config, confirmation=False):
    """Candidate must beat BOTH baseline and compact and a matched-cost control."""
    failures = sum(any(x is None for x in r['scores'].values()) for r in rows)
    if failures:
        return dict(verdict='technical_failure', planned_n=len(rows), failed_n=failures,
                    reason='No complete paired denominator; repair does not choose another algorithm')
    n = len(rows)
    if n == 0:
        return dict(verdict='technical_failure', reason='Empty cohort', planned_n=0)
    group_ids = [r['pixel_sha256'] for r in rows]
    candidate = np.array([r['scores']['candidate'] for r in rows])
    base = np.array([r['scores']['generalist'] for r in rows])
    expert = np.array([r['scores']['compact'] for r in rows])
    control = np.array([r['scores']['control'] for r in rows])
    old_gains, old_harms = expert > base, expert < base
    retained = int(((candidate >= expert) & old_gains).sum())
    repaired = int(((candidate > expert) & old_harms).sum())
    native_full = base == 1
    damaged = int(((candidate < 1) & native_full).sum())
    distinct = len(set(group_ids))
    domains = sorted({r['cohort'] for r in rows})
    # Three contrasts, two risk summaries and each cohort non-inferiority contrast,
    # at two predeclared looks. These intervals are still bootstrap approximations.
    alpha = config['alpha'] / (2 * (5 + len(domains))) if confirmation else config['alpha']
    common = dict(alpha=alpha, repetitions=config['bootstrap_repetitions'], seed=config['seed'])
    contrasts = {name: paired_interval(candidate-v, group_ids, **common)
                 for name, v in [('generalist', base), ('compact', expert), ('control', control)]}
    domain_results = {}
    for domain in domains:
        ids = [i for i, r in enumerate(rows) if r['cohort'] == domain]
        domain_results[domain] = paired_interval((candidate-base)[ids],
                                                 [group_ids[i] for i in ids], **common)
    # Wilson is descriptive only when there are multiple questions per image;
    # confirmation below refuses IID risk certification in that setting.
    risk = wilson(damaged, int(native_full.sum()), alpha)
    retention = wilson(retained, int(old_gains.sum()), alpha)
    result = dict(planned_n=n, completed_n=n, image_groups=distinct, cohorts=len(domains),
        contrasts=contrasts, cohort_contrasts=domain_results, original_gains=int(old_gains.sum()),
        retained_gains=retained, original_harms=int(old_harms.sum()), repaired_harms=repaired,
        improvement_cases=int((candidate>base).sum()), harmful_cases=int((candidate<base).sum()),
        changed_cases=sum(r['arms']['candidate']['token_ids'] != r['arms']['compact']['token_ids'] for r in rows),
        active_cases=sum(not r['arms']['candidate'].get('bypass', False) for r in rows),
        damage_fraction=risk, gain_retention=retention,
        mean_seconds=float(np.mean([r['arms']['candidate']['seconds'] for r in rows])),
        alpha_per_comparison=alpha,
        clinical_claim='Not clinical accuracy; frozen automatic metric only')
    if not result['active_cases'] or not result['changed_cases']:
        return dict(result, verdict='fail', reason='All-bypass/no-output-change cannot pass')
    if result['mean_seconds'] > config['max_seconds_per_case']:
        return dict(result, verdict='fail', reason='Predeclared inference cost budget exceeded')
    sufficient = distinct >= config['min_confirmation_images' if confirmation else 'min_development_images']
    sufficient &= int(old_gains.sum()) >= config['min_original_gains']
    sufficient &= int(old_harms.sum()) >= config['min_original_harms']
    if confirmation:
        sufficient &= len(domains) >= config['min_confirmation_cohorts']
        sufficient &= all(v['groups'] >= config['min_images_per_cohort'] for v in domain_results.values())
        # Independence unit is an image. Conservatively require one question per
        # image for Wilson release gates; otherwise block promotion, not pretend IID.
        sufficient &= distinct == n
    if not sufficient:
        return dict(result, verdict='inconclusive', reason='Insufficient independent units or benefit/harm coverage')
    point_ok = all(contrasts[k]['mean'] >= config['min_delta'] for k in ('generalist', 'compact'))
    point_ok &= contrasts['control']['mean'] > 0
    point_ok &= repaired >= config['min_repaired_harms']
    point_ok &= retained / int(old_gains.sum()) >= config['min_gain_retention']
    point_ok &= risk['rate'] is not None and risk['rate'] <= config['max_damage_rate']
    if not confirmation:
        return dict(result, verdict='pass' if point_ok else 'fail',
                    reason='Development screening, NOT permission for large-scale evaluation')
    ci_ok = all(v['sufficient'] and v['lower'] > 0 for v in contrasts.values())
    ci_ok &= all(v['sufficient'] and v['lower'] >= -config['cohort_noninferiority_margin']
                 for v in domain_results.values())
    ci_ok &= risk['upper'] <= config['max_damage_rate']
    ci_ok &= retention['lower'] >= config['min_gain_retention']
    if point_ok and ci_ok:
        return dict(result, verdict='pass', reason='Frozen candidate passed independent source confirmation')
    clear_fail = any(v['sufficient'] and v['upper'] <= 0 for v in contrasts.values())
    clear_fail |= risk['lower'] > config['max_damage_rate']
    clear_fail |= retention['upper'] < config['min_gain_retention']
    return dict(result, verdict='fail' if clear_fail else 'inconclusive',
                reason='Reject or consume only the predeclared independent extension; never retune here')


def localization(rows, config):
    if any(r.get('diagnostic', {}).get('status') == 'failed' for r in rows):
        return dict(verdict='technical_failure', reason='Diagnostic forward failed')
    active = [r for r in rows if r.get('diagnostic', {}).get('status') == 'divergence'
              and r['scores']['compact'] != r['scores']['generalist']]
    gains = sum(r['scores']['compact'] > r['scores']['generalist'] for r in active)
    harms = len(active)-gains
    if gains < config['min_original_gains'] or harms < config['min_original_harms']:
        return dict(verdict='inconclusive', reason='Need both beneficial and harmful divergence cases')
    sites = sorted({x['layer'] for r in active for x in r['diagnostic']['sites']})
    summaries = {}
    for site in sites:
        values, groups = [], []
        for row in active:
            records = [x for x in row['diagnostic']['sites'] if x['layer'] == site]
            if len(records) != 1:
                return dict(verdict='technical_failure', reason='Incomplete or duplicated localization site')
            rec = records[0]
            effect = rec['restore']['probability_margin']-rec['roll']['probability_margin']
            # Positive = promotes baseline when compact harmed, expert when compact helped.
            values.append(-(row['scores']['compact']-row['scores']['generalist'])*effect)
            groups.append(row['pixel_sha256'])
        summaries[str(site)] = paired_interval(values, groups, alpha=config['alpha']/max(1,len(sites)),
                    repetitions=config['bootstrap_repetitions'], seed=config['seed'])
    passed = [k for k, v in summaries.items() if v['sufficient'] and v['lower'] > 0
              and v['groups'] >= config['min_localization_images']]
    if not passed:
        return dict(verdict='fail', sites=summaries, reason='No supported directional localization; try independent C')
    chosen = max(passed, key=lambda k: (summaries[k]['lower'], -int(k)))
    return dict(verdict='pass', layer=int(chosen), sites=summaries,
                reason='Global development-selected site, never per-case label-based restoration')
