"""Finite scientific decision graph with no forced-success cycle."""
from __future__ import annotations
import math

DEFAULTS = dict(alpha=.05, bootstrap_repetitions=4000, seed=0,
    diagnostic_prefix_tokens=16, min_localization_images=8,
    min_development_images=32, min_confirmation_images=64, min_images_per_cohort=24,
    min_confirmation_cohorts=2, min_original_gains=3, min_original_harms=3,
    min_repaired_harms=1, min_delta=.005, min_gain_retention=.75,
    max_damage_rate=.10, cohort_noninferiority_margin=.02,
    max_seconds_per_case=10.0, max_node_attempts=2, max_total_forwards=200000)
TERMINAL = {'READY_FOR_SCALE', 'STOP_NO_CANDIDATE', 'STOP_FAILED_CONFIRMATION',
            'STOP_INCONCLUSIVE', 'BLOCKED_TECHNICAL', 'BLOCKED_DATA', 'BLOCKED_BUDGET'}


def validate_policy(policy):
    if set(policy) != set(DEFAULTS):
        raise ValueError('Policy must explicitly freeze every field; unknown/missing fields rejected')
    for key, value in policy.items():
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError('Finite policy settings required')
        if type(DEFAULTS[key]) is int and (type(value) is not int or value < (0 if key == 'seed' else 1)):
            raise ValueError('Invalid integer setting: '+key)
    for key in ('alpha','min_gain_retention','max_damage_rate'):
        if not 0 < policy[key] < 1:
            raise ValueError('Fraction must be strictly between zero and one: '+key)
    if not 0 < policy['min_delta'] < 1 or not 0 <= policy['cohort_noninferiority_margin'] < 1:
        raise ValueError('Invalid score thresholds')
    if policy['max_seconds_per_case'] <= 0 or policy['max_node_attempts'] > 2:
        raise ValueError('Positive cost budget and at most one technical retry allowed')
    return policy


def initial_state(plan_sha):
    return dict(schema='merit-chain-state-v1', plan_sha=plan_sha, node='A', candidate=None,
                history=[], attempts={}, forwards_used=0, holdout_consumed=False)


def transition(state, report, artifact_sha):
    """One whole-cohort report -> one transition; never read a partial score."""
    if state['node'] in TERMINAL:
        raise ValueError('Terminal chain cannot auto-resume, retune or switch candidate')
    node, verdict = state['node'], report['verdict']
    if verdict not in {'pass','fail','inconclusive','technical_failure'}:
        raise ValueError('Unknown verdict')
    out = {**state, 'history': [*state['history'], dict(node=node, verdict=verdict,
                                                      report_sha=artifact_sha)],
           'attempts': dict(state['attempts'])}
    if verdict == 'technical_failure':
        out['node'] = 'BLOCKED_TECHNICAL'
        out['blocked_node'] = node
    elif node == 'A':
        if verdict == 'pass':
            out.update(node='B', candidate=dict(algorithm='relay', layer=report['layer'], control='relay_roll'))
        else:
            out.update(node='C', candidate=dict(algorithm='project', layer=None, control='layout_average'))
    elif node in ('B','C'):
        if verdict == 'pass':
            out.update(node='D', holdout_consumed=True)
        elif node == 'B':
            out.update(node='C', candidate=dict(algorithm='project', layer=None, control='layout_average'))
        else:
            out['node'] = 'STOP_NO_CANDIDATE'
    elif node == 'D':
        out['node'] = {'pass':'READY_FOR_SCALE','fail':'STOP_FAILED_CONFIRMATION',
                       'inconclusive':'E'}[verdict]
    elif node == 'E':
        out['node'] = {'pass':'READY_FOR_SCALE','fail':'STOP_FAILED_CONFIRMATION',
                       'inconclusive':'STOP_INCONCLUSIVE'}[verdict]
    else:
        raise ValueError('Unknown experiment node')
    return out


def retry_technical(state, policy):
    """Explicit retry of SAME frozen algorithm/code only, with previous failure preserved."""
    if state['node'] != 'BLOCKED_TECHNICAL':
        raise ValueError('Only technical blockage can be retried')
    node = state['blocked_node']
    attempts = dict(state['attempts'])
    if attempts.get(node, 1) >= policy['max_node_attempts']:
        raise ValueError('Technical retry budget exhausted')
    attempts[node] = attempts.get(node, 1) + 1
    return dict(state, node=node, attempts=attempts)
