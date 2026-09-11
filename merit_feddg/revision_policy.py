"""Experimental candidate-default policy; abstention is NOT a confidence claim.

Reuse the exact legacy verifier audit to isolate fallback policy from scoring,
generation, and expert selection. No fitted weights or reference answers.
"""
import copy


def presented_expert_ids(output):
    """Resolve actual final-context delivery, not merely acquired tool outputs."""
    for event in reversed(output.get('trace', [])):
        if event.get('event') == 'decode' and 'presented' in event.get('evidence_transport', {}):
            return {e['expert_id'] for e in event['evidence_transport']['presented']}
    # Legacy artifacts without transport proof conservatively keep every source.
    return {e['expert_id'] for e in output.get('evidence', [])}


def preserve_unverified_candidate(baseline, candidate, verified):
    audit = copy.deepcopy(verified['answer_arbitration'])
    if verified.get('candidate_answer', {}).get('text') != candidate['text']:
        raise ValueError('verifier audit does not belong to this candidate')
    if verified['candidate_answer'].get('token_ids') != candidate['token_ids']:
        raise ValueError('candidate token identity mismatch')
    decision = audit['decision']
    if decision not in {'accept', 'reject', 'abstain', 'unchanged'}:
        raise ValueError('unknown verification decision')
    # Explicit negative evidence can veto. Missing verification cannot establish
    # that the generalist is better. This default is an ablation, not a guarantee.
    chosen = baseline if decision == 'reject' else candidate
    output = copy.deepcopy(chosen)
    audit.update(policy='candidate_default_v1',
                 resolution='baseline' if decision == 'reject' else 'candidate',
                 candidate_verified=decision == 'accept',
                 fallback_changed=decision == 'abstain',
                 uncertainty_is_correctness_probability=False)
    output['answer_arbitration'] = audit
    output['candidate_answer'] = copy.deepcopy(verified['candidate_answer'])
    output['candidate_evidence'] = copy.deepcopy(candidate.get('evidence', []))
    output['seconds'] = verified['seconds']
    output['timing_includes_baseline_and_candidate'] = True
    output['verifier_audit_reused'] = True
    output.setdefault('trace', []).append({'event': 'answer_arbitration', **audit})
    return output
