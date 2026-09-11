"""Semantic entropy family from Kuhn et al. ICLR 2023 / Farquhar et al. Nature 2024.

Reference: jlko/semantic_uncertainty, a8d9aa8cecd5f3bec09b19ae38ab13552e0846f4,
uncertainty/uncertainty_measures/semantic_entropy.py. Independently implemented
formulae; no trained estimator, no fitted calibration, no correctness guarantee.
"""
import math
from collections import Counter

import numpy as np

METHODS = ('predictive_entropy', 'semantic_entropy', 'discrete_semantic_entropy')


def uncertainty_scores(samples, question, entailment):
    if len(samples) < 2:
        raise ValueError('at least two independent samples required')
    if any(not s.get('finished') or not s.get('text', '').strip() for s in samples):
        return {'available':False, 'reason':'empty_or_truncated_samples', 'scores':{}, 'samples':samples}
    logp = np.asarray([s['mean_log_probability'] for s in samples], dtype=float)
    if not np.isfinite(logp).all() or (logp > 0).any():
        raise ValueError('finite nonpositive mean token log probabilities required')
    # Strict bidirectional entailment; never merge on mere lack of contradiction.
    # Deterministic representative clustering without reassigning assigned items.
    ids, representatives = [], []
    nli_calls = 0
    for sample in samples:
        text = f"Question: {question}\nAnswer: {sample['text']}"
        group = None
        for i, representative in enumerate(representatives):
            if text == representative:
                group = i
                break
            forward = entailment.check_implication(representative, text)
            backward = entailment.check_implication(text, representative)
            nli_calls += 2
            if forward == backward == 2:
                group = i
                break
        if group is None:
            group = len(representatives)
            representatives.append(text)
        ids.append(group)
    counts = np.asarray(list(Counter(ids).values()), dtype=float) / len(ids)
    discrete = -float(np.sum(counts*np.log(counts)))
    # Reference logsumexp_by_id followed by predictive_entropy_rao.
    weights = np.exp(logp-logp.max())
    weights /= weights.sum()
    mass = np.bincount(ids, weights=weights)
    positive = mass[mass > 0]
    semantic = -float(np.sum(positive*np.log(positive)))
    return {'available':True, 'semantic_ids':ids, 'nli_calls':nli_calls,
        'scores':{'predictive_entropy':-float(logp.mean()), 'semantic_entropy':semantic,
                  'discrete_semantic_entropy':discrete},
        'samples':samples, 'length_normalized':True, 'calibrated_probability':False,
        'clustering':'strict_bidirectional_representative', 'question_bound':True}


class FrozenDebertaEntailment:
    """Paper's MNLI backend, from an explicitly prepared local checkpoint only."""
    def __init__(self, checkpoint, device='cpu'):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint, local_files_only=True).to(device).eval().requires_grad_(False)
        labels = {str(v).lower():int(k) for k, v in self.model.config.id2label.items()}
        if set(labels) != {'contradiction', 'neutral', 'entailment'}:
            raise ValueError('NLI checkpoint must declare contradiction/neutral/entailment labels')
        self.mapping = {labels[name]:i for i, name in enumerate(('contradiction', 'neutral', 'entailment'))}

    def check_implication(self, premise, hypothesis):
        inputs = self.tokenizer(premise, hypothesis, return_tensors='pt', truncation=False)
        if inputs['input_ids'].shape[1] > self.model.config.max_position_embeddings:
            raise ValueError('NLI context overflow; refusing prefix-only verification')
        with self.torch.inference_mode():
            logits = self.model(**inputs.to(self.model.device)).logits
        if not self.torch.isfinite(logits).all():
            raise ValueError('nonfinite NLI scores')
        return self.mapping[int(logits.argmax(-1).item())]


def prefer_lower_uncertainty(baseline, candidate, base_audit, candidate_audit, method):
    """Experimental arbitration USING a published estimator, not its paper result.

    Lower entropy chooses an existing answer; ties/unavailable estimates retain
    the ungated candidate. Do not interpret entropy difference as utility probability.
    """
    import copy
    if method not in METHODS:
        raise ValueError('unknown published uncertainty estimator')
    available = base_audit['available'] and candidate_audit['available']
    delta = None
    if available:
        delta = base_audit['scores'][method]-candidate_audit['scores'][method]
        if not math.isfinite(delta):
            raise ValueError('invalid uncertainty difference')
    decision = 'abstain' if delta is None or delta == 0 else 'accept' if delta > 0 else 'reject'
    chosen = baseline if decision == 'reject' else candidate
    result = copy.deepcopy(chosen)
    result['candidate_answer'] = {'text':candidate['text'], 'token_ids':candidate['token_ids']}
    result['candidate_evidence'] = copy.deepcopy(candidate.get('evidence', []))
    result['answer_arbitration'] = {'decision':decision, 'reason':'paired_uncertainty' if available else 'uncertainty_unavailable',
        'resolution':'baseline' if decision == 'reject' else 'candidate', 'method':method,
        'uncertainty_reduction':delta, 'calibrated_probability':False, 'trained':False,
        'baseline_uncertainty':base_audit, 'candidate_uncertainty':candidate_audit}
    result['seconds'] = baseline['seconds'] + candidate['seconds'] + base_audit.get('seconds',0) + candidate_audit.get('seconds',0)
    result['timing_includes_baseline_and_candidate'] = True
    return result


def estimate_answer_uncertainty(probe, row, prompt, arm, output, entailment, *, count, seed):
    """Rebuild the same semantic evidence context; never collect new experts."""
    from time import perf_counter

    from .capabilities import EvidenceItem
    from .capability_runtime import NativeSession, NativeState
    if arm.semantic_spatial:
        raise ValueError('stochastic spatial backend is not implemented; cannot silently drop masks')
    if not hasattr(probe, 'sample_answers'):
        raise ValueError('backend does not implement controlled independent sampling')
    start = perf_counter()
    session = NativeSession(probe, row['image'], prompt, row['question'], arm)
    image, context = session.context(NativeState(items=tuple(EvidenceItem(**e) for e in output.get('evidence', []))))
    samples = probe.sample_answers(image, context, count=count, max_new_tokens=arm.max_new_tokens, seed=seed)
    try:
        audit = uncertainty_scores(samples, row['question'], entailment)
    except (ValueError, RuntimeError) as exc:
        audit = {'available':False, 'reason':'nli_unavailable', 'error':str(exc), 'scores':{}, 'samples':samples}
    audit.update(seconds=perf_counter()-start, sample_count=count, seed=seed,
                 sampling_temperature=1., sampling_top_p=1., sampling_top_k=0,
                 transport=session.last_transport, extra_generation_calls=count)
    return audit
