import math

import pytest

from merit_feddg.semantic_uncertainty import prefer_lower_uncertainty, uncertainty_scores


def samples(texts, logp=-1.):
    return [{'text':t, 'finished':True, 'mean_log_probability':logp} for t in texts]


class NLI:
    def check_implication(self, a, b):
        assert a.startswith('Question: organ?\nAnswer:')
        return 2 if a.split('Answer: ')[1] == b.split('Answer: ')[1] else 0


def test_semantic_entropy_analytic_cases_and_no_probability_claim():
    stable = uncertainty_scores(samples(['liver']*4), 'organ?', NLI())
    uncertain = uncertainty_scores(samples(['liver', 'spleen']*2), 'organ?', NLI())
    assert stable['scores']['discrete_semantic_entropy'] == 0
    assert uncertain['scores']['discrete_semantic_entropy'] == pytest.approx(math.log(2))
    assert uncertain['scores']['semantic_entropy'] == pytest.approx(math.log(2))
    assert uncertain['scores']['predictive_entropy'] == 1
    assert not uncertain['calibrated_probability']


def test_negation_not_merged_when_only_one_direction_entails():
    class Asymmetric:
        def check_implication(self, a, b):
            return 2 if a.split('Answer: ')[1] == 'liver' else 1
    result = uncertainty_scores(samples(['liver', 'not liver']), 'organ?', Asymmetric())
    assert result['semantic_ids'] == [0, 1]


def test_extreme_log_probabilities_are_stable_and_truncated_samples_not_scored():
    result = uncertainty_scores(samples(['a','b'], -10000), 'organ?', NLI())
    assert result['scores']['semantic_entropy'] == pytest.approx(math.log(2))
    truncated = samples(['a','b'])
    truncated[0]['finished'] = False
    assert not uncertainty_scores(truncated, 'organ?', NLI())['available']


def test_uncertainty_gate_selects_existing_answer_without_sampling_again():
    stable = uncertainty_scores(samples(['liver']*4), 'organ?', NLI())
    uncertain = uncertainty_scores(samples(['liver','spleen']*2), 'organ?', NLI())
    base = {'text':'spleen', 'token_ids':[1], 'seconds':1}
    candidate = {'text':'liver', 'token_ids':[2], 'seconds':1}
    output = prefer_lower_uncertainty(base, candidate, uncertain, stable, 'discrete_semantic_entropy')
    assert output['token_ids'] == [2]
    assert output['answer_arbitration']['decision'] == 'accept'
    unavailable = {'available':False, 'scores':{}}
    output = prefer_lower_uncertainty(base, candidate, unavailable, stable, 'semantic_entropy')
    assert output['answer_arbitration']['decision'] == 'abstain'


def test_llava_sampling_restores_rng_and_keeps_protocol_explicit(monkeypatch):
    from types import SimpleNamespace

    import torch

    from merit_feddg import llava_generalist as module
    probe = object.__new__(module.LlavaMedGeneralist)
    probe.torch = torch
    probe._inputs = lambda *a: {}
    probe._validate_context = lambda *a: None
    probe.tokenizer = SimpleNamespace(decode=lambda ids, **kw: str(ids[0]))
    options = []
    def generate(**kw):
        options.append(kw)
        return int(torch.randint(0, 10000, ()).item())
    probe.model = SimpleNamespace(generate=generate)
    monkeypatch.setattr(module, '_generated_rows', lambda result, *a: [((result,), -1., True)])
    before = torch.random.get_rng_state().clone()
    a = probe.sample_answers('image', 'prompt', count=3, max_new_tokens=64, seed=17)
    b = probe.sample_answers('image', 'prompt', count=3, max_new_tokens=64, seed=17)
    assert a == b and torch.equal(before, torch.random.get_rng_state())
    assert all(k['do_sample'] and k['temperature'] == 1 and k['top_k'] == 0 for k in options)
