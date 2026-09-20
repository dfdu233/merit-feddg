"""CPU algebra/cache tests. Synthetic random weights are NOT medical results."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from merit_feddg.pathway_restore import (
    PathwayConfig, PathwayError, PathwayRestorer, VisualContribution,
    VisualSpan, restoration_delta, visual_contribution,
)
from merit_feddg.huatuo_pathway import GreedySpec, PreparedPrompt, paired_generate


def test_native_bos_processor_prefix_matches_transformers():
    from transformers import RepetitionPenaltyLogitsProcessor, MinNewTokensLengthLogitsProcessor
    spec = GreedySpec(10, (0,), 1.2, 1, (0,))
    logits = torch.tensor([6., 5.5, -2., 4.])
    for prefix in ([], [1], [1, 3]):
        ids = torch.tensor([[0] + prefix])
        scores = RepetitionPenaltyLogitsProcessor(1.2)(ids, logits[None].clone())
        scores = MinNewTokensLengthLogitsProcessor(1, 1, 0)(ids, scores)
        assert spec.select(logits, prefix) == int(scores.argmax())


class TinyAttention(nn.Module):
    def __init__(self, width=8, heads=4, kv_heads=2):
        super().__init__()
        self.config = SimpleNamespace(num_attention_heads=heads, num_key_value_heads=kv_heads,
                                      pretraining_tp=1)
        self.q_proj = nn.Linear(width, width, bias=True)
        self.k_proj = nn.Linear(width, width // heads * kv_heads, bias=True)
        self.v_proj = nn.Linear(width, width // heads * kv_heads, bias=True)
        self.o_proj = nn.Linear(width, width, bias=True)

    def forward(self, hidden_states, past_key_value=None, output_attentions=False):
        b, n, width = hidden_states.shape
        h, kv = self.config.num_attention_heads, self.config.num_key_value_heads
        d = width // h
        q = self.q_proj(hidden_states).view(b, n, h, d).transpose(1, 2)
        k = self.k_proj(hidden_states).view(b, n, kv, d).transpose(1, 2)
        v = self.v_proj(hidden_states).view(b, n, kv, d).transpose(1, 2)
        if past_key_value is not None:
            k, v = torch.cat((past_key_value[0], k), 2), torch.cat((past_key_value[1], v), 2)
        keys = k.shape[2]
        score = q @ k.repeat_interleave(h // kv, 1).transpose(-2, -1) / d**0.5
        allowed = torch.arange(keys)[None, :] <= (torch.arange(n)[:, None] + keys - n)
        weights = score.masked_fill(~allowed.to(score.device), -torch.inf).softmax(-1)
        value = weights @ v.repeat_interleave(h // kv, 1)
        result = self.o_proj(value.transpose(1, 2).reshape(b, n, width))
        return result, weights if output_attentions else None, (k, v)


class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(17, 8)
        self.layers = nn.ModuleList([nn.ModuleDict({'attention': TinyAttention()}) for _ in range(2)])
        # Match the native decoder interface without registering duplicate modules.
        self.decoder_view = SimpleNamespace(layers=[SimpleNamespace(self_attn=x['attention']) for x in self.layers])
        self.lm_head = nn.Linear(8, 17, bias=False)
        self.calls = []

    def get_model(self):
        return self.decoder_view

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None,
                position_ids=None, past_key_values=None, use_cache=True,
                output_attentions=False, return_dict=True):
        hidden = self.embedding(input_ids) if inputs_embeds is None else inputs_embeds
        caches = []
        self.calls.append(None if input_ids is None else input_ids.tolist())
        for i, layer in enumerate(self.layers):
            output, _, cache = layer['attention'](hidden,
                past_key_value=None if past_key_values is None else past_key_values[i],
                output_attentions=output_attentions)
            hidden = hidden + output
            caches.append(cache)
        return SimpleNamespace(logits=self.lm_head(hidden), past_key_values=tuple(caches))


@pytest.fixture
def model():
    torch.manual_seed(19)
    return TinyLM().eval().requires_grad_(False)


def prepared(model, n, seed):
    rng = torch.Generator().manual_seed(seed)
    x = torch.randn(1, n, 8, generator=rng)
    return PreparedPrompt(x, torch.ones(1, n, dtype=torch.long),
                          torch.arange(n).unsqueeze(0), VisualSpan(1, 3))


@pytest.mark.parametrize('kv', [1, 2, 4])
def test_visual_algebra_gqa_and_mqa(kv):
    torch.manual_seed(9)
    a = torch.randn(1, 4, 5, 7).softmax(-1)
    v = torch.randn(1, kv, 3, 2)
    result = visual_contribution(a, v, VisualSpan(2, 5))
    expected = (a[:, :, -1:, 2:5] @ v.repeat_interleave(4 // kv, 1)).squeeze(-2)
    torch.testing.assert_close(result.vector, expected)
    torch.testing.assert_close(result.mass, a[:, :, -1, 2:5].sum(-1, keepdim=True))


@pytest.mark.parametrize('mode', ['restore_mass', 'restore_vector'])
def test_identity_delta(mode):
    part = VisualContribution(torch.ones(1, 4, 2), torch.full((1, 4, 1), .2))
    assert torch.count_nonzero(restoration_delta(part, part, mode)) == 0


def test_mass_keeps_receiver_direction():
    receiver = VisualContribution(torch.tensor([[[2., 4.]]]), torch.tensor([[[.2]]]))
    reference = VisualContribution(torch.tensor([[[99., -2.]]]), torch.tensor([[[.3]]]))
    delta = restoration_delta(receiver, reference, 'restore_mass')
    torch.testing.assert_close(receiver.vector + delta, torch.tensor([[[3., 6.]]]))


def test_mass_zero_is_not_silently_invented():
    a = VisualContribution(torch.zeros(1, 1, 2), torch.zeros(1, 1, 1))
    b = VisualContribution(torch.ones(1, 1, 2), torch.ones(1, 1, 1))
    with pytest.raises(PathwayError, match='undefined'):
        restoration_delta(a, b, 'restore_mass')
    assert torch.count_nonzero(restoration_delta(a, a, 'restore_mass')) == 0


@pytest.mark.parametrize('span', [(-1, 3), (3, 3), (5, 2)])
def test_invalid_span(span):
    with pytest.raises(ValueError):
        VisualSpan(*span)


@pytest.mark.parametrize('layers', [(2,), (-3,), (-1, 1)])
def test_invalid_layers(model, layers):
    with pytest.raises(ValueError):
        PathwayRestorer(model, PathwayConfig('audit', layers))


def test_audit_matches_unhooked_greedy_and_private_cache(model):
    r, e = prepared(model, 5, 1), prepared(model, 8, 2)
    spec = GreedySpec(6, (16,), 1.2, 1)
    original = paired_generate(model, r, e, spec, PathwayConfig('off'), context_limit=30)
    model.calls.clear()
    audit = paired_generate(model, r, e, spec, PathwayConfig('audit'), context_limit=30)
    assert original['token_ids'] == audit['token_ids']
    # Each incremental reference/receiver pair consumes exactly the same token.
    assert all(model.calls[i] == model.calls[i+1] for i in range(0, len(model.calls), 2))
    assert not any(event['patched'] for event in audit['events'])
    after = paired_generate(model, r, e, spec, PathwayConfig('off'), context_limit=30)
    assert after['token_ids'] == original['token_ids']


@pytest.mark.parametrize('mode', ['audit', 'restore_mass', 'restore_vector'])
def test_same_input_fast_path(model, mode):
    r = prepared(model, 5, 1)
    value = paired_generate(model, r, r, GreedySpec(3, (16,)), PathwayConfig(mode), context_limit=20)
    assert value['effective_mode'] == 'off'
    assert value['reference_forwards'] == 0 and value['events'] == []


@pytest.mark.parametrize('mode', ['restore_mass', 'restore_vector'])
def test_real_forward_intervention_without_weight_updates(model, mode):
    old = {k: v.clone() for k, v in model.state_dict().items()}
    r, e = prepared(model, 5, 1), prepared(model, 8, 2)
    result = paired_generate(model, r, e, GreedySpec(4, (16,)),
                             PathwayConfig(mode, (0, 1)), context_limit=20)
    assert any(x['patched'] for x in result['events'])
    assert not result['medical_correctness_estimated'] and not result['weights_updated']
    for k, v in model.state_dict().items():
        assert torch.equal(v, old[k])
    assert all(not layer['attention']._forward_hooks for layer in model.layers)


def test_patch_only_last_query_and_do_not_double_output_bias(model):
    attention = model.layers[0]['attention']
    base, evidence = torch.randn(1, 5, 8), torch.randn(1, 7, 8)
    with torch.inference_mode():
        clean = attention(base, output_attentions=True)
        native = attention(evidence, output_attentions=True)
        def part(x, output):
            values = attention.v_proj(x)[:, 1:3].reshape(1, 2, 2, 2).transpose(1, 2)
            return visual_contribution(output[1], values, VisualSpan(1, 3))
        delta = (part(base, clean).vector - part(evidence, native).vector).reshape(1, 1, -1)
        expected = native[0].clone()
        expected[:, -1:] += torch.nn.functional.linear(delta, attention.o_proj.weight, None)
        with PathwayRestorer(model, PathwayConfig('restore_vector', (0,))) as restorer:
            with restorer.pass_context('reference', 0, 5, VisualSpan(1, 3)):
                attention(base)
            with restorer.pass_context('receiver', 0, 7, VisualSpan(1, 3)):
                restored = attention(evidence)
        torch.testing.assert_close(restored[0], expected)
        assert torch.equal(restored[0][:, :-1], native[0][:, :-1])
        assert not torch.allclose(restored[0][:, -1], clean[0][:, -1])  # Expert/nonvisual path retained.


def test_exception_removes_all_hooks(model):
    with pytest.raises(RuntimeError, match='deliberate'):
        with PathwayRestorer(model, PathwayConfig('audit')):
            raise RuntimeError('deliberate')
    assert not hasattr(model, '_merit_pathway_active')
    for layer in model.layers:
        attention = layer['attention']
        assert not attention._forward_hooks and not attention._forward_pre_hooks
        assert not attention.v_proj._forward_hooks


def test_stale_or_missing_reference_rejected(model):
    with PathwayRestorer(model, PathwayConfig('restore_vector')) as restorer:
        with pytest.raises(PathwayError, match='same-step'):
            with restorer.pass_context('receiver', 0, 5, VisualSpan(1, 3)):
                pass
        with pytest.raises(PathwayError, match='Out-of-order'):
            with restorer.pass_context('reference', 1, 5, VisualSpan(1, 3)):
                pass


def test_native_cache_matches_full_prefix_off(model):
    r, e = prepared(model, 5, 1), prepared(model, 8, 2)
    spec = GreedySpec(5, (16,), 1.1)
    cached = paired_generate(model, r, e, spec, PathwayConfig('off'), context_limit=20)
    tokens = []
    with torch.inference_mode():
        for _ in range(5):
            extra = model.embedding(torch.tensor([tokens], dtype=torch.long))
            output = model(inputs_embeds=torch.cat((e.embeds, extra), 1))
            token = spec.select(output.logits[0, -1], tokens)
            tokens.append(token)
            if token == 16:
                break
    assert tokens == cached['token_ids']


def test_context_overflow_does_not_run_model(model):
    r, e = prepared(model, 5, 1), prepared(model, 8, 2)
    with pytest.raises(PathwayError, match='reserved'):
        paired_generate(model, r, e, GreedySpec(5, (16,)), context_limit=10)
    assert model.calls == []


def test_train_mode_rejected(model):
    model.train()
    with pytest.raises(PathwayError, match='eval'):
        with PathwayRestorer(model, PathwayConfig('audit')):
            pass


def test_minimum_length_and_repetition():
    spec = GreedySpec(4, (2,), 2., 1)
    assert spec.select(torch.tensor([1., 3., 10.]), []) == 1
    assert spec.select(torch.tensor([3., 4., 0.]), [1]) == 0


def test_bad_attention_weights():
    with pytest.raises(PathwayError):
        visual_contribution(None, torch.ones(1, 1, 2, 2), VisualSpan(0, 2))
    with pytest.raises(PathwayError):
        visual_contribution(torch.full((1, 1, 1, 2), float('nan')),
                            torch.ones(1, 1, 2, 2), VisualSpan(0, 2))


def test_flash_kernel_rejected_without_hook_leak(model):
    class FakeFlashAttention(TinyAttention):
        pass
    replacement = FakeFlashAttention().eval().requires_grad_(False)
    model.decoder_view.layers[1].self_attn = replacement
    with pytest.raises(PathwayError, match='FlashAttention'):
        with PathwayRestorer(model, PathwayConfig('audit')):
            pass
    assert not hasattr(model, '_merit_pathway_active')
    assert not replacement._forward_hooks


def test_missing_native_weights_is_not_fallback(model):
    class NoWeights(TinyAttention):
        def forward(self, hidden_states, past_key_value=None, output_attentions=False):
            return super().forward(hidden_states, past_key_value, False)
    attention = NoWeights().eval().requires_grad_(False)
    model.decoder_view.layers[0].self_attn = attention
    with pytest.raises(PathwayError, match='full-cache attention'):
        with PathwayRestorer(model, PathwayConfig('audit', (0,))) as restorer:
            with restorer.pass_context('reference', 0, 5, VisualSpan(1, 3)):
                attention(torch.randn(1, 5, 8))
    assert not attention._forward_hooks and not attention.v_proj._forward_hooks


def test_controller_is_not_concurrent(model):
    with PathwayRestorer(model, PathwayConfig('audit')):
        with pytest.raises(PathwayError, match='Concurrent'):
            with PathwayRestorer(model, PathwayConfig('audit')):
                pass
        assert model._merit_pathway_active
    assert not hasattr(model, '_merit_pathway_active')


def test_eos_outside_vocabulary_is_explicit():
    with pytest.raises(PathwayError, match='vocabulary'):
        GreedySpec(3, (99,)).select(torch.ones(5), [])
