"""Opt-in, training-free restoration of a receiver's visual attention pathway.

Uses native attention/value projections, never an attention-as-confidence score.
Only the last query at each decode step is patched. The expert's prompt remains
intact. Unsupported kernels fail closed; no global monkey-patching or training.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import inspect
from typing import Literal

import torch
import torch.nn.functional as F

Mode = Literal['off', 'audit', 'restore_mass', 'restore_vector']


class PathwayError(RuntimeError):
    """Invalid or unsupported intervention; do not count as a baseline fallback."""


@dataclass(frozen=True)
class VisualSpan:
    start: int
    stop: int

    def __post_init__(self):
        if type(self.start) is not int or type(self.stop) is not int or not 0 <= self.start < self.stop:
            raise ValueError('Require a nonempty, nonnegative, half-open visual span')


@dataclass(frozen=True)
class PathwayConfig:
    mode: Mode = 'off'
    layers: tuple[int, ...] = (-1,)

    def __post_init__(self):
        if self.mode not in ('off', 'audit', 'restore_mass', 'restore_vector'):
            raise ValueError('Unknown pathway mode')
        if not self.layers or any(type(i) is not int for i in self.layers):
            raise ValueError('Declare decoder layer indices explicitly')


@dataclass(frozen=True)
class VisualContribution:
    vector: torch.Tensor  # [batch, query_heads, head_dim], float32
    mass: torch.Tensor    # [batch, query_heads, 1], float32


def visual_contribution(weights, visual_values, span):
    """Native last-query A_V @ V, including GQA/MQA and value-projection bias.

    weights: [B, query_heads, Q, K]; values: [B, kv_heads, V, head_dim].
    Image positions refer to expanded decoder embeddings, not raw image tokens.
    """
    if weights is None or weights.ndim != 4 or visual_values.ndim != 4:
        raise PathwayError('Native attention weights and projected visual values required')
    b, heads, queries, keys = weights.shape
    vb, kv_heads, nv, _ = visual_values.shape
    if not queries or b != vb or kv_heads < 1 or heads % kv_heads or nv != span.stop - span.start:
        raise PathwayError('Invalid native GQA/visual-value layout')
    if span.stop > keys:
        raise PathwayError('Visual span exceeds actual attention keys')
    a = weights[:, :, -1:, span.start:span.stop].float()
    v = visual_values.repeat_interleave(heads // kv_heads, dim=1).float()
    if not torch.isfinite(a).all() or not torch.isfinite(v).all() or (a < 0).any():
        raise PathwayError('Non-finite/negative native visual attention')
    mass = a.sum(-1)
    if (mass > 1.01).any():
        raise PathwayError('Attention is not a normalized evaluation-time probability')
    return VisualContribution((a @ v).squeeze(-2), mass)


def restoration_delta(receiver, reference, mode):
    """Return an activation difference, not a correctness estimate.

    Mass restoration retains the receiver's within-image attention direction.
    A positive reference mass with zero receiver mass has no defined direction;
    raise rather than introduce an arbitrary epsilon-based direction.
    """
    if receiver.vector.shape != reference.vector.shape or receiver.mass.shape != reference.mass.shape:
        raise PathwayError('Reference/receiver head layouts differ')
    if mode in ('off', 'audit'):
        return torch.zeros_like(receiver.vector)
    if mode == 'restore_vector':
        result = reference.vector - receiver.vector
    elif mode == 'restore_mass':
        zero = receiver.mass == 0
        if (zero & (reference.mass > 0)).any():
            raise PathwayError('Cannot restore positive mass onto an undefined zero-mass direction')
        denominator = torch.where(zero, torch.ones_like(receiver.mass), receiver.mass)
        result = receiver.vector * (reference.mass / denominator - 1)
    else:
        raise ValueError('Unknown pathway mode')
    if not torch.isfinite(result).all():
        raise PathwayError('Non-finite restoration delta')
    return result


class PathwayRestorer:
    """Scoped hooks on selected native Llama/Mistral/Qwen2-family attention.

    Caller supplies a reference pass before a receiver pass, with the SAME
    generated-token prefix. This class verifies step order/key lengths; the
    paired decoder below verifies prefix ownership. Only visual values are
    retained from prefill, not a second copy of all projected values.
    """

    def __init__(self, model, config):
        self.model, self.config = model, config
        decoder = model.get_model() if hasattr(model, 'get_model') else model.model
        self.layers = decoder.layers
        indices = tuple(i if i >= 0 else len(self.layers) + i for i in config.layers)
        if len(set(indices)) != len(indices) or any(not 0 <= i < len(self.layers) for i in indices):
            raise ValueError('Duplicate or out-of-range decoder layers')
        self.indices = indices
        self.handles, self.events = [], []
        self.active = None
        self.references, self.values, self.next_steps = {}, {}, {}
        self.entered = False

    def __enter__(self):
        if self.entered:
            raise PathwayError('Controller is not reentrant')
        if self.model.training or any(p.requires_grad for p in self.model.parameters()):
            raise PathwayError('Require eval() and frozen weights before intervention')
        if getattr(self.model, '_merit_pathway_active', False):
            raise PathwayError('Concurrent pathway controllers on one model are unsupported')
        self.entered = True
        self.model._merit_pathway_active = True
        try:
            if self.config.mode != 'off':
                for index in self.indices:
                    module = self.layers[index].self_attn
                    if module.training:
                        raise PathwayError('Selected attention module must be in evaluation mode')
                    if 'flash' in type(module).__name__.lower():
                        raise PathwayError('FlashAttention cannot expose native weights; use a separately audited eager load')
                    if not isinstance(module.v_proj, torch.nn.Linear) or not isinstance(module.o_proj, torch.nn.Linear):
                        raise PathwayError('Only ordinary separate Linear V/O projections are supported')
                    if getattr(module.config, 'pretraining_tp', 1) != 1:
                        raise PathwayError('Tensor-parallel projection bypasses are not supported')
                    signature = inspect.signature(module.forward)
                    if 'output_attentions' not in signature.parameters:
                        raise PathwayError('Native module has no explicit output_attentions argument')
                    self.handles.append(module.register_forward_pre_hook(
                        self._enable_attention(signature), with_kwargs=True))
                    self.handles.append(module.v_proj.register_forward_hook(self._capture_values(index, module)))
                    self.handles.append(module.register_forward_hook(self._finish_attention(index, module)))
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.references.clear()
        self.values.clear()
        self.next_steps.clear()
        self.active = None
        if self.entered:
            del self.model._merit_pathway_active
            self.entered = False

    def _enable_attention(self, signature):
        def hook(module, args, kwargs):
            if self.active is None:
                return None
            bound = signature.bind_partial(*args, **kwargs)
            bound.arguments['output_attentions'] = True
            return bound.args, bound.kwargs
        return hook

    @contextmanager
    def pass_context(self, branch, step, prompt_length, span):
        if not self.entered or self.active is not None:
            raise PathwayError('Open exactly one controller/pass context')
        if branch not in ('reference', 'receiver') or step != self.next_steps.get(branch, 0):
            raise PathwayError('Out-of-order branch step or stale cache')
        if type(step) is not int or step < 0 or span.stop >= prompt_length:
            raise PathwayError('Need a prompt query after the visual span')
        if branch == 'receiver' and self.config.mode != 'off':
            if any(self.references.get(i, (None,))[0] != step for i in self.indices):
                raise PathwayError('Receiver requires same-step reference contributions')
        self.active = dict(branch=branch, step=step, length=prompt_length, span=span, seen=set())
        try:
            yield
            if self.config.mode != 'off' and self.active['seen'] != set(self.indices):
                raise PathwayError('Selected attention modules were not all executed')
            self.next_steps[branch] = step + 1
        finally:
            self.active = None

    def _capture_values(self, index, module):
        def hook(projection, args, output):
            state = self.active
            if state is None:
                return
            kv = module.config.num_key_value_heads
            heads = module.config.num_attention_heads
            dim = module.q_proj.out_features // heads
            if output.ndim != 3 or output.shape[0] != 1 or output.shape[-1] != kv * dim:
                raise PathwayError('Only batch-one standard GQA value projections supported')
            key = (state['branch'], index)
            if state['step'] == 0:
                if output.shape[1] != state['length']:
                    raise PathwayError('Prefill values do not match expanded prompt')
                span = state['span']
                self.values[key] = output[:, span.start:span.stop].detach().reshape(
                    1, span.stop - span.start, kv, dim).transpose(1, 2).contiguous()
            elif output.shape[1] != 1 or key not in self.values:
                raise PathwayError('Incremental decode requires a private prefilled KV cache')
        return hook

    def _finish_attention(self, index, module):
        def hook(attention, args, output):
            state = self.active
            if state is None:
                return None
            if index in state['seen'] or not isinstance(output, tuple) or len(output) < 2:
                raise PathwayError('Unexpected native attention return/call count')
            state['seen'].add(index)
            weights = output[1]
            qlen = state['length'] if state['step'] == 0 else 1
            if weights is None or weights.shape[-2:] != (qlen, state['length'] + state['step']):
                raise PathwayError('Missing/full-cache attention required; sliding eviction is unsupported')
            current = visual_contribution(weights, self.values[(state['branch'], index)], state['span'])
            event = dict(branch=state['branch'], step=state['step'], layer=index,
                         visual_mass=float(current.mass.mean()), delta_norm=0.0, patched=False)
            if state['branch'] == 'reference':
                self.references[index] = (state['step'], current)
                self.events.append(event)
                return None
            reference = self.references[index][1]
            delta = restoration_delta(current, reference, self.config.mode)
            event['reference_visual_mass'] = float(reference.mass.mean())
            event['delta_norm'] = float(torch.linalg.vector_norm(delta))
            self.events.append(event)
            if not torch.count_nonzero(delta):
                return None  # Exact identity; do not introduce rounding through +/- zero.
            # Project ONLY the delta. Adding O's bias here would incorrectly add it twice.
            update = F.linear(delta.reshape(1, 1, -1).to(module.o_proj.weight.dtype),
                              module.o_proj.weight, bias=None).to(output[0].dtype)
            if not torch.isfinite(update).all():
                raise PathwayError('Non-finite projected update')
            patched = output[0].clone()
            patched[:, -1:, :] += update
            event['patched'] = True
            return (patched, *output[1:])
        return hook
