"""Training-free experimental operators, not medical correctness estimators.

A: first-decision divergence and two predeclared residual interventions.
B: mid-layer last-query restoration; later layers can re-read expert context.
C: Fisher-weighted projection of layout-sensitive context log-prob residuals.
All decoding streams own their KV cache and consume identical committed IDs.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
import time
from typing import Sequence

import torch

from ..huatuo_pathway import PreparedPrompt, _Stream, _sync
from ..pathway_restore import PathwayError


@dataclass(frozen=True)
class DecodePolicy:
    max_new_tokens: int
    eos_token_ids: tuple[int, ...]
    repetition_penalty: float = 1.0
    min_new_tokens: int = 0
    processor_prefix: tuple[int, ...] = ()

    def __post_init__(self):
        if type(self.max_new_tokens) is not int or self.max_new_tokens < 1:
            raise ValueError('Explicit positive generation budget required')
        if type(self.min_new_tokens) is not int or not 0 <= self.min_new_tokens < self.max_new_tokens:
            raise ValueError('Invalid min_new_tokens')
        if not math.isfinite(self.repetition_penalty) or self.repetition_penalty <= 0:
            raise ValueError('Invalid repetition penalty')
        if not self.eos_token_ids or any(type(t) is not int or t < 0
                                        for t in (*self.eos_token_ids, *self.processor_prefix)):
            raise ValueError('Invalid explicit special tokens')

    def scores(self, logits, prefix):
        scores = logits.float().clone()
        if scores.ndim != 1 or not torch.isfinite(scores).all():
            raise PathwayError('Finite next-token logits required')
        ids = (*self.processor_prefix, *prefix)
        if any(type(t) is not int or not 0 <= t < scores.numel() for t in (*ids, *self.eos_token_ids)):
            raise PathwayError('Token outside vocabulary')
        if ids and self.repetition_penalty != 1:
            index = torch.tensor(sorted(set(ids)), device=scores.device)
            old = scores[index]
            scores[index] = torch.where(old < 0, old * self.repetition_penalty,
                                        old / self.repetition_penalty)
        if len(prefix) < self.min_new_tokens:
            scores[list(self.eos_token_ids)] = -torch.inf
        if not torch.isfinite(scores).any():
            raise PathwayError('All tokens suppressed')
        return scores

    def select(self, logits, prefix):
        return int(self.scores(logits, prefix).argmax())


def decoder_layers(model):
    core = model.get_model() if hasattr(model, 'get_model') else model.model
    return core.layers


def cut_layers(model):
    """Two fixed relative-depth sites, not a search over every layer/head."""
    count = len(decoder_layers(model))
    if count < 3:
        raise PathwayError('At least three decoder blocks required')
    return tuple(dict.fromkeys((max(0, count // 3 - 1), max(0, 2 * count // 3 - 1))))


def check_inputs(model, prompts, policy, context_limit):
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise PathwayError('Require eval and frozen weights')
    if getattr(model, '_merit_pathway_active', False) or getattr(model, '_merit_chain_active', False):
        raise PathwayError('Concurrent controllers unsupported')
    first = prompts[0]
    first.validate()
    for p in prompts:
        p.validate()
        if p.length + policy.max_new_tokens > context_limit:
            raise PathwayError('Full answer reserve exceeds KV context limit')
        a, b = first.visual_span, p.visual_span
        if not torch.equal(first.embeds[:, a.start:a.stop], p.embeds[:, b.start:b.stop]):
            raise PathwayError('Every branch must preserve identical image embeddings')


def _last(output):
    hidden = output[0] if isinstance(output, tuple) else output
    if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3 or hidden.shape[0] != 1:
        raise PathwayError('Expected native batch-one block hidden states')
    return hidden


class ResidualRelay:
    """Scoped native block-output hook; no global attention/kernel replacement.

    Only the last query is restored. This is the whole residual at ONE middle
    boundary, not a pure visual component. Expert prompt-token KV and later
    blocks remain expert-conditioned. Roll control preserves pre-addition norm
    but changes coordinate direction; it is not a medical negative label.
    """
    def __init__(self, model, layers):
        self.model, self.layers = model, tuple(layers)
        if not self.layers or len(set(self.layers)) != len(self.layers):
            raise ValueError('Unique nonempty sites required')
        if any(type(i) is not int or not 0 <= i < len(decoder_layers(model)) for i in self.layers):
            raise ValueError('Invalid block index')
        self.handles, self.references, self.events = [], {}, []
        self.mode = None

    def __enter__(self):
        if self.model.training or any(p.requires_grad for p in self.model.parameters()):
            raise PathwayError('Frozen evaluation model required')
        if getattr(self.model, '_merit_chain_active', False) or getattr(self.model, '_merit_pathway_active', False):
            raise PathwayError('Concurrent controllers unsupported')
        self.model._merit_chain_active = True
        try:
            for i in self.layers:
                self.handles.append(decoder_layers(self.model)[i].register_forward_hook(self._hook(i)))
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles.clear()
        self.references.clear()
        self.mode = None
        if getattr(self.model, '_merit_chain_active', False):
            del self.model._merit_chain_active

    @contextmanager
    def use(self, mode, step, layer=None):
        if self.mode is not None or mode not in ('reference', 'audit', 'restore', 'roll'):
            raise PathwayError('Invalid/nested relay pass')
        if mode in ('restore', 'roll') and self.references.get(layer, (None,))[0] != step:
            raise PathwayError('Missing same-step reference')
        self.mode, self.step, self.selected = mode, step, layer
        self.seen = set()
        try:
            yield
            if self.seen != set(self.layers):
                raise PathwayError('Native hooks not all exercised')
        finally:
            self.mode = None

    def _hook(self, layer):
        def hook(module, args, output):
            if self.mode is None:
                return None
            if layer in self.seen:
                raise PathwayError('Repeated block invocation')
            self.seen.add(layer)
            hidden = _last(output)
            if self.mode == 'reference':
                self.references[layer] = (self.step, hidden[:, -1:].detach().clone())
                return None
            if self.mode == 'audit' or layer != self.selected:
                return None
            ref = self.references[layer][1]
            delta = ref.float() - hidden[:, -1:].float()
            if self.mode == 'roll':
                delta = delta.roll(shifts=max(1, delta.shape[-1] // 2), dims=-1)
            norm = float(delta.norm())
            if not math.isfinite(norm):
                raise PathwayError('Nonfinite residual intervention')
            updated = hidden.clone()
            # Direct replacement is numerically exact for restore, rather than h+(ref-h).
            updated[:, -1:] = ref if self.mode == 'restore' else (hidden[:, -1:].float() + delta).to(hidden.dtype)
            actual = float((updated[:, -1:].float() - hidden[:, -1:].float()).norm())
            self.events.append(dict(step=self.step, layer=layer, mode=self.mode,
                                    proposed_norm=norm, applied_norm=actual, patched=actual > 0))
            if not torch.isfinite(updated).all():
                raise PathwayError('Nonfinite block output')
            return (updated, *output[1:]) if isinstance(output, tuple) else updated
        return hook


def _full_logits(model, prepared, prefix):
    core = model.get_model() if hasattr(model, 'get_model') else model.model
    embedding = core.embed_tokens if hasattr(core, 'embed_tokens') else model.get_input_embeddings()
    extra = embedding(torch.tensor([prefix], device=prepared.embeds.device, dtype=torch.long))
    embeds = torch.cat((prepared.embeds, extra), dim=1)
    positions = torch.arange(embeds.shape[1], device=embeds.device).unsqueeze(0)
    out = model(inputs_embeds=embeds, position_ids=positions,
                attention_mask=torch.ones(embeds.shape[:2], device=embeds.device, dtype=prepared.mask.dtype),
                use_cache=False, output_attentions=False, return_dict=True)
    return out.logits[0, -1]


def _contrast(logits, policy, prefix, a, b):
    raw = float(logits[a].float() - logits[b].float())
    scores = policy.scores(logits, prefix)
    # EOS can be suppressed, but a/b were selected under the same processors.
    if not torch.isfinite(scores[[a, b]]).all():
        raise PathwayError('Compared tokens must be allowed under frozen generation processors')
    p = scores.softmax(-1)
    return dict(raw_margin=raw, processed_margin=float(scores[a] - scores[b]),
                probability_margin=float(p[a] - p[b]))


def diagnose(model, reference, receiver, policy, *, context_limit, max_prefix=16):
    """A. Locate actual first greedy disagreement, then re-prefill at that prefix.

    No labels or reference answers enter this routine. Full-prefix replay must
    preserve incremental greedy decisions; logit differences are reported.
    Clinical good/bad orientation happens ONLY in the separate evaluator.
    """
    check_inputs(model, [reference, receiver], policy, context_limit)
    if type(max_prefix) is not int or max_prefix < 1:
        raise ValueError('Positive fixed diagnostic prefix budget required')
    layers = cut_layers(model)
    streams = [_Stream(model, p) for p in (reference, receiver)]
    prefix, forwards = [], 0
    _sync(receiver.embeds.device)
    started = time.perf_counter()
    result = dict(status='no_divergence', sites=[], prefix_ids=[], forwards=0)
    with torch.inference_mode():
        for step in range(min(max_prefix, policy.max_new_tokens)):
            z0, ze = [s.advance(prefix) for s in streams]
            forwards += 2
            if streams[0].cache is streams[1].cache:
                raise PathwayError('Shared KV cache')
            a, b = policy.select(z0, prefix), policy.select(ze, prefix)
            if a != b:
                result.update(status='divergence', a=a, b=b, prefix_ids=list(prefix), step=step)
                with ResidualRelay(model, layers) as relay:
                    with relay.use('reference', step):
                        fresh0 = _full_logits(model, reference, prefix)
                    with relay.use('audit', step):
                        freshE = _full_logits(model, receiver, prefix)
                    forwards += 2
                    if policy.select(fresh0, prefix) != a or policy.select(freshE, prefix) != b:
                        raise PathwayError('Full-prefix and cached decision parity failed')
                    result['baseline'] = _contrast(fresh0, policy, prefix, a, b)
                    result['expert'] = _contrast(freshE, policy, prefix, a, b)
                    # Report numerical differences; equality of greedy tokens is not logit equality.
                    result['cache_logit_max_abs'] = [float((fresh0.float()-z0.float()).abs().max()),
                                                     float((freshE.float()-ze.float()).abs().max())]
                    for layer in layers:
                        records = {}
                        for mode in ('restore', 'roll'):
                            with relay.use(mode, step, layer):
                                logits = _full_logits(model, receiver, prefix)
                            forwards += 1
                            records[mode] = _contrast(logits, policy, prefix, a, b)
                            records[mode]['token'] = policy.select(logits, prefix)
                        result['sites'].append(dict(layer=layer, **records))
                    result['events'] = list(relay.events)
                break
            prefix.append(a)
            if a in policy.eos_token_ids:
                result['status'] = 'same_answer'
                break
    _sync(receiver.embeds.device)
    result.update(forwards=forwards, seconds=time.perf_counter()-started,
                  labels_used=False, prefix_budget=max_prefix, tested_layers=list(layers))
    return result


def project_residual(base_logits, expert_logits, control_logits: Sequence[torch.Tensor], *, rtol=1e-6):
    """C. Remove measured nuisance span in the p0-weighted log-prob geometry.

    Controls MUST contain the same native evidence, not false/synthetic medical
    evidence. rtol is numerical rank tolerance, never a correctness threshold.
    A direction stable across layouts may still be wrong. No correctness claim.
    """
    if len(control_logits) != 2 or not 0 < rtol < 1:
        raise ValueError('Exactly two predeclared controls and valid numeric rtol required')
    values = [base_logits, expert_logits, *control_logits]
    if any(z.ndim != 1 or z.shape != base_logits.shape or not torch.isfinite(z).all() for z in values):
        raise PathwayError('Finite aligned vocabulary logits required')
    # Small Gram solve in float64; full-vocabulary log probabilities stay float32.
    logs = [z.float().log_softmax(-1) for z in values]
    weight = logs[0].exp().double()
    r = (logs[1] - logs[0]).double()
    r -= (weight * r).sum()
    nuisance = torch.stack([(v - logs[1]).double() for v in logs[2:]], dim=-1)
    nuisance -= (weight[:, None] * nuisance).sum(0, keepdim=True)
    gram = nuisance.T @ (weight[:, None] * nuisance)
    rhs = nuisance.T @ (weight * r)
    coeff = torch.linalg.pinv(gram, rtol=rtol, hermitian=True) @ rhs
    removed = nuisance @ coeff
    retained = r - removed
    output = logs[0].double() + retained
    if not torch.isfinite(output).all():
        raise PathwayError('Nonfinite projected distribution')
    residual_error = nuisance.T @ (weight * retained)
    return output.float(), dict(rank=int(torch.linalg.matrix_rank(gram, rtol=rtol, hermitian=True)),
        removed_weighted_norm=float((weight * removed.square()).sum().sqrt()),
        retained_weighted_norm=float((weight * retained.square()).sum().sqrt()),
        orthogonality_error=float(residual_error.abs().max()),
        correctness_estimated=False)


def generate(model, reference, receiver, policy, *, algorithm, context_limit, layer=None, controls=()):
    """B restore/roll; C projection/geometric-layout-average; off/audit controls.

    layout_average uses the same four forwards as C (including baseline shadow)
    so a benefit cannot be attributed solely to extra model calls.
    """
    if algorithm not in ('off', 'audit', 'relay', 'relay_roll', 'project', 'layout_average'):
        raise ValueError('Unknown chain algorithm')
    use_views = algorithm in ('project', 'layout_average')
    if use_views and len(controls) != 2:
        raise PathwayError('Projection and its matched control require two certified views')
    prompts = [reference, receiver, *(controls if use_views else ())]
    check_inputs(model, prompts, policy, context_limit)
    if algorithm in ('relay', 'relay_roll', 'audit') and layer not in cut_layers(model):
        raise PathwayError('Use an A-predeclared middle boundary, not an arbitrary late layer')
    equal = torch.equal(reference.embeds, receiver.embeds)
    if equal and use_views and any(not torch.equal(p.embeds, reference.embeds) for p in controls):
        raise PathwayError('Empty central evidence cannot have nonempty controls')
    effective = 'off' if equal else algorithm
    use_shadow = effective != 'off'
    streams = [_Stream(model, p) for p in prompts]
    tokens, diagnostics, forward_counts = [], [], [0] * len(streams)
    _sync(receiver.embeds.device)
    if receiver.embeds.device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(receiver.embeds.device)
    started = time.perf_counter()
    relay = ResidualRelay(model, (layer,)) if effective in ('relay', 'relay_roll', 'audit') else None
    from contextlib import nullcontext
    with torch.inference_mode(), (relay if relay is not None else nullcontext()):
        for step in range(policy.max_new_tokens):
            if relay:
                with relay.use('reference', step):
                    streams[0].advance(tokens)
                forward_counts[0] += 1
                mode = {'relay': 'restore', 'relay_roll': 'roll', 'audit': 'audit'}[effective]
                with relay.use(mode, step, layer):
                    scores = streams[1].advance(tokens)
                forward_counts[1] += 1
            elif effective in ('project', 'layout_average'):
                logits = [s.advance(tokens) for s in streams]
                forward_counts = [n + 1 for n in forward_counts]
                if effective == 'project':
                    scores, diagnostic = project_residual(logits[0], logits[1], logits[2:])
                    diagnostics.append(dict(step=step, **diagnostic))
                else:
                    scores = torch.stack([z.float().log_softmax(-1) for z in logits[1:]]).mean(0)
            else:
                scores = streams[1].advance(tokens)
                forward_counts[1] += 1
            live = [s.cache for i, s in enumerate(streams) if forward_counts[i]]
            if len({id(v) for v in live}) != len(live):
                raise PathwayError('All active branches require private KV caches')
            # For log-prob fusion, applying repetition penalties depends on a logit
            # offset. Re-center to receiver logsumexp to preserve native scale.
            if effective in ('project', 'layout_average'):
                scores = scores - scores.logsumexp(-1) + logits[1].float().logsumexp(-1)
            token = policy.select(scores, tokens)
            tokens.append(token)
            if token in policy.eos_token_ids:
                break
        events = list(relay.events) if relay else []
    _sync(receiver.embeds.device)
    return dict(token_ids=tokens, algorithm=algorithm, effective_algorithm=effective,
        seconds=time.perf_counter()-started, forwards=forward_counts,
        bypass=equal, events=events, projection=diagnostics,
        changed_internal_steps=sum(x['patched'] for x in events) if relay else
            sum(x['removed_weighted_norm'] > 0 for x in diagnostics),
        stop_reason='eos' if tokens[-1] in policy.eos_token_ids else 'length',
        peak_allocated_bytes=torch.cuda.max_memory_allocated(receiver.embeds.device)
            if receiver.embeds.device.type == 'cuda' else None,
        weights_updated=False, clinical_correctness_estimated=False)
