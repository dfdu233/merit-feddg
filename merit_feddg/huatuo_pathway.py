"""Native Huatuo input expansion and paired incremental pathway decoding.

The no-evidence shadow consumes the receiver's committed tokens. It is NOT an
independently generated baseline answer once their trajectories diverge.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time

import torch

from .pathway_restore import PathwayConfig, PathwayError, PathwayRestorer, VisualSpan


@dataclass(frozen=True)
class PreparedPrompt:
    embeds: torch.Tensor
    mask: torch.Tensor
    positions: torch.Tensor
    visual_span: VisualSpan

    @property
    def length(self):
        return self.embeds.shape[1]

    def validate(self):
        if self.embeds.ndim != 3 or self.embeds.shape[0] != 1:
            raise PathwayError('Only an unpadded, batch-one prompt is supported')
        if self.mask.shape != (1, self.length) or not bool(self.mask.bool().all()):
            raise PathwayError('Padding is unsupported in this paired decoder')
        expected = torch.arange(self.length, device=self.positions.device).unsqueeze(0)
        if not torch.equal(self.positions, expected) or self.visual_span.stop >= self.length:
            raise PathwayError('Non-contiguous positions or no query after image')
        if not torch.isfinite(self.embeds).all():
            raise PathwayError('Non-finite multimodal embeddings')


@dataclass(frozen=True)
class GreedySpec:
    max_new_tokens: int
    eos_token_ids: tuple[int, ...]
    repetition_penalty: float = 1.0
    min_new_tokens: int = 0
    processor_prefix: tuple[int, ...] = ()

    def __post_init__(self):
        if type(self.max_new_tokens) is not int or self.max_new_tokens < 1:
            raise ValueError('Positive explicit answer budget required')
        if not self.eos_token_ids or any(type(t) is not int or t < 0 for t in self.eos_token_ids):
            raise ValueError('Explicit EOS IDs required')
        if not math.isfinite(self.repetition_penalty) or self.repetition_penalty <= 0:
            raise ValueError('Invalid repetition penalty')
        if type(self.min_new_tokens) is not int or not 0 <= self.min_new_tokens < self.max_new_tokens:
            raise ValueError('Invalid minimum generation length')

    def select(self, logits, prefix):
        scores = logits.float().clone()
        if scores.ndim != 1 or not torch.isfinite(scores).all():
            raise PathwayError('Non-finite or non-vector next-token logits')
        if max(self.eos_token_ids) >= scores.numel():
            raise PathwayError('EOS token outside the model vocabulary')
        processor_tokens = self.processor_prefix + tuple(prefix)
        if processor_tokens and self.repetition_penalty != 1:
            ids = torch.tensor(sorted(set(processor_tokens)), device=scores.device)
            old = scores[ids]
            scores[ids] = torch.where(old < 0, old * self.repetition_penalty,
                                     old / self.repetition_penalty)
        if len(prefix) < self.min_new_tokens:
            scores[list(self.eos_token_ids)] = -torch.inf
        if not torch.isfinite(scores).any():
            raise PathwayError('Generation constraints removed all tokens')
        return int(scores.argmax())


def native_inputs(adapter, image, prompt):
    # Existing HuatuoOEAdapter has been exercised with PIL RGB inputs.
    from pathlib import Path
    from PIL import Image
    if isinstance(image, (str, Path)):
        with Image.open(image) as opened:
            image = opened.convert('RGB')
    return adapter._inputs(image, prompt)


def prepare_native(adapter, image, prompt):
    """Use the existing adapter's exact single-image serializer, then native expansion.

    Unique position markers are used as bookkeeping labels to detect actual
    visual spans and any silent truncation. They are NOT task/reference labels.
    No labels are passed to the language model or used to calculate a loss.
    """
    from llava.constants import IMAGE_TOKEN_INDEX

    ids, pixels = native_inputs(adapter, image, prompt)
    if ids.ndim != 2 or ids.shape[0] != 1 or int((ids == IMAGE_TOKEN_INDEX).sum()) != 1:
        raise PathwayError('Require exactly one image placeholder and one prompt')
    if pixels is None or pixels.ndim != 4 or pixels.shape[0] != 1:
        raise PathwayError('Require the original single-image tensor; no tiled/multiview substitution')
    marks = torch.arange(ids.shape[1], device=ids.device).unsqueeze(0)
    with torch.inference_mode():
        _, positions, mask, _, embeds, aligned = adapter.model.prepare_inputs_labels_for_multimodal_new(
            ids, None, None, None, marks, pixels)
    if aligned is None or embeds is None:
        raise PathwayError('Native multimodal expansion did not expose aligned markers')
    if not torch.equal(aligned[aligned != -100], marks[ids != IMAGE_TOKEN_INDEX]):
        raise PathwayError('Native expansion truncated or reordered the prompt')
    locations = (aligned[0] == -100).nonzero().flatten()
    if locations.numel() == 0 or int(locations[-1] - locations[0] + 1) != locations.numel():
        raise PathwayError('Cannot identify one contiguous native visual span')
    span = VisualSpan(int(locations[0]), int(locations[-1]) + 1)
    if mask is None:
        mask = torch.ones(embeds.shape[:2], device=embeds.device, dtype=torch.long)
    if positions is None:
        positions = torch.arange(embeds.shape[1], device=embeds.device).unsqueeze(0)
    prepared = PreparedPrompt(embeds, mask, positions, span)
    prepared.validate()
    return prepared, pixels


def prepare_pair(adapter, image, baseline_prompt, evidence_prompt):
    reference, original_pixels = prepare_native(adapter, image, baseline_prompt)
    receiver, evidence_pixels = prepare_native(adapter, image, evidence_prompt)
    if not torch.equal(original_pixels, evidence_pixels):
        raise PathwayError('Branches must use identical original pixels')
    a, b = reference.visual_span, receiver.visual_span
    if not torch.equal(reference.embeds[:, a.start:a.stop], receiver.embeds[:, b.start:b.stop]):
        raise PathwayError('Branches must use identical pre-decoder image embeddings')
    return reference, receiver


class _Stream:
    def __init__(self, model, prepared):
        self.model, self.prepared = model, prepared
        self.cache = None
        self.consumed = ()

    def advance(self, prefix):
        if self.cache is None:
            if prefix:
                raise PathwayError('Initial prefill must have no committed output')
            kwargs = dict(inputs_embeds=self.prepared.embeds,
                          position_ids=self.prepared.positions, attention_mask=self.prepared.mask)
        else:
            if tuple(prefix[:-1]) != self.consumed or len(prefix) != len(self.consumed) + 1:
                raise PathwayError('A stream can consume exactly one new committed token')
            device = self.prepared.embeds.device
            kwargs = dict(input_ids=torch.tensor([[prefix[-1]]], device=device),
                          position_ids=torch.tensor([[self.prepared.length + len(prefix) - 1]], device=device),
                          attention_mask=torch.ones((1, self.prepared.length + len(prefix)),
                                                    dtype=self.prepared.mask.dtype, device=device),
                          past_key_values=self.cache)
        output = self.model(**kwargs, use_cache=True, output_attentions=False, return_dict=True)
        if output.past_key_values is None:
            raise PathwayError('Backend did not return a reusable private KV cache')
        self.cache = output.past_key_values
        self.consumed = tuple(prefix)
        return output.logits[0, -1]


def _sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def paired_generate(model, reference, receiver, spec, config=PathwayConfig(), *, context_limit):
    """Greedy reference/receiver streams, private KV, same committed token prefix.

    Does not use ground truth, candidate rankings, confidence thresholds, or
    trained probes. Explicit generation settings must pass native parity before
    interpreting results. The off arm installs no attention hooks.
    """
    reference.validate()
    receiver.validate()
    if max(reference.length, receiver.length) + spec.max_new_tokens > context_limit:
        raise PathwayError('Prompt plus full reserved answer exceeds model context')
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise PathwayError('Freeze the model explicitly before calling paired_generate')
    equal = (torch.equal(reference.embeds, receiver.embeds)
             and torch.equal(reference.positions, receiver.positions)
             and torch.equal(reference.mask, receiver.mask))
    active_config = PathwayConfig('off', config.layers) if equal else config
    use_shadow = active_config.mode != 'off'
    base, target = _Stream(model, reference), _Stream(model, receiver)
    tokens = []
    _sync(receiver.embeds.device)
    if receiver.embeds.device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(receiver.embeds.device)
    started = time.perf_counter()
    try:
        with torch.inference_mode(), PathwayRestorer(model, active_config) as restorer:
            for step in range(spec.max_new_tokens):
                if use_shadow:
                    with restorer.pass_context('reference', step, reference.length, reference.visual_span):
                        base.advance(tokens)
                with restorer.pass_context('receiver', step, receiver.length, receiver.visual_span):
                    logits = target.advance(tokens)
                if use_shadow and base.cache is target.cache:
                    raise PathwayError('Reference and receiver must own separate KV caches')
                token = spec.select(logits, tokens)
                tokens.append(token)
                if token in spec.eos_token_ids:
                    break
            events = list(restorer.events)
    except PathwayError as error:
        _sync(receiver.embeds.device)
        error.decode_diagnostics = dict(
            token_ids=list(tokens), seconds=time.perf_counter() - started,
            reference_forwards=len(tokens) + 1 if use_shadow else 0,
            receiver_forwards=len(tokens) + 1, identical_input_bypass=equal,
            events=getattr(error, 'pathway_events', []),
            peak_allocated_bytes=(torch.cuda.max_memory_allocated(receiver.embeds.device)
                                  if receiver.embeds.device.type == 'cuda' else None),
            peak_reserved_bytes=(torch.cuda.max_memory_reserved(receiver.embeds.device)
                                 if receiver.embeds.device.type == 'cuda' else None),
            hooks_removed=not getattr(model, '_merit_pathway_active', False))
        raise
    _sync(receiver.embeds.device)
    return dict(token_ids=tokens, mode=config.mode, effective_mode=active_config.mode,
                identical_input_bypass=equal, events=events, seconds=time.perf_counter() - started,
                private_cache_identity_checked=use_shadow,
                reference_forwards=len(tokens) if use_shadow else 0, receiver_forwards=len(tokens),
                stop_reason='eos' if tokens[-1] in spec.eos_token_ids else 'length',
                reference_semantics='same committed prefix, not standalone baseline trajectory',
                peak_allocated_bytes=(torch.cuda.max_memory_allocated(receiver.embeds.device)
                                      if receiver.embeds.device.type == 'cuda' else None),
                peak_reserved_bytes=(torch.cuda.max_memory_reserved(receiver.embeds.device)
                                     if receiver.embeds.device.type == 'cuda' else None),
                medical_correctness_estimated=False, weights_updated=False)
