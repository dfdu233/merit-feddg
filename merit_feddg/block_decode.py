"""Expert intervention before committing short, dynamically generated token blocks.

This is block-level decoding, not token-level logit injection or a claim detector.
No references or fixed answer vocabulary are accepted by this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

import numpy as np

from .claims import CandidateProposition, ClaimSpec
from .evidence_bridges import EvidenceBridgeRegistry
from .med_defer import NativeEvidence


@dataclass(frozen=True)
class Block:
    tokens: tuple[int, ...]
    text: str
    log_probability: float  # Mean true transition log probability, including EOS.
    finished: bool = False


class BlockSession(Protocol):
    def propose(self, prefix: tuple[int, ...], count: int, length: int) -> list[Block]: ...
    def decode(self, tokens: tuple[int, ...]) -> str: ...


@dataclass(frozen=True)
class BlockConfig:
    candidates: int = 3
    block_tokens: int = 12
    max_new_tokens: int = 60
    max_calls: int = 3
    strength: float = 0.5
    plausibility_gap: float = 2.0

    def __post_init__(self):
        if min(self.candidates, self.block_tokens, self.max_new_tokens) < 1:
            raise ValueError("positive generation sizes required")
        if self.max_calls < 0 or not np.isfinite([self.strength, self.plausibility_gap]).all():
            raise ValueError("invalid intervention budget")
        if self.strength < 0 or self.plausibility_gap < 0:
            raise ValueError("nonnegative strength and plausibility gap required")


def decode_blocks(
    session: BlockSession,
    *,
    question: str,
    modality: str,
    capability: str,
    config: BlockConfig,
    evidence=None,
    expert_id: str | None = None,
    reverse_scores: bool = False,
) -> dict:
    """One expert per block; selecting no expert never loads an expert model.

    ``evidence(claim, prefix)`` returns native evidence for these exact dynamic
    propositions. Mask/box/text evidence must explicitly map to candidate IDs;
    ungrounded native payloads fail closed in the shared evidence bridges.
    """
    prefix: tuple[int, ...] = ()
    trace, calls = [], 0
    start = perf_counter()
    bridges = EvidenceBridgeRegistry()
    while len(prefix) < config.max_new_tokens:
        candidates = session.propose(
            prefix,
            config.candidates,
            min(config.block_tokens, config.max_new_tokens - len(prefix)),
        )
        if not candidates or any(not c.tokens for c in candidates):
            raise ValueError("generalist returned an empty block")
        if any(len(c.tokens) > config.max_new_tokens - len(prefix) for c in candidates):
            raise ValueError("generalist exceeded token budget")
        base = np.asarray([c.log_probability for c in candidates], dtype=float)
        if not np.isfinite(base).all():
            raise ValueError("nonfinite generalist transition scores")
        before = session.decode(prefix)
        adjusted, delta = base.copy(), np.zeros(len(base))
        reason, used, expert_seconds = "NONE:no_qualified_expert", None, 0.0
        evidence_summary = None
        if evidence is not None and calls < config.max_calls and len(candidates) >= 2:
            # This closed set is ephemeral beam continuations, NOT dataset answers.
            claim = ClaimSpec(
                claim_id=f"block-{len(trace)}",
                question=question,
                modality=modality,
                required_capabilities=(capability,),
                closed_set=True,
                propositions=tuple(
                    CandidateProposition(
                        candidate_id=f"b{i}",
                        answer=c.text or "[end of answer]",
                        proposition=f"For the clinical question {question} Answer: {session.decode(prefix + c.tokens)}",
                    )
                    for i, c in enumerate(candidates)
                ),
                metadata={"kind": "dynamic_token_block", "prefix": before},
            )
            calls += 1
            call_start = perf_counter()
            native = evidence(claim, before)
            expert_seconds = perf_counter() - call_start
            if not isinstance(native, NativeEvidence) or native.expert_id != expert_id:
                raise ValueError("adapter evidence identity mismatch")
            converted = bridges.convert(claim, native)
            reason = converted.reason
            evidence_summary = {
                "capability": native.capability,
                "confidence": converted.confidence,
                "mask_count": len(native.masks),
                "boxes": list(native.boxes),
                "generated_text": native.generated_text,
                "candidate_support": [
                    {
                        "id": c.candidate_id,
                        "support": c.support,
                        "contradiction": c.contradiction,
                        "coverage": c.coverage,
                    }
                    for c in converted.candidates
                ],
            }
            if not converted.abstained:
                by_id = {c.candidate_id: c.signed_support for c in converted.candidates}
                support = np.asarray([by_id[p.candidate_id] for p in claim.propositions])
                if reverse_scores:
                    support = support[::-1].copy()
                # Bounded, mean-centered perturbation; no arbitrary z-score amplification.
                delta = config.strength * converted.confidence * (support - support.mean())
                adjusted = base + delta
                adjusted[base < base.max() - config.plausibility_gap] = -np.inf
                used = native.expert_id
        elif evidence is not None:
            reason = "NONE:budget_or_single_candidate"
        chosen = int(np.argmax(adjusted))
        selected = candidates[chosen]
        trace.append(
            {
                "block": len(trace),
                "prefix": before,
                "token_start": len(prefix),
                "token_end": len(prefix) + len(selected.tokens),
                "expert": used,
                "reason": reason,
                "expert_seconds": expert_seconds,
                "evidence": evidence_summary,
                "candidates": [
                    {"text": c.text, "token_ids": list(c.tokens), "finished": c.finished}
                    for c in candidates
                ],
                "base_scores": base.tolist(),
                "expert_delta": delta.tolist(),
                "selected": chosen,
                "base_selected": int(np.argmax(base)),
            }
        )
        prefix += selected.tokens
        if selected.finished:
            break
    return {
        "text": session.decode(prefix),
        "token_ids": list(prefix),
        "trace": trace,
        "expert_calls": calls,
        "seconds": perf_counter() - start,
    }


class QwenBlockSession:
    """Real autoregressive beam proposals; preserve exact committed token IDs.

    Each block re-prefills the prefix. Cross-block KV reuse is NOT implemented;
    measure end-to-end latency rather than claiming a decoding speedup.
    """

    def __init__(self, probe, image, prompt: str):
        from .experts.base import load_rgb

        self.probe = probe
        self.inputs = probe._inputs(load_rgb(image), prompt)

    def decode(self, tokens):
        return self.probe.processor.tokenizer.decode(tokens, skip_special_tokens=True)

    def propose(self, prefix, count, length):
        torch, model = self.probe.torch, self.probe.model
        inputs = dict(self.inputs)
        if prefix:
            extra = torch.tensor(
                [prefix], device=inputs["input_ids"].device, dtype=inputs["input_ids"].dtype
            )
            inputs["input_ids"] = torch.cat((inputs["input_ids"], extra), dim=1)
            inputs["attention_mask"] = torch.cat(
                (inputs["attention_mask"], torch.ones_like(extra)), dim=1
            )
        start = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                num_beams=count,
                num_return_sequences=count,
                max_new_tokens=length,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
                length_penalty=1.0,
            )
            scores = (
                model.compute_transition_scores(
                    out.sequences,
                    out.scores,
                    getattr(out, "beam_indices", None),
                    normalize_logits=True,
                )
                .float()
                .cpu()
                .tolist()
            )
        eos = model.generation_config.eos_token_id
        eos_ids = set(eos if isinstance(eos, (tuple, list)) else [eos])
        blocks, seen = [], set()
        for ids, values in zip(out.sequences[:, start:].cpu().tolist(), scores, strict=True):
            end = next((i + 1 for i, token in enumerate(ids) if token in eos_ids), len(ids))
            ids = tuple(ids[:end])
            if ids in seen:
                continue
            seen.add(ids)
            if end > len(values):
                raise ValueError("transition scores do not cover generated tokens")
            blocks.append(
                Block(
                    ids,
                    self.decode(ids),
                    float(np.mean(values[:end])),
                    bool(ids and ids[-1] in eos_ids),
                )
            )
        return blocks
