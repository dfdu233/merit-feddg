"""Single-image evidence-guided tokens with auditable, expiring intervention.

Uses the existing native expert runtime. Correctness-first production KV replay;
no cross-branch KV sharing or medical performance is claimed.
"""

from __future__ import annotations

from time import perf_counter

import numpy as np

from .block_decode import Block
from .capability_runtime import NativeSession, native_observation_prompt
from .evidence_decode import GuidanceConfig, bounded_distribution
from .evidence_packets import active_packet
from .open_study import fingerprint


class BoundedNativeSession(NativeSession):
    def __init__(self, probe, image, prompt, question, config, guidance=None):
        if config.visual_views != 0 or isinstance(image, (tuple, list)):
            raise ValueError("bounded evidence decoding requires exactly one original image")
        if not hasattr(probe, "new_answer_session"):
            raise TypeError("backend must provide a native answer session")
        super().__init__(probe, image, prompt, question, config)
        self.guidance = guidance or GuidanceConfig()
        self._base = probe.new_answer_session(image, prompt)
        if not hasattr(self._base, "next_scores"):
            raise TypeError("bounded decoding backend needs next_scores; no silent fallback")
        self._evidence_key, self._evidence_session = None, None
        self.last_guidance_trace = []
        self.last_guidance_spent = 0.0

    def propose(self, state, length):
        self.last_guidance_trace = []
        spent = state.guidance_spent
        self.last_guidance_spent = spent
        prefix, tokens, logps = tuple(state.prefix), [], []
        finished = False
        while len(tokens) < length:
            packet, audit = active_packet(
                state.items, self.question, self.config, prefix_tokens=len(prefix),
                lifetime=self.guidance.evidence_tokens, control=self.guidance.control,
            )
            if (not packet or self.guidance.strength == 0 or self.guidance.clip == 0
                    or self.guidance.token_kl == 0 or spent >= self.guidance.case_kl):
                # Important: NONE uses the original production greedy path,
                # without a new prompt, processor or a double forward pass.
                block = self._base.propose(prefix, 1, length - len(tokens))[0]
                tokens.extend(block.tokens)
                logps.extend([block.log_probability] * len(block.tokens))
                finished = block.finished
                self.last_guidance_trace.append({"event": "guidance_none", "token_start": len(prefix),
                                                "reason": "no_active_packet_or_budget", "audit": audit})
                break
            started = perf_counter()
            key = fingerprint(packet)
            if key != self._evidence_key:
                self._evidence_session = self.probe.new_answer_session(
                    self.image, native_observation_prompt(self.prompt, packet)
                )
                self._evidence_key = key
            base = self._base.next_scores(prefix)
            evidence = self._evidence_session.next_scores(prefix)
            scores, info = bounded_distribution(base, evidence, self.guidance, spent=spent)
            chosen = int(np.argmax(scores))
            info.update(event="guidance", token_start=len(prefix), token=chosen,
                        prefix_sha256=fingerprint(prefix), packet_sha256=key,
                        evidence_ids=[v["evidence_id"] for v in packet], audit=audit,
                        seconds=perf_counter() - started, backend="production_kv_replay")
            self.last_guidance_trace.append(info)
            spent = info["spent"]
            tokens.append(chosen)
            logps.append(float(scores[chosen]))
            prefix += (chosen,)
            if chosen in self._base.eos_ids:
                finished = True
                break
        self.last_guidance_spent = spent
        return Block(tuple(tokens), self.decode(tokens), float(np.mean(logps)) if logps else 0.0, finished)
