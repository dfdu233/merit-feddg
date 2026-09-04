from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from .med_defer import ClaimRequest, ClaimTrace, LazyExpertPool, MedDeferEngine

RequestFactory = Callable[[str, float, int], ClaimRequest]


class MedDeferLogitsProcessor:
    """Transformers-compatible, lazy specialist guidance during autoregressive decoding.

    A decision is made at the beginning of generation and after punctuation-delimited
    clinical claims. The selected expert is called once per claim. Its concept scores
    are converted to bounded phrase-token bias until the claim ends.
    """

    def __init__(
        self,
        tokenizer: Any,
        engine: MedDeferEngine,
        pool: LazyExpertPool,
        request_factory: RequestFactory,
        uncertainty_threshold: float = 0.35,
        top_k_entropy: int = 64,
        boundary_text: str = ".;:!?\n",
    ) -> None:
        self.tokenizer = tokenizer
        self.engine = engine
        self.pool = pool
        self.request_factory = request_factory
        self.uncertainty_threshold = float(uncertainty_threshold)
        self.top_k_entropy = int(top_k_entropy)
        self.boundary_text = boundary_text
        self.prompt_length: int | None = None
        self.claim_index = 0
        self.claim_started = False
        self.active_trace: ClaimTrace | None = None
        self.active_concepts: tuple[str, ...] = ()
        self.traces: list[ClaimTrace] = []
        self._concept_tokens: dict[str, tuple[tuple[int, ...], ...]] = {}

    def _uncertainty(self, scores: Any) -> float:
        import torch

        count = min(self.top_k_entropy, int(scores.shape[-1]))
        values = torch.topk(scores.float(), k=count, dim=-1).values
        probabilities = torch.softmax(values, dim=-1)
        entropy = -(probabilities * torch.log(probabilities.clamp_min(1e-12))).sum(dim=-1)
        return float((entropy / np.log(max(count, 2))).mean().detach().cpu())

    def _is_boundary(self, generated: Any) -> bool:
        if generated.shape[-1] == 0:
            return True
        text = self.tokenizer.decode(generated[0, -1:], skip_special_tokens=True)
        return any(marker in text for marker in self.boundary_text)

    def _tokens(self, concept: str) -> tuple[tuple[int, ...], ...]:
        if concept not in self._concept_tokens:
            sentence_case = concept[:1].upper() + concept[1:]
            variants = (concept, " " + concept, sentence_case, " " + sentence_case)
            encoded_variants = []
            for variant in variants:
                encoded = self.tokenizer(variant, add_special_tokens=False)["input_ids"]
                if encoded and isinstance(encoded[0], list):
                    encoded = encoded[0]
                tokens = tuple(int(token) for token in encoded)
                if tokens and tokens not in encoded_variants:
                    encoded_variants.append(tokens)
            self._concept_tokens[concept] = tuple(encoded_variants)
        return self._concept_tokens[concept]

    @staticmethod
    def _suffix_matches(generated: list[int], prefix: tuple[int, ...]) -> bool:
        return not prefix or (
            len(generated) >= len(prefix) and generated[-len(prefix) :] == list(prefix)
        )

    def _apply_phrase_bias(self, input_ids: Any, scores: Any, trace: ClaimTrace) -> Any:
        generated = input_ids[0, self.prompt_length :].tolist()
        for concept, weight in zip(self.active_concepts, trace.concept_delta):
            variants = self._tokens(concept)
            if not variants or weight == 0.0:
                continue
            # Cover sentence-initial/non-initial casing and continue any
            # in-progress multi-token concept phrase. Token alternatives are
            # de-duplicated so equivalent tokenizer forms are not overweighted.
            next_tokens: set[int] = set()
            for tokens in variants:
                next_token = tokens[0]
                for prefix_length in range(len(tokens) - 1, 0, -1):
                    if self._suffix_matches(generated, tokens[:prefix_length]):
                        next_token = tokens[prefix_length]
                        break
                next_tokens.add(next_token)
            for next_token in next_tokens:
                scores[:, next_token] += float(weight)
        return scores

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        if self.prompt_length is None:
            self.prompt_length = int(input_ids.shape[-1])
        generated = input_ids[:, self.prompt_length :]
        at_boundary = generated.shape[-1] == 0 or self._is_boundary(generated)
        if at_boundary and self.claim_started:
            self.claim_index += 1
            self.claim_started = False
            self.active_trace = None

        uncertainty = self._uncertainty(scores)
        if at_boundary and not self.claim_started:
            prefix = self.tokenizer.decode(generated[0], skip_special_tokens=True)
            request = self.request_factory(prefix, uncertainty, self.claim_index)
            self.active_trace = self.engine.guide(request, self.pool)
            self.active_concepts = request.concepts
            self.traces.append(self.active_trace)
            self.claim_started = True

        if self.active_trace is not None and self.active_trace.gate > 0.0:
            scores = self._apply_phrase_bias(input_ids, scores, self.active_trace)
        return scores
