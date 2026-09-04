from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
from PIL import Image

from .claims import CandidateProposition, ClaimSpec
from .evidence_bridges import EvidenceBridgeRegistry
from .experts.base import ConceptExpert
from .med_defer import ClaimRequest, NativeEvidence


class LazyConceptExpertProvider:
    """Bridge an existing concept expert into Med-DEFER without eager model loading."""

    def __init__(
        self,
        expert_id: str,
        capability: str,
        expert_factory: Callable[[], ConceptExpert],
        image: str | Path | Image.Image,
        prompt: str,
        bridge_registry: EvidenceBridgeRegistry | None = None,
    ) -> None:
        self.expert_id = expert_id
        self.capability = capability
        self.expert_factory = expert_factory
        self.image = image
        self.prompt = prompt
        self.bridge_registry = bridge_registry or EvidenceBridgeRegistry()
        self._expert: ConceptExpert | None = None

    @property
    def loaded(self) -> bool:
        return self._expert is not None

    def release(self) -> None:
        """Drop the lazily loaded specialist before accelerator collection."""

        self._expert = None

    def __call__(self, request: ClaimRequest) -> NativeEvidence:
        if self._expert is None:
            self._expert = self.expert_factory()
        question = request.question or self.prompt
        expert_queries = list(request.expert_queries or request.concepts)
        if hasattr(self._expert, "score_claims"):
            raw_scores = self._expert.score_claims(
                self.image,
                question,
                request.generated_prefix,
                expert_queries,
            )
        else:
            # Third-party adapters written against the v0.4 contract remain
            # usable; new adapters should implement prefix-conditioned claims.
            contextual_prompt = question
            if request.generated_prefix.strip():
                contextual_prompt += f"\nAnswer so far: {request.generated_prefix.strip()}"
            raw_scores = self._expert.image_null_scores(
                self.image, contextual_prompt, expert_queries
            )
        scores = np.asarray(
            raw_scores,
            dtype=float,
        )
        if scores.shape != (len(request.concepts),):
            raise ValueError("concept expert returned scores with the wrong shape")
        ordered = np.sort(scores)
        margin = float(ordered[-1] - ordered[-2])
        confidence = float(1.0 / (1.0 + np.exp(-margin)))
        raw_evidence = NativeEvidence(
            expert_id=self.expert_id,
            capability=self.capability,
            concept_scores=dict(zip(expert_queries, scores)),
            confidence=confidence,
            provenance={
                "adapter": type(self._expert).__name__,
                "question": question,
                "generated_prefix": request.generated_prefix,
                "expert_queries": expert_queries,
                "score_semantics": "logit",
            },
        )
        claim = ClaimSpec(
            claim_id=request.claim_id,
            question=question,
            modality=request.modality,
            required_capabilities=(self.capability,),
            propositions=tuple(
                CandidateProposition(
                    candidate_id=f"candidate-{index}",
                    answer=concept,
                    proposition=query,
                )
                for index, (concept, query) in enumerate(
                    zip(request.concepts, expert_queries)
                )
            ),
            closed_set=True,
            metadata={"task_type": request.task_type},
        )
        converted = self.bridge_registry.convert(claim, raw_evidence)
        provenance = {
            **raw_evidence.provenance,
            "semantic_bridge_validated": not converted.abstained,
            "semantic_bridge_reason": converted.reason,
        }
        if converted.abstained:
            # Preserve the native semantic payload for audit, but do not remap it
            # into decoder candidates. MedDeferEngine rejects the unvalidated flag.
            mapped_scores = dict(raw_evidence.concept_scores)
        else:
            mapped_scores = {
                concept: candidate.signed_support
                for concept, candidate in zip(request.concepts, converted.candidates)
            }
        return NativeEvidence(
            expert_id=self.expert_id,
            capability=self.capability,
            concept_scores=mapped_scores,
            confidence=confidence,
            provenance=provenance,
        )
