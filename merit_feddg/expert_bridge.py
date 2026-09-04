from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
from PIL import Image

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
    ) -> None:
        self.expert_id = expert_id
        self.capability = capability
        self.expert_factory = expert_factory
        self.image = image
        self.prompt = prompt
        self._expert: ConceptExpert | None = None

    @property
    def loaded(self) -> bool:
        return self._expert is not None

    def __call__(self, request: ClaimRequest) -> NativeEvidence:
        if self._expert is None:
            self._expert = self.expert_factory()
        scores = np.asarray(
            self._expert.image_null_scores(self.image, self.prompt, list(request.concepts)),
            dtype=float,
        )
        if scores.shape != (len(request.concepts),):
            raise ValueError("concept expert returned scores with the wrong shape")
        ordered = np.sort(scores)
        margin = float(ordered[-1] - ordered[-2])
        confidence = float(1.0 / (1.0 + np.exp(-margin)))
        return NativeEvidence(
            expert_id=self.expert_id,
            capability=self.capability,
            concept_scores=dict(zip(request.concepts, scores)),
            confidence=confidence,
            provenance={"adapter": type(self._expert).__name__, "prompt": self.prompt},
        )
