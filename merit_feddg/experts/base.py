from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from PIL import Image


def load_rgb(image: str | Path | Image.Image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    with Image.open(image) as handle:
        return handle.convert("RGB")


def null_image_like(image: Image.Image) -> Image.Image:
    return Image.new("RGB", image.size, color=(127, 127, 127))


class ConceptExpert(ABC):
    """Every heterogeneous specialist is reduced to image-induced concept evidence."""

    @abstractmethod
    def image_null_scores(
        self,
        image: str | Path | Image.Image,
        prompt: str,
        concepts: list[str],
    ) -> np.ndarray:
        raise NotImplementedError

    def score_claims(
        self,
        image: str | Path | Image.Image,
        question: str,
        generated_prefix: str,
        claims: list[str],
    ) -> np.ndarray:
        """Score semantic claims while retaining compatibility with older adapters."""

        prompt = question.strip()
        if generated_prefix.strip():
            prompt = f"{prompt}\nAnswer so far: {generated_prefix.strip()}"
        return self.image_null_scores(image, prompt, claims)
