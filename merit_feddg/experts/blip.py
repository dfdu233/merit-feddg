from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .base import ConceptExpert, load_rgb, null_image_like


class BlipConceptExpert(ConceptExpert):
    """Phrase-likelihood adapter for the compact 247M LO-VLM checkpoint."""

    def __init__(self, model_id: str, device: str = "auto") -> None:
        try:
            import torch
            from transformers import BlipForConditionalGeneration, BlipProcessor
        except ImportError as exc:
            raise RuntimeError("install merit-feddg[research] to load BLIP experts") from exc
        self.torch = torch
        self.processor = BlipProcessor.from_pretrained(model_id)
        self.model = BlipForConditionalGeneration.from_pretrained(model_id)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(self.device).eval()

    def _score(self, image: Image.Image, prompt: str, concept: str) -> float:
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
        labels = self.processor.tokenizer(
            concept,
            return_tensors="pt",
            add_special_tokens=True,
        ).input_ids.to(self.device)
        with self.torch.inference_mode():
            output = self.model(**inputs, labels=labels)
        token_count = max(int(labels.numel()), 1)
        return -float(output.loss.detach().cpu()) * token_count

    def image_null_scores(
        self,
        image: str | Path | Image.Image,
        prompt: str,
        concepts: list[str],
    ) -> np.ndarray:
        native = load_rgb(image)
        null = null_image_like(native)
        return np.asarray(
            [
                self._score(native, prompt, item) - self._score(null, prompt, item)
                for item in concepts
            ],
            dtype=float,
        )
