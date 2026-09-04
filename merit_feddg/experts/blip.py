from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .base import ConceptExpert, load_rgb, null_image_like


def _continuation_labels(prompt_ids, continuation_ids):
    """Return a full decoder sequence while scoring only the continuation."""

    import torch

    full_ids = torch.cat([prompt_ids, continuation_ids], dim=1)
    labels = full_ids.clone()
    labels[:, : prompt_ids.shape[1]] = -100
    return full_ids, labels


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
        continuation_ids = self.processor.tokenizer(
            " " + concept,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids.to(self.device)
        full_ids, labels = _continuation_labels(inputs["input_ids"], continuation_ids)
        inputs["input_ids"] = full_ids
        inputs["attention_mask"] = self.torch.ones_like(full_ids)
        with self.torch.inference_mode():
            output = self.model(**inputs, labels=labels)
        token_count = max(int(continuation_ids.numel()), 1)
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
