"""PLIP pathology image-text verifier.

PLIP (Nature Medicine 2023) is used only as a frozen contrastive verifier. Its
similarity scores are not diagnosis probabilities and acquire no commit
authority without a source-only MERIT qualification card.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import ConceptExpert, load_rgb, null_image_like


class PlipConceptExpert(ConceptExpert):
    def __init__(self, model_id: str = "vinid/plip", device: str = "auto"):
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise RuntimeError("PLIP requires torch and transformers") from exc
        self.torch = torch
        kwargs = {"local_files_only": True} if Path(model_id).is_dir() else {}
        self.processor = CLIPProcessor.from_pretrained(model_id, **kwargs)
        self.model = CLIPModel.from_pretrained(model_id, **kwargs)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = self.model.to(self.device).eval().requires_grad_(False)

    def _image_embedding(self, image):
        rgb = load_rgb(image)
        inputs = self.processor(images=rgb, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        with self.torch.inference_mode():
            value = self.model.get_image_features(pixel_values=pixel_values)
        value = value / value.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return value

    def _text_embeddings(self, prompts):
        inputs = self.processor(
            text=list(prompts),
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        limit = self.model.config.text_config.max_position_embeddings
        if inputs["input_ids"].shape[-1] > limit:
            raise ValueError(
                f"PLIP claim exceeds native text capacity ({limit} tokens); "
                "refusing to truncate transaction evidence"
            )
        payload = {
            key: value.to(self.device)
            for key, value in inputs.items()
            if key in {"input_ids", "attention_mask"}
        }
        with self.torch.inference_mode():
            value = self.model.get_text_features(**payload)
        return value / value.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    @staticmethod
    def _claim_phrase(concept):
        text = " ".join(str(concept).strip().split())
        if not text:
            raise ValueError("PLIP claim cannot be empty")
        if text.endswith("."):
            return text
        return text + "."

    def score_claims(self, image, _question, _prefix, concepts):
        prompts = [self._claim_phrase(value) for value in concepts]
        native = self._image_embedding(image)
        null = self._image_embedding(null_image_like(load_rgb(image)))
        text = self._text_embeddings(prompts)
        scores = ((native - null) @ text.T).squeeze(0)
        result = scores.detach().float().cpu().numpy()
        if result.shape != (len(prompts),) or not np.isfinite(result).all():
            raise ValueError("PLIP returned invalid claim scores")
        return result

    def image_null_scores(self, image, prompt, concepts):
        return self.score_claims(image, prompt, "", concepts)

    def domain_embedding(self, image):
        return self._image_embedding(image).squeeze(0).detach().float().cpu().numpy()
