from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .base import ConceptExpert, load_rgb, null_image_like

MODALITY_PROMPTS = {
    "cxr": (
        "a chest X-ray radiograph",
        "a frontal chest radiograph",
        "a medical X-ray image of the thorax",
    ),
    "pathology": (
        "a histopathology microscopy image",
        "an H&E stained pathology slide",
        "a microscopic image of tissue pathology",
    ),
    "oct": (
        "a retinal optical coherence tomography image",
        "a retinal OCT B-scan",
        "an optical coherence tomography scan of the retina",
    ),
}


def route_probabilities(scores: np.ndarray, available: list[str]) -> dict[str, float]:
    values = np.asarray(scores, dtype=float)
    if values.shape != (len(available),):
        raise ValueError("router scores must match the available modality list")
    values = values - np.max(values)
    probabilities = np.exp(np.clip(values, -60.0, 60.0))
    probabilities = probabilities / probabilities.sum()
    return {name: float(value) for name, value in zip(available, probabilities)}


class BiomedClipAdapter(ConceptExpert):
    """Small biomedical contrastive model used for routing and a broad control.

    Routing uses only modality descriptions. Candidate evidence remains image-minus-null
    similarity and is used only by the broad-specialist baseline, never by MERIT itself.
    """

    def __init__(self, model_id: str, device: str = "auto") -> None:
        try:
            import torch
            from open_clip import create_model_from_pretrained, get_tokenizer
        except ImportError as exc:
            raise RuntimeError("install merit-feddg[research] for the BiomedCLIP router") from exc

        source = Path(model_id)
        model_source = f"local-dir:{source.resolve()}" if source.is_dir() else f"hf-hub:{model_id}"
        self.torch = torch
        self.model, self.preprocess = create_model_from_pretrained(model_source)
        self.tokenizer = get_tokenizer(model_source)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(self.device).eval()

    def _image_embedding(self, image: Image.Image):
        pixels = self.preprocess(image).unsqueeze(0).to(self.device)
        with self.torch.inference_mode():
            features = self.model.encode_image(pixels, normalize=True)
        return features

    def _text_embeddings(self, phrases: list[str]):
        try:
            tokens = self.tokenizer(phrases, context_length=256)
        except TypeError:
            tokens = self.tokenizer(phrases)
        tokens = tokens.to(self.device)
        with self.torch.inference_mode():
            return self.model.encode_text(tokens, normalize=True)

    def route(self, image: str | Path | Image.Image, available: list[str]) -> dict[str, float]:
        unknown = [name for name in available if name not in MODALITY_PROMPTS]
        if unknown:
            raise ValueError(f"BiomedCLIP has no modality prompts for: {unknown}")
        native = load_rgb(image)
        image_features = self._image_embedding(native)
        modality_features = []
        for name in available:
            prompts = list(MODALITY_PROMPTS[name])
            features = self._text_embeddings(prompts).mean(dim=0, keepdim=True)
            features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            modality_features.append(features)
        text_features = self.torch.cat(modality_features, dim=0)
        scale = self.model.logit_scale.exp().detach().clamp(max=100.0)
        scores = (scale * image_features @ text_features.T).squeeze(0).float().cpu().numpy()
        return route_probabilities(scores, available)

    def image_null_scores(
        self,
        image: str | Path | Image.Image,
        prompt: str,
        concepts: list[str],
    ) -> np.ndarray:
        native = load_rgb(image)
        null = null_image_like(native)
        phrases = [f"a medical image showing {concept}" for concept in concepts]
        text_features = self._text_embeddings(phrases)
        scale = self.model.logit_scale.exp().detach().clamp(max=100.0)
        native_scores = scale * self._image_embedding(native) @ text_features.T
        null_scores = scale * self._image_embedding(null) @ text_features.T
        return (native_scores - null_scores).squeeze(0).float().cpu().numpy()
