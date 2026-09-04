from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image

from .base import ConceptExpert, load_rgb, null_image_like


def resolve_checkpoint_source(model_id: str) -> str:
    source = Path(model_id)
    if not source.is_dir():
        return model_id
    preferred = source / "pytorch_model.bin"
    if preferred.is_file():
        return str(preferred)
    candidates = sorted(
        candidate
        for candidate in source.rglob("*")
        if candidate.is_file() and candidate.suffix in {".bin", ".pt", ".pth"}
    )
    if not candidates:
        raise FileNotFoundError(f"CONCH checkpoint was not found below {source}")
    return str(candidates[0])


class ConchConceptExpert(ConceptExpert):
    """Contrastive pathology adapter using the official CONCH package."""

    def __init__(self, model_id: str = "hf_hub:MahmoodLab/CONCH", device: str = "auto") -> None:
        try:
            import torch
            from conch.open_clip_custom import (
                create_model_from_pretrained,
                get_tokenizer,
                tokenize,
            )
        except ImportError as exc:
            raise RuntimeError(
                "install CONCH with: pip install git+https://github.com/Mahmoodlab/CONCH.git"
            ) from exc
        self.torch = torch
        self.tokenize = tokenize
        self.tokenizer = get_tokenizer()
        token = os.getenv("HF_TOKEN")
        self.model, self.preprocess = create_model_from_pretrained(
            "conch_ViT-B-16", resolve_checkpoint_source(model_id), hf_auth_token=token
        )
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(self.device).eval()

    def _image_embedding(self, image: Image.Image):
        pixels = self.preprocess(image).unsqueeze(0).to(self.device)
        with self.torch.inference_mode():
            return self.model.encode_image(pixels, proj_contrast=True, normalize=True)

    def image_null_scores(
        self,
        image: str | Path | Image.Image,
        prompt: str,
        concepts: list[str],
    ) -> np.ndarray:
        scores, _ = self.score_and_domain_embedding(image, prompt, concepts)
        return scores

    def score_and_domain_embedding(
        self,
        image: str | Path | Image.Image,
        prompt: str,
        concepts: list[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Reuse the native image forward for evidence and post-call OOD."""

        native = load_rgb(image)
        null = null_image_like(native)
        phrases = [self._claim_phrase(concept) for concept in concepts]
        text = self.tokenize(self.tokenizer, phrases).to(self.device)
        with self.torch.inference_mode():
            text_features = self.model.encode_text(text, normalize=True)
            native_feature = self._image_embedding(native)
            native_scores = native_feature @ text_features.T
            null_scores = self._image_embedding(null) @ text_features.T
        scores = (native_scores - null_scores).squeeze(0).float().cpu().numpy()
        feature = native_feature.squeeze(0).float().cpu().numpy()
        return scores, feature

    @staticmethod
    def _claim_phrase(concept: str) -> str:
        """Avoid wrapping an already semantic claim in a second sentence."""

        clean = " ".join(str(concept).strip().split())
        lowered = clean.casefold()
        semantic_starts = (
            "the image ",
            "the tissue ",
            "histopathology ",
            "for the clinical question ",
        )
        if lowered.startswith(semantic_starts):
            return clean if clean.endswith(".") else f"{clean}."
        return f"Histopathology shows {clean.rstrip('.')}."

    def domain_embedding(self, image: str | Path | Image.Image) -> np.ndarray:
        """Return the frozen native CONCH feature used for source-only OOD fitting."""

        feature = self._image_embedding(load_rgb(image))
        return feature.squeeze(0).float().cpu().numpy()
