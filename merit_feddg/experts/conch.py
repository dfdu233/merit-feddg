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
        native = load_rgb(image)
        null = null_image_like(native)
        phrases = [f"Histopathology shows {concept}." for concept in concepts]
        text = self.tokenize(self.tokenizer, phrases).to(self.device)
        with self.torch.inference_mode():
            text_features = self.model.encode_text(text, normalize=True)
            native_scores = self._image_embedding(native) @ text_features.T
            null_scores = self._image_embedding(null) @ text_features.T
        return (native_scores - null_scores).squeeze(0).float().cpu().numpy()
