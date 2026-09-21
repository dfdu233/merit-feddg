"""Frozen MONET dermatology image-text claim verifier.

MONET is a dermatology image-text foundation model published in Nature Medicine
2024.  The official release exposes Hugging Face zero-shot image
classification.  MERIT-Tx uses only candidate-specific image-text scores and
never interprets them as calibrated diagnosis probabilities.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .base import ConceptExpert, load_rgb, null_image_like


class MonetConceptExpert(ConceptExpert):
    def __init__(self, model_id: str, *, device: str = "auto", dtype: str = "auto"):
        try:
            import torch
            from transformers import AutoModelForZeroShotImageClassification, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "MONET requires the existing torch/transformers research environment"
            ) from exc
        source = Path(model_id).expanduser().resolve()
        if not source.is_dir():
            raise FileNotFoundError(f"MONET checkpoint directory is missing: {source}")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if dtype == "auto":
            dtype = "float16" if str(device).startswith("cuda") else "float32"
        if dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("MONET dtype must be float32, float16, bfloat16 or auto")
        self.torch = torch
        self.device = torch.device(device)
        self.dtype = getattr(torch, dtype)
        self.processor = AutoProcessor.from_pretrained(
            source,
            local_files_only=True,
            trust_remote_code=False,
        )
        self.model = AutoModelForZeroShotImageClassification.from_pretrained(
            source,
            local_files_only=True,
            trust_remote_code=False,
        ).to(device=self.device, dtype=self.dtype).eval().requires_grad_(False)

    @staticmethod
    def _claim_phrase(value):
        clean = " ".join(str(value).strip().split())
        lowered = clean.casefold()
        if lowered.startswith(
            (
                "the image ",
                "this image ",
                "the skin ",
                "the dermatology ",
                "dermatology image ",
                "clinical photograph ",
            )
        ):
            return clean if clean.endswith(".") else f"{clean}."
        return f"Dermatology image showing {clean.rstrip('.')}."

    def image_null_scores(
        self,
        image: str | Path | Image.Image,
        prompt: str,
        concepts: list[str],
    ) -> np.ndarray:
        del prompt
        if not concepts:
            raise ValueError("MONET claim scoring requires at least one proposition")
        native = load_rgb(image)
        null = null_image_like(native)
        phrases = [self._claim_phrase(value) for value in concepts]
        inputs = self.processor(
            images=[native, null],
            text=phrases,
            padding=True,
            truncation=False,
            return_tensors="pt",
        )
        values = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }
        if "pixel_values" in values:
            values["pixel_values"] = values["pixel_values"].to(dtype=self.dtype)
        with self.torch.inference_mode():
            outputs = self.model(**values)
        logits = outputs.logits_per_image.detach().float().cpu().numpy()
        if logits.shape != (2, len(phrases)) or not np.isfinite(logits).all():
            raise ValueError("MONET returned invalid image-text logits")
        return np.asarray(logits[0] - logits[1], dtype=np.float32)

    def close(self):
        self.model = None
        self.processor = None
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
