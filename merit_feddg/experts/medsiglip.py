"""Frozen MedSigLIP image-text claim verifier.

The adapter follows the official Google MedSigLIP Hugging Face interface:
AutoModel + AutoProcessor with logits_per_image.  It never trains a probe or
interprets similarity as a calibrated diagnosis probability.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .base import ConceptExpert, load_rgb, null_image_like


class MedSiglipConceptExpert(ConceptExpert):
    def __init__(self, model_id: str, device: str = "auto", dtype: str = "auto") -> None:
        try:
            import torch
            from transformers import AutoModel, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "MedSigLIP requires the existing torch/transformers research environment"
            ) from exc
        source = Path(model_id).expanduser().resolve()
        if not source.is_dir():
            raise FileNotFoundError(
                "MedSigLIP is gated and must be downloaded explicitly after accepting "
                f"its upstream terms: {source}"
            )
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if dtype == "auto":
            dtype = "bfloat16" if str(device).startswith("cuda") else "float32"
        if dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("MedSigLIP dtype must be float32, float16, bfloat16 or auto")

        self.torch = torch
        self.device = torch.device(device)
        self.dtype = getattr(torch, dtype)
        self.processor = AutoProcessor.from_pretrained(
            source,
            local_files_only=True,
            trust_remote_code=False,
        )
        self.model = AutoModel.from_pretrained(
            source,
            local_files_only=True,
            trust_remote_code=False,
        ).to(device=self.device, dtype=self.dtype).eval().requires_grad_(False)

    @staticmethod
    def _claim_phrase(value: str) -> str:
        clean = " ".join(str(value).strip().split())
        lowered = clean.casefold()
        if lowered.startswith(
            (
                "the image ",
                "this image ",
                "the radiograph ",
                "the scan ",
                "histopathology ",
                "a medical image ",
            )
        ):
            return clean if clean.endswith(".") else f"{clean}."
        return f"A medical image showing {clean.rstrip('.')}."

    def _inputs(self, images, texts=None):
        kwargs = {
            "images": list(images),
            "padding": "max_length",
            "return_tensors": "pt",
        }
        if texts is not None:
            kwargs["text"] = list(texts)
        values = self.processor(**kwargs)
        values = {
            key: value.to(self.device)
            for key, value in values.items()
        }
        if "pixel_values" in values:
            values["pixel_values"] = values["pixel_values"].to(dtype=self.dtype)
        return values

    def image_null_scores(
        self,
        image: str | Path | Image.Image,
        prompt: str,
        concepts: list[str],
    ) -> np.ndarray:
        del prompt
        if not concepts:
            raise ValueError("MedSigLIP claim scoring requires at least one proposition")
        native = load_rgb(image)
        null = null_image_like(native)
        phrases = [self._claim_phrase(value) for value in concepts]
        inputs = self._inputs((native, null), phrases)
        with self.torch.inference_mode():
            outputs = self.model(**inputs)
        logits = outputs.logits_per_image.detach().float().cpu().numpy()
        if logits.shape != (2, len(phrases)) or not np.isfinite(logits).all():
            raise ValueError("MedSigLIP returned invalid image-text logits")
        return np.asarray(logits[0] - logits[1], dtype=np.float32)

    def domain_embedding(self, image: str | Path | Image.Image) -> np.ndarray:
        inputs = self._inputs((load_rgb(image),))
        with self.torch.inference_mode():
            vector = self.model.get_image_features(**inputs)
        vector = vector[0].detach().float().cpu().numpy()
        if vector.ndim != 1 or not vector.size or not np.isfinite(vector).all():
            raise ValueError("MedSigLIP returned an invalid image embedding")
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-12:
            raise ValueError("MedSigLIP image embedding has zero norm")
        return vector / norm

    def close(self):
        self.model = None
        self.processor = None
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
