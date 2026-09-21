"""Frozen MUSK pathology image-text claim verifier.

The implementation follows the official Nature 2025 MUSK zero-shot interface:
timm.create_model("musk_large_patch16_384"), the released SentencePiece text
tokenizer, and with_head=True, out_norm=True, ms_aug=False for aligned
image/text embeddings.

MUSK similarity is not a calibrated diagnosis probability. MERIT-Tx grants no
commit authority until the exact source modality/task/claim cell passes v3
qualification and portfolio selection.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from .base import ConceptExpert, load_rgb, null_image_like


class MuskConceptExpert(ConceptExpert):
    def __init__(
        self,
        model_id: str,
        *,
        source_path: str,
        device: str = "auto",
        dtype: str = "auto",
    ) -> None:
        try:
            import torch
            import torchvision
            from timm.data.constants import (
                IMAGENET_INCEPTION_MEAN,
                IMAGENET_INCEPTION_STD,
            )
            from timm.models import create_model
            from transformers import XLMRobertaTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "MUSK requires the official MUSK research environment dependencies"
            ) from exc

        source = Path(source_path).expanduser().resolve()
        if not (source / "musk/modeling.py").is_file():
            raise FileNotFoundError(
                f"MUSK source_path does not contain musk/modeling.py: {source}"
            )
        tokenizer_path = source / "musk/models/tokenizer.spm"
        if not tokenizer_path.is_file():
            raise FileNotFoundError(f"MUSK tokenizer is missing: {tokenizer_path}")
        checkpoint = Path(model_id).expanduser().resolve()
        if checkpoint.is_dir():
            checkpoint = checkpoint / "model.safetensors"
        if not checkpoint.is_file():
            raise FileNotFoundError(
                "MUSK is gated; download xiangjx/musk after accepting its terms. "
                f"Missing checkpoint: {checkpoint}"
            )
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
        # Importing modeling registers musk_large_patch16_384 with timm.
        importlib.import_module("musk.modeling")
        utils = importlib.import_module("musk.utils")

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if dtype == "auto":
            dtype = "float16" if str(device).startswith("cuda") else "float32"
        if dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("MUSK dtype must be float32, float16, bfloat16 or auto")

        self.torch = torch
        self.device = torch.device(device)
        self.dtype = getattr(torch, dtype)
        self.utils = utils
        self.model = create_model("musk_large_patch16_384")
        utils.load_model_and_may_interpolate(
            str(checkpoint),
            self.model,
            "model|module",
            "",
        )
        self.model = (
            self.model.to(device=self.device, dtype=self.dtype)
            .eval()
            .requires_grad_(False)
        )
        self.tokenizer = XLMRobertaTokenizer(str(tokenizer_path))
        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(384, interpolation=3, antialias=True),
                torchvision.transforms.CenterCrop((384, 384)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=IMAGENET_INCEPTION_MEAN,
                    std=IMAGENET_INCEPTION_STD,
                ),
            ]
        )

    @staticmethod
    def _phrase(value):
        clean = " ".join(str(value).strip().split())
        lowered = clean.casefold()
        if lowered.startswith(
            (
                "the image ",
                "this image ",
                "histopathology ",
                "the histopathology ",
            )
        ):
            return clean if clean.endswith(".") else f"{clean}."
        return f"Histopathology image showing {clean.rstrip('.')}."

    def _image_embedding(self, image):
        tensor = self.transform(load_rgb(image)).unsqueeze(0)
        tensor = tensor.to(device=self.device, dtype=self.dtype)
        with self.torch.inference_mode():
            embedding = self.model(
                image=tensor,
                with_head=True,
                out_norm=True,
                ms_aug=False,
                return_global=True,
            )[0]
        if embedding.ndim != 2 or embedding.shape[0] != 1:
            raise ValueError("MUSK returned an invalid image embedding")
        return embedding

    def _text_embeddings(self, concepts):
        phrases = [self._phrase(value) for value in concepts]
        encoded = [
            self.utils.xlm_tokenizer(text, self.tokenizer, max_len=100)
            for text in phrases
        ]
        ids = self.torch.tensor(
            [value[0] for value in encoded],
            dtype=self.torch.long,
            device=self.device,
        )
        padding = self.torch.tensor(
            [value[1] for value in encoded],
            dtype=self.torch.bool,
            device=self.device,
        )
        with self.torch.inference_mode():
            embedding = self.model(
                text_description=ids,
                padding_mask=padding,
                with_head=True,
                out_norm=True,
                ms_aug=False,
                return_global=True,
            )[1]
        if embedding.ndim != 2 or embedding.shape[0] != len(concepts):
            raise ValueError("MUSK returned invalid text embeddings")
        return embedding

    def image_null_scores(
        self,
        image: str | Path | Image.Image,
        prompt: str,
        concepts: list[str],
    ) -> np.ndarray:
        del prompt
        if not concepts:
            raise ValueError("MUSK claim scoring requires at least one proposition")
        native = load_rgb(image)
        null = null_image_like(native)
        image_embeddings = self.torch.cat(
            (self._image_embedding(native), self._image_embedding(null)),
            dim=0,
        )
        text = self._text_embeddings(concepts)
        scores = (image_embeddings @ text.T).detach().float().cpu().numpy()
        if scores.shape != (2, len(concepts)) or not np.isfinite(scores).all():
            raise ValueError("MUSK returned invalid image-text scores")
        return np.asarray(scores[0] - scores[1], dtype=np.float32)

    def close(self):
        self.model = None
        self.tokenizer = None
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
