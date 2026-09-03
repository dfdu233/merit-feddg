from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .experts.base import load_rgb


class RadDinoPatchValidator:
    """Frozen CXR patch encoder for offline occlusion/region validation only."""

    def __init__(self, model_id: str = "microsoft/rad-dino", device: str = "auto") -> None:
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as exc:
            raise RuntimeError("install merit-feddg[research] to load RAD-DINO") from exc
        self.torch = torch
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(self.device).eval()

    def patch_features(self, image: str | Path | Image.Image) -> np.ndarray:
        inputs = self.processor(images=load_rgb(image), return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            output = self.model(**inputs)
        return output.last_hidden_state[:, 1:, :].float().cpu().numpy()


def occlusion_sensitivity(
    score_function,
    image: Image.Image,
    grid: int = 8,
    fill: int = 127,
) -> np.ndarray:
    """Finite-difference patch audit; avoids treating attention as causality."""

    if grid <= 0:
        raise ValueError("grid must be positive")
    native = load_rgb(image)
    baseline = float(score_function(native))
    width, height = native.size
    heatmap = np.zeros((grid, grid), dtype=float)
    for row in range(grid):
        for column in range(grid):
            left = round(column * width / grid)
            right = round((column + 1) * width / grid)
            top = round(row * height / grid)
            bottom = round((row + 1) * height / grid)
            occluded = native.copy()
            occluded.paste((fill, fill, fill), (left, top, right, bottom))
            heatmap[row, column] = baseline - float(score_function(occluded))
    return heatmap
