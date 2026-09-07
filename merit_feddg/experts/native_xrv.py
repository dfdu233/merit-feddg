"""Local-only TorchXRayVision native capabilities for chest radiographs.

Uses the official DenseNet architecture and ChestX-Det PSPNet architecture. The
adapter never downloads weights. DenseNet's official legacy checkpoint contains
a pickled model: only load a trusted official checkpoint, optionally with a pinned
SHA-256. These outputs are research observations, not clinical diagnoses.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .base import load_rgb


def _checkpoint(path, sha256=None):
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(
            f"TorchXRayVision needs an existing checkpoint_path file: {resolved}. "
            "Inference does not download weights."
        )
    if sha256:
        digest = hashlib.sha256()
        with resolved.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest().lower() != str(sha256).lower():
            raise ValueError("TorchXRayVision checkpoint SHA-256 mismatch")
    return resolved


class XrvCapabilityAdapter:
    def __init__(
        self,
        checkpoint_path,
        capability,
        device="auto",
        weights="densenet121-res224-all",
        sha256=None,
    ):
        path = _checkpoint(checkpoint_path, sha256)
        if capability not in {"classification", "segmentation"}:
            raise ValueError("XRV supports classification or anatomical segmentation")
        try:
            import torch
            import torchxrayvision as xrv
        except ImportError as exc:
            raise RuntimeError("Install the optional torchxrayvision==1.5.4 dependency") from exc
        self.torch, self.xrv = torch, xrv
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device, self.capability = torch.device(device), capability
        if capability == "classification":
            if weights not in xrv.models.model_urls or not str(weights).startswith("densenet"):
                raise ValueError("XRV weights must name an official DenseNet checkpoint variant")
            metadata = xrv.models.model_urls[weights]
            self.targets = tuple(metadata["labels"])
            self.model = xrv.models.DenseNet(weights=None, apply_sigmoid=True)
            # Official release serializes an nn.Module, not a tensor-only state_dict.
            # Never obtain this path from tool-generated text or a retrieved case.
            saved = torch.load(path, map_location="cpu", weights_only=False)
            if hasattr(saved, "modules"):
                for child in saved.modules():
                    if not hasattr(child, "_non_persistent_buffers_set"):
                        child._non_persistent_buffers_set = set()
            state = saved.state_dict() if hasattr(saved, "state_dict") else saved
            self.model.load_state_dict(state, strict=True)
            self.model.targets = list(self.targets)
            self.model.input_resolution = int(metadata["input_resolution"])
            self.model.op_threshs = None  # raw sigmoid, NOT operating-point normalized scores
            self.resolution = self.model.input_resolution
        else:
            from torchxrayvision.baseline_models.chestx_det import PSPNet
            from torchxrayvision.baseline_models.chestx_det.ptsemseg.pspnet import pspnet

            self.targets = tuple(PSPNet.targets)
            self.model = pspnet(len(self.targets))
            state = torch.load(path, map_location="cpu", weights_only=True)
            state = state.get("state_dict", state)
            state = {key.removeprefix("module."): value for key, value in state.items()}
            self.model.load_state_dict(state, strict=True)
            self.resolution = 512
        self.model = self.model.to(self.device).eval()

    def _inputs(self, image):
        rgb = load_rgb(image)
        width, height = rgb.size
        side = min(width, height)
        left, top = width // 2 - side // 2, height // 2 - side // 2
        # Match XRV's documented center-crop + resizer preprocessing, retaining
        # the crop transform so masks never claim to cover unseen image margins.
        gray = np.asarray(rgb.convert("L"), dtype=np.float32)
        gray = gray[top : top + side, left : left + side]
        normalized = self.xrv.datasets.normalize(gray, 255)[None, ...]
        resized = self.xrv.datasets.XRayResizer(self.resolution)(normalized)
        tensor = self.torch.from_numpy(np.asarray(resized, dtype=np.float32))[None].to(self.device)
        transform = {
            "original_size_hw": [height, width],
            "crop_box_xyxy_normalized": [
                left / width, top / height, (left + side) / width, (top + side) / height
            ],
            "crop_size_hw": [side, side],
            "model_size_hw": [self.resolution, self.resolution],
            "outside_crop": "unknown",
        }
        return tensor, transform

    def classify(self, image):
        tensor, transform = self._inputs(image)
        with self.torch.inference_mode():
            scores = self.model(tensor).detach().float().cpu().reshape(-1).numpy()
        if scores.size != len(self.targets) or not np.isfinite(scores).all():
            raise ValueError("XRV classification produced invalid scores")
        return self.targets, scores, transform

    def domain_embedding(self, image):
        """Frozen DenseNet pre-classifier features, not sigmoid finding scores."""
        if self.capability != "classification":
            raise NotImplementedError("XRV anatomy has no validated native embedding adapter yet")
        tensor, _ = self._inputs(image)
        with self.torch.inference_mode():
            features = self.model.features(tensor)
            vector = self.torch.relu(features).mean(dim=(-2, -1))[0]
        return vector.detach().float().cpu().numpy()

    def segment(self, image):
        tensor, transform = self._inputs(image)
        tensor = ((tensor + 1024) / 2048).repeat(1, 3, 1, 1)
        means = tensor.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
        stds = tensor.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
        with self.torch.inference_mode():
            logits = self.model((tensor - means) / stds)
            # Anatomical structures can overlap; independent sigmoid, not argmax.
            probabilities = self.torch.sigmoid(logits)[0].detach().float().cpu().numpy()
        if probabilities.shape != (len(self.targets), 512, 512):
            raise ValueError("XRV anatomical segmentation shape mismatch")
        if not np.isfinite(probabilities).all():
            raise ValueError("XRV anatomical segmentation produced nonfinite values")
        return self.targets, probabilities, transform
