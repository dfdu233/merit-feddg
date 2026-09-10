"""Independent typed encoders + set reader + gated visual-token residual.

Architectural references (not copied source): Prismer's expert resampler;
Osprey's mask pooling; ProxyCLIP's separation of spatial weights and content.
The learned gate is NOT a reliability estimate or a clinical safety guarantee.
"""

from contextlib import contextmanager
from dataclasses import asdict, replace

import torch
from torch import nn

from .tensor_evidence import TensorContract


class NativeTensorBridge(nn.Module):
    def __init__(self, contract, visual_dim, width=128, queries=16, heads=4):
        super().__init__()
        if any(type(v) is not int or v < 1 for v in (visual_dim, width, queries, heads)):
            raise ValueError("positive bridge dimensions required")
        if width % heads:
            raise ValueError("bridge width must be divisible by heads")
        self.contract = contract
        self.config = {"visual_dim": visual_dim, "width": width, "queries": queries, "heads": heads}
        self.numeric_encoders = nn.ModuleList([
            nn.Sequential(nn.Linear(10, width), nn.GELU(), nn.Linear(width, width))
            for _ in range(3)
        ])
        self.concepts = nn.Embedding(len(contract.concepts), width)
        self.scopes = nn.Embedding(len(contract.scopes), width)
        self.semantics = nn.Embedding(4, width)
        self.shapes = nn.Linear(contract.grid_size**2, width, bias=False)
        self.visual = nn.Linear(visual_dim, width)
        self.queries = nn.Parameter(torch.randn(1, queries, width) * 0.02)
        self.reader = nn.MultiheadAttention(width, heads, batch_first=True, dropout=0)
        self.inject = nn.MultiheadAttention(width, heads, batch_first=True, dropout=0)
        self.norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, visual_dim, bias=False)
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, visual, packet):
        """Learned fusion strength is distinct from runtime evidence admission."""
        self._validate_visual(visual)
        if not len(packet):
            return visual
        if not self.training and self.gate.detach().item() == 0:
            return visual
        evidence = self.encode_evidence(visual, packet)
        return self.fuse_evidence(visual, evidence)

    def _validate_visual(self, visual):
        if visual.ndim != 3 or visual.shape != (
            1, self.contract.grid_size**2, self.config["visual_dim"]
        ):
            raise ValueError("tensor bridge requires one square patch-only image grid")

    def encode_evidence(self, visual, packet):
        """Unified [1, records, width] interface for typed native observations.

        No expert-specific hidden vectors or generated report text are required.
        The returned vectors require this trained bridge's weights and contract.
        Source identity remains in packet.sources for audit, not embeddings.
        """
        self._validate_visual(visual)
        device, dtype = self.gate.device, self.gate.dtype
        if not len(packet):
            return torch.empty((1, 0, self.config["width"]), device=device, dtype=dtype)
        values = torch.as_tensor(packet.numeric, device=device, dtype=dtype)
        regions = torch.as_tensor(packet.regions, device=device, dtype=dtype)
        ids = [torch.as_tensor(x, device=device, dtype=torch.long) for x in (
            packet.kind, packet.concept, packet.scope, packet.semantics
        )]
        h = self.visual(visual.to(device=device, dtype=dtype))
        # R H: region weights from the expert, visual content from the VLM.
        weights = regions / regions.sum(-1, keepdim=True).clamp_min(torch.finfo(dtype).tiny)
        pooled = weights @ h[0]
        encoded = torch.stack([encoder(values) for encoder in self.numeric_encoders])
        encoded = encoded[ids[0], torch.arange(len(packet), device=device)]
        encoded = (encoded + self.concepts(ids[1]) + self.scopes(ids[2])
                   + self.semantics(ids[3]) + self.shapes(regions) + pooled)
        return self.norm(encoded).unsqueeze(0)

    def fuse_evidence(self, visual, evidence):
        """Fuse only admitted vectors; an empty set is an exact identity."""
        self._validate_visual(visual)
        if evidence.ndim != 3 or evidence.shape[0] != 1 or evidence.shape[2] != self.config["width"]:
            raise ValueError("evidence must have shape [1, records, bridge width]")
        if not evidence.shape[1]:
            return visual
        h = self.visual(visual.to(device=self.gate.device, dtype=self.gate.dtype))
        z, _ = self.reader(self.queries, evidence, evidence, need_weights=False)
        delta, _ = self.inject(h, z, z, need_weights=False)
        residual = self.gate.tanh() * self.output(delta)
        return visual + residual.to(device=visual.device, dtype=visual.dtype)

    def save(self, path, *, training_steps, source_domains, base_identity=None):
        if type(training_steps) is not int or training_steps < 1 or not source_domains:
            raise ValueError("checkpoint requires nonzero training and source-domain provenance")
        torch.save({"version": 1, "contract": asdict(self.contract), "config": self.config,
                    "state_dict": self.state_dict(), "training_steps": training_steps,
                    "source_domains": sorted(set(source_domains)), "source_only": True,
                    "base_identity": base_identity}, path)

    def bind_expert(self, bindings):
        """Extend native-name aliases only; new concepts/scopes still need training.

        This changes no weights. Include the extended contract in saved experiment
        provenance. A shared contract does not establish equal expert calibration.
        """
        self.contract = replace(self.contract, bindings=(*self.contract.bindings, *bindings))

    @classmethod
    def load(cls, path):
        value = torch.load(path, map_location="cpu", weights_only=True)
        if (value.get("version") != 1 or value.get("source_only") is not True
                or type(value.get("training_steps")) is not int or value["training_steps"] < 1
                or not value.get("source_domains")):
            raise ValueError("not a trained source-only tensor bridge checkpoint")
        model = cls(TensorContract.from_dict(value["contract"]), **value["config"])
        model.base_identity = value.get("base_identity")
        model.load_state_dict(value["state_dict"], strict=True)
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise ValueError("non-finite tensor bridge weights")
        return model.eval()


def validate_tensor_backend(probe, bridge):
    """Only the explicitly matched preprocessing geometry is supported."""
    tower = probe.model.get_vision_tower()
    processor = probe.image_processor
    grid = bridge.contract.grid_size
    crop = getattr(processor, "crop_size", {})
    size = getattr(processor, "size", {})
    if (not getattr(probe, "deterministic_image_padding", False)
            or getattr(probe.model.config, "image_aspect_ratio", None) != "pad"
            or getattr(tower, "select_feature", None) != "patch"
            or int(tower.num_patches) != grid**2
            or not isinstance(crop, dict) or crop.get("height") != crop.get("width")
            or not isinstance(size, dict) or size.get("shortest_edge") != crop.get("height")
            or not getattr(processor, "do_center_crop", False)
            or not getattr(processor, "do_resize", False)
            or getattr(probe.model.config, "mm_patch_merge_type", "flat") != "flat"):
        raise ValueError("tensor evidence requires deterministic square pad + matched CLIP patch grid")
    if int(probe.model.config.hidden_size) != bridge.config["visual_dim"]:
        raise ValueError("bridge checkpoint does not match VLM hidden size")


@contextmanager
def tensor_projector_context(probe, packet):
    """Scoped hook shared by real generation and differentiable teacher forcing.

    A probe is single-request, like the existing generation backend. Reentrant
    tensor calls are rejected; hooks are always removed even after an exception.
    """
    if packet is None or not len(packet):
        yield
        return
    bridge = probe.tensor_bridge
    validate_tensor_backend(probe, bridge)
    if getattr(probe, "_tensor_hook_active", False):
        raise RuntimeError("tensor evidence probe cannot serve concurrent/reentrant requests")
    probe._tensor_hook_active = True
    calls = 0

    def inject(_module, _inputs, output):
        nonlocal calls
        calls += 1
        return bridge(output, packet)

    handle = None
    try:
        handle = probe.model.get_model().mm_projector.register_forward_hook(inject)
        yield
        if calls == 0:
            raise RuntimeError("VLM did not execute its visual projector; no evidence was injected")
    finally:
        if handle is not None:
            handle.remove()
        probe._tensor_hook_active = False
