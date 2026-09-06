"""Explicit medical-generalist backends; local checkpoints are never redownloaded."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from .extract import _local_or_remote


def resolve_generalist_spec(spec):
    result = dict(spec)
    for key in ("checkpoint_path", "source_path", "vision_tower_path"):
        if result.get(key):
            result[key] = os.path.expanduser(os.path.expandvars(result[key]))
            if "$" in result[key]:
                raise ValueError(f"unresolved environment variable in generalist.{key}")
    return result


def load_generalist(spec, artifacts=None):
    spec = resolve_generalist_spec(spec)
    backend = spec.get("backend", "qwen")
    model_path = spec.get("checkpoint_path") or _local_or_remote(spec["id"], artifacts)
    if backend == "llava_med":
        from .llava_generalist import LlavaMedGeneralist

        return LlavaMedGeneralist(
            model_path, source_path=spec.get("source_path"),
            vision_tower_path=spec.get("vision_tower_path"),
            dtype=spec.get("dtype", "float16"), device_map=spec.get("device_map", "auto"),
            local_files_only=True,
        )
    if backend != "qwen":
        raise ValueError(f"unsupported medical generalist backend: {backend}")
    from .generalist import QwenLayerProbe

    return QwenLayerProbe(
        model_path, layers=[-1], dtype=spec.get("dtype", "bfloat16"),
        device_map=spec.get("device_map", "auto"),
    )


def make_capability_session(probe, image, prompt, config=None):
    from .capability_generation import QwenCapabilitySession

    session = QwenCapabilitySession(probe, image, prompt)
    if config is not None:
        session.control_protocol = config.control_protocol
    return session


def generalist_provenance(spec, artifacts):
    """Include external official source and the separate CLIP checkpoint in cache identity."""
    from .open_study import model_provenance

    spec = resolve_generalist_spec(spec)
    result = model_provenance(spec, artifacts)
    if spec.get("backend") != "llava_med":
        return result
    from .llava_generalist import inspect_llava_checkpoint

    info = inspect_llava_checkpoint(
        spec.get("checkpoint_path", spec["id"]),
        vision_tower_path=spec.get("vision_tower_path"),
    )
    result["vision_tower"] = model_provenance(
        {"id": "llava-visual-dependency", "checkpoint_path": info["vision_tower_path"]}, artifacts
    )
    if spec.get("source_path"):
        source = Path(spec["source_path"]) / "llava"
    else:
        import importlib.util

        module = importlib.util.find_spec("llava")
        if module is None or not module.origin:
            raise FileNotFoundError("LLaVA source_path or installed official llava is required")
        source = Path(module.origin).parent
    if not (source / "model/language_model/llava_mistral.py").is_file():
        raise FileNotFoundError(f"Missing official LLaVA-Med source below {source}")
    result["external_source"] = {
        path.relative_to(source).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(source.rglob("*.py"))
    }
    return result
