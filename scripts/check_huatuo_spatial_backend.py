"""Technical canary for HuatuoGPT-Vision native spatial evidence.

This does not read benchmark references or claim medical improvement. It checks
that a synthetic geometry packet reaches Huatuo's native mm_projector and
changes the receiver score vector while all model weights remain frozen.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from merit_feddg.capabilities import EvidenceItem
from merit_feddg.generalist_factory import load_generalist, resolve_generalist_spec
from merit_feddg.io import load_experiment_yaml
from merit_feddg.spatial_evidence import encode_soft_mask
from merit_feddg.experts.base import load_rgb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument(
        "--config",
        default="configs/matched_bard_huatuo.yaml",
    )
    parser.add_argument(
        "--prompt",
        default="Describe the image briefly using only visible information.",
    )
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    config = load_experiment_yaml(args.config)
    spec = resolve_generalist_spec(config["generalist"])
    if spec.get("backend") != "huatuo_vision":
        raise ValueError("canary requires backend=huatuo_vision")
    probe = load_generalist(spec, args.artifacts)
    if not getattr(probe.tensor_bridge, "training_free", False):
        raise RuntimeError("Huatuo spatial bridge is not training-free")
    if list(probe.tensor_bridge.parameters()):
        raise RuntimeError("Huatuo spatial bridge unexpectedly has trainable parameters")
    if any(parameter.requires_grad for parameter in probe.model.parameters()):
        raise RuntimeError("Huatuo model must be frozen before the spatial canary")

    image = load_rgb(args.image)
    height, width = image.height, image.width
    mask = np.ones((height, width), dtype=np.float32)
    item = EvidenceItem(
        evidence_id="technical-canary:full-image-mask",
        expert_id="technical_canary",
        capability="segmentation",
        scope="technical_geometry",
        payload={
            "structures": [
                {
                    "label": "technical full-image region",
                    "soft_mask": encode_soft_mask(mask),
                    "mask_coordinate_system": "original_image",
                }
            ]
        },
        provenance={
            "technical_canary": True,
            "target_mask_used": False,
            "target_annotations_used": False,
        },
    )
    packet = probe.tensor_packet(
        (item,),
        args.image,
        question=args.prompt,
        weighting="equal",
    )
    if len(packet) != 1:
        raise RuntimeError(
            f"technical spatial packet was not preserved: records={len(packet)} "
            f"rejected={list(packet.rejected)}"
        )

    plain = probe.new_answer_session(args.image, args.prompt)
    spatial = probe.new_tensor_answer_session(
        args.image,
        args.prompt,
        (item,),
        question=args.prompt,
        weighting="equal",
    )
    plain_scores = plain.next_scores([])
    spatial_scores = spatial.next_scores([])
    if plain_scores.shape != spatial_scores.shape:
        raise RuntimeError("plain/spatial Huatuo vocabularies differ")
    delta = np.abs(spatial_scores - plain_scores)
    audit = dict(getattr(probe.tensor_bridge, "last_audit", {}))
    if audit.get("records") != 1:
        raise RuntimeError("Huatuo mm_projector spatial hook was not exercised")
    if not np.isfinite(delta).all() or float(delta.max()) <= 0:
        raise RuntimeError("Huatuo spatial hook produced no finite receiver-logit change")

    result = {
        "schema": "huatuo-spatial-technical-canary-v1",
        "backend": spec["backend"],
        "checkpoint": spec["checkpoint_path"],
        "source_path": spec["source_path"],
        "image": str(Path(args.image).resolve()),
        "image_size": [width, height],
        "prompt": args.prompt,
        "spatial_records": len(packet),
        "spatial_rejected": list(packet.rejected),
        "bridge_audit": audit,
        "plain_argmax": int(np.argmax(plain_scores)),
        "spatial_argmax": int(np.argmax(spatial_scores)),
        "argmax_changed": bool(np.argmax(plain_scores) != np.argmax(spatial_scores)),
        "max_abs_score_delta": float(delta.max()),
        "mean_abs_score_delta": float(delta.mean()),
        "model_weights_updated": False,
        "references_read": False,
        "medical_correctness_evaluated": False,
    }
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
