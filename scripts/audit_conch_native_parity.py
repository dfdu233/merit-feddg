"""Compare the current CONCH adapter with direct official image/text encoding.

No task labels, catalog change, null-image subtraction, or actor modification.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from merit_feddg.pathology_pilot import file_sha, write_new  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--expert", default="conch_tissue")
    p.add_argument("--artifacts", default="artifacts")
    p.add_argument("--gpu-uuid", required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError("fresh audit path required")
    import numpy as np
    import torch
    import yaml
    from PIL import Image

    from merit_feddg.capabilities import CapabilityRequest
    from merit_feddg.capability_experts import CapabilityPool, _catalog
    from run_quilt_worker import assert_single_gpu

    assert_single_gpu(torch, args.gpu_uuid)
    specs = yaml.safe_load(args.config.read_text())["experts"]
    spec = specs[args.expert]
    if spec.get("adapter") not in {"conch", "contrastive_conch"}:
        raise ValueError("expected existing CONCH native adapter")
    pool = CapabilityPool({args.expert: spec}, args.artifacts)
    request = CapabilityRequest("parity", str(args.image), "", "pathology", "open_vqa",
                                "source", "parity", "classification", scope=spec["scope"])
    result = pool.infer(args.expert, request)
    if len(result.items) != 1:
        raise ValueError("expected one native catalog result")
    model = pool._model(spec)
    catalog = _catalog(spec)
    with Image.open(args.image) as im:
        rgb = im.convert("RGB")
    with torch.inference_mode():
        pixels = model.preprocess(rgb).unsqueeze(0).to(model.device)
        image = model.model.encode_image(pixels, proj_contrast=True, normalize=True)
        tokens = model.tokenize(model.tokenizer, [prompt for _, prompt in catalog]).to(model.device)
        text = model.model.encode_text(tokens, normalize=True)
        direct = (image @ text.T).squeeze(0).float().cpu().numpy()
    observed = {e["concept"]: e["similarity"] for e in result.items[0].payload["catalog"]}
    actual = np.asarray([observed[name] for name, _ in catalog])
    delta = float(np.max(np.abs(direct - actual)))
    passed = bool(np.allclose(direct, actual, rtol=1e-4, atol=1e-5))
    write_new(args.output, {"passed": passed, "max_abs_delta": delta,
              "rtol": 1e-4, "atol": 1e-5, "catalog": list(catalog),
              "direct": direct.tolist(), "adapter": actual.tolist(),
              "image_file_sha256": file_sha(args.image), "config_sha256": file_sha(args.config),
              "note": "same checkpoint/API parity, not task accuracy or independent checkpoint authenticity"})
    if not passed:
        raise RuntimeError("native parity failed; inspect loading/precision/transform before replacing CONCH")


if __name__ == "__main__":
    main()
