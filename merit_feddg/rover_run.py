"""python -m merit_feddg.rover_run --help; offline, source-only, no fitting."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image

from .contribution import answer_metrics
from .generalist_factory import generalist_provenance, load_generalist
from .io import load_yaml
from .open_data import pixel_digest, read_manifest
from .open_study import atomic_json, fingerprint
from .rover import decode_regions, region_boxes


def read_regions(path, rows):
    regions = json.loads(Path(path).read_text(encoding="utf-8"))
    by_id = {r["id"]: r for r in rows}
    for key, item in regions.items():
        if key not in by_id:
            continue
        if set(item) != {"box", "producer", "checkpoint_sha256", "image_sha256", "kind"}:
            raise ValueError("region schema must match documented predicted-region fields")
        if item["kind"] not in {"predicted_anatomy", "predicted_lesion", "predicted_object"}:
            raise ValueError("only predicted regions accepted, not annotation/oracle boxes")
        if not item["producer"] or len(item["checkpoint_sha256"]) != 64:
            raise ValueError("record region producer and checkpoint SHA256")
        if item["image_sha256"] != by_id[key]["image_sha256"]:
            raise ValueError("region/image fingerprint mismatch")
    return regions


def prepare_xrv(args, rows):
    from .experts.native_xrv import XrvCapabilityAdapter

    if not args.xrv_target:
        raise ValueError("--prepare-xrv requires --xrv-target (exact PSPNet target name)")
    path = Path(args.prepare_xrv)
    digest = _file_hash(path)
    model = XrvCapabilityAdapter(str(path), "segmentation", device="auto")
    if args.xrv_target not in model.targets:
        raise ValueError(f"choose --xrv-target from {model.targets}")
    regions, skipped = {}, {}
    for row in rows:
        if row["modality"].casefold() != "cxr":
            skipped[row["id"]] = "not a CXR; no automatic modality override"
            continue
        _, masks, transform = model.segment(row["image"])
        yy, xx = np.where(masks[model.targets.index(args.xrv_target)] >= .5)
        if not len(xx):
            skipped[row["id"]] = "empty predicted anatomy"
            continue
        a, b, c, d = transform["crop_box_xyxy_normalized"]
        height, width = masks.shape[1:]
        box = [a+xx.min()/width*(c-a), b+yy.min()/height*(d-b),
               a+(xx.max()+1)/width*(c-a), b+(yy.max()+1)/height*(d-b)]
        regions[row["id"]] = {"box": [float(x) for x in box],
                              "producer": f"XRV PSPNet:{args.xrv_target}",
                              "checkpoint_sha256": digest, "image_sha256": row["image_sha256"],
                              "kind": "predicted_anatomy"}
    atomic_json(args.regions, regions)
    atomic_json(str(args.regions)+".audit.json", {"skipped": skipped, "count": len(regions)})
    print(f"Prepared {len(regions)} anatomy proposals, {len(skipped)} skipped. No VLM loaded.", flush=True)


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--regions", required=True, help="predicted region JSON, see docs/ROVER_QUICKSTART.md")
    parser.add_argument("--references", help="read only after ALL inference completes")
    parser.add_argument("--config", default="configs/llava_med_capabilities.yaml")
    parser.add_argument("--model-path")
    parser.add_argument("--llava-source")
    parser.add_argument("--vision-tower-path")
    parser.add_argument("--output", default="runs/rover-source")
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=24)
    parser.add_argument("--weight", type=float, default=.5)
    parser.add_argument("--prepare-xrv", metavar="CHECKPOINT", help="prepare anatomy regions and exit")
    parser.add_argument("--xrv-target")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if min(args.limit, args.max_tokens) < 1 or not 0 <= args.weight <= 1:
        parser.error("positive budgets and weight in [0,1] required")
    rows = read_manifest(args.source_manifest, "source")
    for row in rows:
        if pixel_digest(row["image"]) != row["image_sha256"]:
            raise ValueError(f"image fingerprint mismatch: {row['id']}")
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError("duplicate IDs")
    if args.prepare_xrv:
        prepare_xrv(args, rows)
        return
    regions = read_regions(args.regions, rows)
    selected, skipped, seen_groups, seen_pixels = [], {}, set(), set()
    for row in sorted(rows, key=lambda r: r["id"]):
        if row["id"] not in regions:
            skipped[row["id"]] = "missing predicted region"
            continue
        try:
            with Image.open(row["image"]) as image:
                boxes = region_boxes(regions[row["id"]]["box"], image.size)
        except ValueError as exc:
            skipped[row["id"]] = str(exc)
            continue
        if row["group_id"] in seen_groups or row["image_sha256"] in seen_pixels:
            skipped[row["id"]] = "duplicate independent image/patient group"
            continue
        seen_groups.add(row["group_id"])
        seen_pixels.add(row["image_sha256"])
        selected.append((row, boxes))
    selected = selected[:args.limit]
    if not selected:
        raise ValueError(f"no eligible regions: {skipped}")
    print(f"Eligible selected={len(selected)}, skipped={len(skipped)}", flush=True)
    if args.check_only:
        print(json.dumps({"ids": [r["id"] for r, _ in selected], "skipped": skipped}, indent=2))
        return
    config = load_yaml(args.config)
    spec = config["generalist"].copy()
    for option, field in (("model_path", "checkpoint_path"), ("llava_source", "source_path"),
                          ("vision_tower_path", "vision_tower_path")):
        if getattr(args, option):
            spec[field] = getattr(args, option)
    if spec.get("backend") != "llava_med":
        raise ValueError("pilot requires LLaVA-Med")
    spec["deterministic_image_padding"] = True
    provenance = generalist_provenance(spec, "artifacts")
    code = {str(p.relative_to(Path(__file__).parent)): _file_hash(p)
            for p in Path(__file__).parent.rglob("*.py")}
    settings = {"model": provenance, "code": code, "tokens": args.max_tokens,
                "weight": args.weight, "suffix": config.get("prompt_suffix", ""),
                "input": "single crop per visual branch; original branch always retained"}
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    probe, results = None, []
    for row, boxes in selected:
        identity = fingerprint({"settings": settings, "row": row, "boxes": boxes,
                                "region": regions[row["id"]]})
        cache = out / "cache" / f"{identity}.json"
        if cache.exists():
            results.append(json.loads(cache.read_text(encoding="utf-8")))
            print(f"Cache hit {row['id']}", flush=True)
            continue
        if probe is None:
            probe = load_generalist(spec)
        with Image.open(row["image"]) as image:
            original = image.convert("RGB")
        views = {"base": original, **{name: original.crop(box) for name, box in boxes.items()}}
        view_dir = out / "views" / identity
        view_dir.mkdir(parents=True, exist_ok=True)
        for name, view in views.items():
            view.save(view_dir / f"{name}.png")
        prompt = row["question"] + "\n" + settings["suffix"]
        sessions = {name: probe.new_answer_session(view, prompt) for name, view in views.items()}
        for session in sessions.values():
            probe._validate_context(session.inputs, args.max_tokens)
        answers = {}
        for mode in ("base", "crop", "control", "fusion", "rover"):
            start = perf_counter()
            answers[mode] = decode_regions(sessions, mode, max_tokens=args.max_tokens, weight=args.weight)
            answers[mode]["seconds"] = perf_counter()-start
            print(f"{row['id']} {mode}: {answers[mode]['text']}", flush=True)
        production = probe.generate_with_usage(original, prompt, max_new_tokens=args.max_tokens)
        if production["token_ids"] != answers["base"]["token_ids"]:
            raise RuntimeError("production/Base replay token parity failed; stop before interpreting gains")
        result = {"id": row["id"], "domain": row["domain"], "group_id": row["group_id"],
                  "image_sha256": row["image_sha256"], "region": regions[row["id"]],
                  "boxes_pixels": boxes, "answers": answers, "production_parity": True,
                  "fingerprint": identity}
        cache.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(cache, result)
        results.append(result)
        del sessions
        gc.collect()
    report = {"settings": settings, "cases": results, "skipped": skipped,
              "target_generations": 0, "domain_generalization_verified": False,
              "note": "Fixed source-only heuristic pilot; anatomy is not lesion evidence. "
                      "Replay decoder is not optimized. Token-F1 is not clinical factuality."}
    # Reference files are deliberately unavailable to generation and selection.
    if args.references:
        references = json.loads(Path(args.references).read_text(encoding="utf-8"))
        scores = {}
        for mode in ("base", "crop", "control", "fusion", "rover"):
            values = []
            for row in results:
                ref = references[row["id"]]
                if isinstance(ref, str):
                    ref = [ref]
                values.append(answer_metrics(row["answers"][mode]["text"], ref)["token_f1"])
            scores[mode] = values
        report["metrics"] = {}
        for mode, values in scores.items():
            delta = np.asarray(values)-np.asarray(scores["base"])
            report["metrics"][mode] = {"token_f1": float(np.mean(values)),
                "paired_gain": float(delta.mean()), "lexical_improved": int((delta>0).sum()),
                "lexical_harmed": int((delta<0).sum())}
    atomic_json(out / "result.json", report)
    lines = ["# ROVER source pilot", "", report["note"], "", f"Cases: {len(results)}; target: 0.", ""]
    for mode, metrics in report.get("metrics", {}).items():
        lines.append(f"- {mode}: {metrics}")
    lines += ["", "Inspect views/ and per-token traces in result.json before judging medical benefit."]
    (out / "result.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(f"Saved {out / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
