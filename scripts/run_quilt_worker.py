"""Run frozen Quilt-LLaVA in its OWN environment, before loading the generalist.

Uses the official aldraus/quilt-llava loader/tokenization/generation path. Only
local complete checkpoints and an already-populated offline vision cache work.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Import only the dependency-light pilot helpers, never the actor's llava package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from merit_feddg.pathology_pilot import (  # noqa: E402
    SCHEMA, digest, file_sha, prediction_jobs, read_json, write_new,
)


def checkpoint_identity(directory):
    root = Path(directory).expanduser().resolve()
    if not (root / "config.json").is_file():
        raise ValueError("local full checkpoint config missing")
    indices = list(root.glob("*.index.json"))
    required = set()
    for path in indices:
        required.update(read_json(path).get("weight_map", {}).values())
    if any(not (root / name).is_file() for name in required):
        raise ValueError("incomplete checkpoint shards")
    files = [p for p in root.iterdir() if p.is_file()
             and p.suffix in {".json", ".bin", ".safetensors", ".model", ".txt"}]
    if not any(p.suffix in {".bin", ".safetensors"} and "projector" not in p.name for p in files):
        raise ValueError("full merged weights required, not a projector/LoRA alone")
    return {p.name: file_sha(p) for p in sorted(files)}


def assert_single_gpu(torch, uuid):
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one authorized visible CUDA device required")
    observed = getattr(torch.cuda.get_device_properties(0), "uuid", None)
    if observed is None:
        import subprocess
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
            raise RuntimeError("old torch requires verified container CUDA_VISIBLE_DEVICES=0 mapping")
        observed = subprocess.check_output(
            ["nvidia-smi", "-i", "0", "--query-gpu=uuid", "--format=csv,noheader"],
            text=True, timeout=10).strip()
    if str(observed) != uuid:
        raise RuntimeError("GPU UUID does not match authorization")


def suffix_tokens(full, prefix):
    """Official Quilt returns prompt+continuation. Do not guess an unknown layout."""
    if len(full) < len(prefix) or full[:len(prefix)] != prefix:
        raise RuntimeError("unexpected Quilt output layout/prompt echo; stop rather than slice guessed text")
    return full[len(prefix):]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--gpu-uuid", required=True)
    p.add_argument("--precision", choices=("fp16", "8bit", "4bit"), default="4bit")
    p.add_argument("--formal", action="store_true", help="Explicit full-test protocol, never relabelled TRAIN")
    p.add_argument("--canary", action="store_true", help="Formal scheduling stop at frozen engineering cases")
    args = p.parse_args()
    frozen = read_json(args.run / "frozen.json")
    formal = args.formal
    if formal:
        if frozen['schema'] != 'pathology-quilt-formal-v1' or frozen['split'] != 'test':
            raise ValueError('explicit formal TEST identity required')
        if args.precision != frozen['quilt_precision']:
            raise ValueError('frozen precision mismatch')
    elif frozen["schema"] != SCHEMA or frozen["split"] != "train":
        raise ValueError("frozen pathology TRAIN pilot required")
    from run_pathology_quilt_pilot import code_identity
    if frozen["source_code"] != code_identity():
        raise ValueError("implementation changed since prepare")
    final = args.run / "quilt_predictions.json"
    attempt = args.run / "quilt_attempt.json"
    if formal:
        attempt = args.run / ('quilt_attempt_' + str(time.time_ns()) + '.json')
        final = args.run / ('quilt_canary.json' if args.canary else 'quilt_predictions.json')
    if final.exists() or attempt.exists():
        raise FileExistsError("do not overwrite/retry an existing worker attempt")
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ[key] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    source = args.source.expanduser().resolve()
    if not (source / "llava/model/builder.py").is_file():
        raise FileNotFoundError("official Quilt-LLaVA source checkout required")
    if "llava" in sys.modules:
        raise RuntimeError("run in isolated Quilt interpreter, not the actor process")
    sys.path.insert(0, str(source))
    import llava
    import torch
    import transformers
    from huggingface_hub import snapshot_download
    from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import SeparatorStyle, conv_templates
    from llava.mm_utils import KeywordsStoppingCriteria, process_images, tokenizer_image_token
    from llava.model.builder import load_pretrained_model
    from PIL import Image

    if not Path(llava.__file__).resolve().is_relative_to(source):
        raise RuntimeError("wrong llava package imported")
    assert_single_gpu(torch, args.gpu_uuid)
    torch.set_num_threads(4)
    seed = frozen['seed'] if formal else 0
    torch.manual_seed(seed)
    checkpoint = args.checkpoint.expanduser().resolve()
    cfg = read_json(checkpoint / "config.json")
    if cfg.get("architectures") != ["LlavaLlamaForCausalLM"] or cfg.get("model_type") != "llava":
        raise ValueError("Quilt's merged Llama checkpoint required; Mistral/PathGen not interchangeable")
    if cfg.get("mm_vision_tower") != "openai/clip-vit-large-patch14-336":
        raise ValueError("unexpected released Quilt-7B vision tower; inspect before using another model")
    vision_path = snapshot_download(cfg["mm_vision_tower"], local_files_only=True)
    identity = {
        "declared_model": "wisdomik/Quilt-Llava-v1.5-7b",
        "checkpoint": checkpoint_identity(checkpoint), "vision": checkpoint_identity(vision_path),
        "source": {str(f.relative_to(source)): file_sha(f) for f in sorted((source / "llava").rglob("*.py"))},
        "precision": args.precision, "conversation": "llava_v1", "seed": seed,
        "torch": torch.__version__, "transformers": transformers.__version__,
        "worker_sha256": file_sha(__file__), "gpu_uuid": args.gpu_uuid,
        "training_overlap": "not independently ruled out", "training_performed": False,
    }
    jobs = prediction_jobs(frozen)
    if formal and args.canary:
        jobs = [j for j in jobs if j['target_id'] in frozen['canary_ids']]
    for job in jobs:
        if file_sha(job["image"]) != job["image_file_sha256"]:
            raise ValueError("source image changed")
    write_new(attempt, {"identity": digest(frozen), "model": identity, "complete": False})
    records, started = {}, time.perf_counter()
    current_key = None
    try:
        tokenizer, model, processor, context_len = load_pretrained_model(
            str(checkpoint), None, "Quilt-Llava-v1.5-7b",
            load_8bit=args.precision == "8bit", load_4bit=args.precision == "4bit",
            device_map={"": 0}, device="cuda:0")
        model.eval().requires_grad_(False)
        torch.cuda.synchronize()
        load_seconds = time.perf_counter() - started
        torch.cuda.reset_peak_memory_stats()
        log_path = args.run / ('quilt_jobs_' + str(time.time_ns()) + '.jsonl' if formal else 'quilt_jobs.jsonl')
        with log_path.open("x", encoding="utf-8") as log:
            import json
            for job in jobs:
                current_key = job["key"]
                cached_path = args.run / 'quilt-cache' / (current_key + '.json')
                if formal and cached_path.exists():
                    cached = read_json(cached_path)
                    if cached['identity'] != digest(frozen) or cached['model_identity'] != digest(identity) or cached['record']['job'] != job:
                        raise ValueError('formal Quilt cache identity mismatch')
                    records[current_key] = cached['record']
                    continue
                t0 = time.perf_counter()
                with Image.open(job["image"]) as im:
                    image = im.convert("RGB")
                pixels = process_images([image], processor, model.config)
                pixels = ([v.to("cuda:0", dtype=torch.float16) for v in pixels]
                          if isinstance(pixels, list) else pixels.to("cuda:0", dtype=torch.float16))
                token = DEFAULT_IMAGE_TOKEN
                if model.config.mm_use_im_start_end:
                    token = DEFAULT_IM_START_TOKEN + token + DEFAULT_IM_END_TOKEN
                conv = conv_templates["llava_v1"].copy()
                conv.append_message(conv.roles[0], token + "\n" + job["prompt"])
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX,
                                            return_tensors="pt").unsqueeze(0).to("cuda:0")
                patches = model.get_vision_tower().num_patches
                limit = min(context_len, getattr(model.config, "max_position_embeddings", context_len))
                if ids.shape[1] - 1 + patches + frozen["max_new_tokens"] > limit:
                    raise ValueError("no silent context truncation allowed")
                stop = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                with torch.inference_mode():
                    full = model.generate(ids, images=pixels, do_sample=False, num_beams=1,
                                          max_new_tokens=frozen["max_new_tokens"], use_cache=True,
                                          stopping_criteria=[KeywordsStoppingCriteria([stop], tokenizer, ids)])
                tokens = suffix_tokens(full[0].tolist(), ids[0].tolist())
                text = tokenizer.decode(tokens, skip_special_tokens=True).strip()
                if stop and text.endswith(stop):
                    text = text[:-len(stop)].strip()
                if not text:
                    raise ValueError("empty Quilt prediction; no fallback text")
                torch.cuda.synchronize()
                record = {"job": job, "text": text, "token_ids": tokens, "status": "ok",
                          "input_tokens": int(ids.shape[1]), "expanded_input_tokens": int(ids.shape[1] - 1 + patches),
                          "output_tokens": len(tokens), "prompt_sha256": digest(prompt),
                          "hit_max_new_tokens": len(tokens) == frozen["max_new_tokens"],
                          "seconds": time.perf_counter() - t0}
                records[job["key"]] = record
                if formal:
                    if record['hit_max_new_tokens']:
                        raise RuntimeError('formal specialist reached output cap; preserve failure, no truncation fallback')
                    write_new(cached_path, {'identity': digest(frozen), 'model_identity': digest(identity), 'record': record})
                log.write(json.dumps(record, ensure_ascii=False) + "\n")
                log.flush()
                print("QUILT", len(records), "/", len(jobs), flush=True)
        write_new(final, {"schema": frozen['schema'], "identity": digest(frozen), "model": identity,
                          "complete": True, "predictions": records, "model_load_seconds": load_seconds,
                          "actual_model_calls": len(records), "wall_seconds": time.perf_counter() - started,
                          "peak_allocated_bytes": torch.cuda.max_memory_allocated()})
    except Exception as exc:
        failure = 'quilt_failure_' + str(time.time_ns()) + '.json' if formal else 'quilt_failure.json'
        write_new(args.run / failure, {"type": type(exc).__name__, "message": str(exc),
                  "current_job": current_key, "completed_calls": len(records),
                  "wall_seconds": time.perf_counter() - started, "complete": False})
        raise


if __name__ == "__main__":
    main()
