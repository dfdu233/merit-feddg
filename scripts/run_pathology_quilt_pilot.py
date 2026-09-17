"""Bounded TRAIN study: prepare -> isolated Quilt worker -> frozen actor replay.

No new gate, no clinical labels at inference, and no automatic full benchmark.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from merit_feddg.pathology_pilot import (  # noqa: E402
    ARMS, SCHEMA, digest, donor_map, file_sha, prediction_jobs, quilt_item,
    read_json, select_rows, validate_predictions, write_new,
)


ROOT = Path(__file__).resolve().parents[1]


def code_identity():
    paths = list((ROOT / "merit_feddg").rglob("*.py"))
    paths += [ROOT / "scripts" / name for name in (
        "run_pathology_quilt_pilot.py", "run_quilt_worker.py", "evaluate_pathology_quilt_pilot.py")]
    return {str(p.relative_to(ROOT)): file_sha(p) for p in sorted(paths)}


def scorer_identity(root):
    files = sorted((Path(root) / "anchor").rglob("*.py"))
    if not files:
        raise ValueError("existing ANCHOR scorer source required to freeze evaluation")
    return {str(p.relative_to(root)): file_sha(p) for p in files}


def prepare(args):
    from merit_feddg.matched_evaluation import generation_prompt
    from merit_feddg.open_data import pixel_digest

    if args.output.exists():
        raise FileExistsError("fresh output directory required")
    if not all((args.manifest, args.incumbent_run, args.anchor_root)):
        raise ValueError("prepare requires manifest, incumbent-run and anchor-root")
    manifest = args.manifest.resolve()
    if not any(part.lower() == "train" for part in manifest.parts):
        raise ValueError("use the unchanged official data/train/manifest.jsonl; do not re-label test inputs")
    protocol_path = args.incumbent_run.resolve() / "protocol.json"
    old_path = args.incumbent_run.resolve() / (args.incumbent_arm + ".json")
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    old, protocol = read_json(old_path), read_json(protocol_path)
    if protocol.get("shards_complete") is not True or protocol.get("n") != len(rows):
        raise ValueError("complete frozen incumbent protocol required")
    config = protocol["config"]
    if config.get("prompt_contract", "legacy_suffix") not in {"unified-short-v1", "legacy_suffix"}:
        raise ValueError("need existing shared free-text prompt, not CE/OE-specific answer contracts")
    selected, coverage = select_rows(rows, old, limit=args.limit, split=args.split)
    for row in selected:
        image = Path(row["image"]).expanduser().resolve()
        if pixel_digest(image) != row["image_sha256"]:
            raise ValueError("manifest pixel identity mismatch")
        row["image"] = str(image)
        row["image_file_sha256"] = file_sha(image)
        row["prompt"] = generation_prompt(row, config)
        settings = old[row["id"]]["generation_config"]
        if (settings["max_new_tokens"] != 64 or settings.get("evidence_style") != "semantic"
                or not settings.get("compact_native", False)):
            raise ValueError("pilot requires existing 64-token semantic compact baseline")
        if (settings.get("vector_gate", "off") != "off" or settings.get("semantic_spatial", False)
                or settings.get("admission_mode", "legacy") != "legacy"):
            raise ValueError("do not silently disable a gate/spatial/admission method in this pilot")
    frozen = {"schema": SCHEMA, "split": "train", "rows": selected, "coverage": coverage,
              "donors": donor_map(selected), "max_new_tokens": 64, "arms": list(ARMS),
              "manifest": str(manifest), "manifest_sha256": file_sha(manifest),
              "incumbent": str(old_path), "incumbent_sha256": file_sha(old_path),
              "protocol": str(protocol_path), "protocol_sha256": file_sha(protocol_path),
              "conch_expert_id": args.conch_expert_id, "evidence_chars": args.evidence_chars,
              "anchor_root": str(args.anchor_root.resolve()), "scorer": scorer_identity(args.anchor_root.resolve()),
              "source_code": code_identity(), "references_used": False,
              "novelty": "task-matched specialist ablation; not a new inference algorithm"}
    write_new(args.output / "frozen.json", frozen)
    print("PREPARED", digest(frozen), len(selected), "cases; no model calls")


def visible(transport):
    return {(v["expert_id"], v["evidence_id"]) for v in transport.get("presented", ())}


def generate(args):
    from merit_feddg.capabilities import EvidenceItem
    from merit_feddg.capability_runtime import NativeSession, NativeState, ValueGenerationConfig
    from merit_feddg.generalist_factory import generalist_provenance, load_generalist
    from run_quilt_worker import assert_single_gpu

    import torch

    frozen = read_json(args.output / "frozen.json")
    if frozen["source_code"] != code_identity():
        raise ValueError("implementation changed since prepare")
    for field in ("manifest", "incumbent", "protocol"):
        if file_sha(frozen[field]) != frozen[field + "_sha256"]:
            raise ValueError("frozen " + field + " changed")
    cache = read_json(args.output / "quilt_predictions.json")
    predictions = validate_predictions(frozen, cache)
    if not args.gpu_uuid:
        raise ValueError("generate requires explicit authorized GPU UUID")
    assert_single_gpu(torch, args.gpu_uuid)
    protocol, old = read_json(frozen["protocol"]), read_json(frozen["incumbent"])
    identity = digest(frozen)
    # Independent processes: the Quilt worker must have exited before this starts.
    write_new(args.output / "actor_attempt.json", {"identity": identity,
              "source_cache_sha256": file_sha(args.output / "quilt_predictions.json"),
              "generalist": generalist_provenance(protocol["config"]["generalist"], "artifacts"),
              "complete": False, "gpu_uuid": args.gpu_uuid})
    start = time.perf_counter()
    probe = load_generalist(protocol["config"]["generalist"], "artifacts")
    probe.model.eval().requires_grad_(False)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - start
    torch.cuda.reset_peak_memory_stats()
    jobs = {(j["target_id"], j["variant"]): j for j in prediction_jobs(frozen)}
    completed, calls = [], 0

    def context(row, cfg, items):
        session = NativeSession(probe, row["image"], row["prompt"], row["question"], cfg)
        images, prompt = session.context(NativeState(items=tuple(items)))
        return images, prompt, session.last_transport

    def answer(images, prompt):
        nonlocal calls
        calls += 1
        t0 = time.perf_counter()
        with torch.inference_mode():
            block = probe.new_answer_session(images, prompt).propose((), count=1, length=64)[0]
        torch.cuda.synchronize()
        if not block.text.strip():
            raise ValueError("empty actor answer")
        return {"text": block.text, "token_ids": list(block.tokens),
                "seconds": time.perf_counter() - t0, "output_tokens": len(block.tokens),
                "prompt_sha256": digest(prompt)}

    try:
        for index, row in enumerate(frozen["rows"]):
            if file_sha(row["image"]) != row["image_file_sha256"]:
                raise ValueError("input image changed")
            before = old[row["id"]]
            cfg = ValueGenerationConfig(**before["generation_config"])
            raw = tuple(EvidenceItem(**e) for e in before["evidence"])
            images, original_prompt, original_transport = context(row, cfg, raw)
            allowed = visible(original_transport)
            kept = tuple(e for e in raw if (e.expert_id, e.evidence_id) in allowed)
            # Explicit study budget; prefilter avoids resurrecting hidden old packets.
            cfg = replace(cfg, max_evidence_chars=frozen["evidence_chars"])
            _, base_prompt, base_transport = context(row, cfg, kept)
            if base_prompt != original_prompt or visible(base_transport) != allowed:
                raise RuntimeError("expanded study budget changed the compact input")
            arms = {"compact": answer(images, base_prompt)}
            if arms["compact"]["token_ids"] != before["token_ids"]:
                raise RuntimeError("historical compact token parity failed; do not score mismatched baselines")
            no_conch = tuple(e for e in kept if e.expert_id != frozen["conch_expert_id"])
            im, prompt, deletion_transport = context(row, cfg, no_conch)
            if visible(deletion_transport) != { (e.expert_id, e.evidence_id) for e in no_conch }:
                raise RuntimeError("deletion resurrected/displaced evidence")
            arms["without_conch"] = answer(im, prompt)
            matched = predictions[jobs[row["id"], "matched"]["key"]]
            arms["quilt_alone"] = {k: matched[k] for k in ("text", "token_ids", "seconds", "output_tokens")}
            transports = {"compact": base_transport, "without_conch": deletion_transport}
            for variant, arm in (("matched", "compact_quilt"), ("wrong_image", "compact_wrong_image")):
                record = predictions[jobs[row["id"], variant]["key"]]
                item = quilt_item(record)
                im, prompt, transport = context(row, cfg, kept + (item,))
                expected = allowed | {(item.expert_id, item.evidence_id)}
                if visible(transport) != expected:
                    raise RuntimeError("new evidence omitted or old evidence displaced; report coverage failure")
                escaped = json.dumps(record["text"], ensure_ascii=False)[1:-1]
                if escaped not in prompt and record["text"] not in prompt:
                    raise RuntimeError("full specialist text not present in actual prompt")
                transports[arm] = transport
                arms[arm] = answer(im, prompt)
            result = {"identity": identity, "id": row["id"], "arms": arms,
                      "image_sha256": row["image_sha256"], "historical_token_parity": True,
                      "conch_was_delivered": len(no_conch) != len(kept),
                      "transport": transports, "wrong_image_donor": frozen["donors"][row["id"]]}
            write_new(args.output / "cases" / f"{index:04d}.json", result)
            completed.append(row["id"])
            print("ACTOR", len(completed), "/", len(frozen["rows"]), flush=True)
        write_new(args.output / "result.json", {"identity": identity, "complete": True,
                  "case_ids": completed, "actual_actor_calls": calls, "actor_load_seconds": load_seconds,
                  "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                  "worker_model_calls": cache["actual_model_calls"],
                  "medical_accuracy_evaluated": False, "scope": frozen["coverage"]})
    except Exception as exc:
        write_new(args.output / "actor_failure.json", {"type": type(exc).__name__, "message": str(exc),
                  "case_ids_completed": completed, "actual_actor_calls_attempted": calls, "complete": False})
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("prepare", "generate"), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--incumbent-run", type=Path)
    p.add_argument("--incumbent-arm", default="compact_rows")
    p.add_argument("--anchor-root", type=Path)
    p.add_argument("--split", choices=("train",), default="train")
    p.add_argument("--limit", type=int, default=12)
    p.add_argument("--conch-expert-id", default="conch_tissue")
    p.add_argument("--evidence-chars", type=int, default=4096)
    p.add_argument("--gpu-uuid")
    args = p.parse_args()
    if not 1600 <= args.evidence_chars <= 8000:
        raise ValueError("explicit bounded study evidence budget required")
    if args.stage == "prepare":
        prepare(args)
    else:
        generate(args)


if __name__ == "__main__":
    main()
