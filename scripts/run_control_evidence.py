"""Run CRES on one unchanged full manifest; resume and scheduling stops supported."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
from time import perf_counter

from merit_feddg.control_study import ABLATIONS, ARMS, control_case, validate_outputs
from merit_feddg.matched_evaluation import load_manifest
from merit_feddg.open_study import atomic_json, fingerprint


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-run", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    p.add_argument("--config", type=Path, default=Path("configs/control_evidence.json"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-cases", type=int, default=0, help="Scheduling stop; never changes manifest")
    p.add_argument("--check-only", action="store_true")
    args = p.parse_args()
    if args.max_cases < 0:
        raise ValueError("max-cases must be nonnegative")
    protocol = json.loads((args.source_run / "protocol.json").read_text())
    options = json.loads(args.config.read_text())
    expected_options = {"schema", "prompt_contract", "max_strength", "kl_budget", "fixed_strength", "ablations"}
    if (set(options) != expected_options or options["schema"] != "cres-v1"
            or type(options["ablations"]) is not bool):
        raise ValueError("invalid frozen CRES options")
    from merit_feddg.control_evidence import constrained_scores
    from merit_feddg.control_study import prompt_config
    from merit_feddg.soft_guidance import mix_scores
    constrained_scores([0], [0], max_strength=options["max_strength"], kl_budget=options["kl_budget"])
    mix_scores([0], [0], options["fixed_strength"])
    prompt_config({}, options["prompt_contract"])
    # Source metadata is needed solely for verifying the old prompt binding.
    # Uniform generation never branches on answer_type; no references are loaded.
    rows = load_manifest(args.manifest)
    routes = json.loads((args.source_run / "routing.json").read_text())
    ids = [r["id"] for r in rows]
    if (protocol.get("n") != len(rows) or protocol.get("shards_complete", True) is not True
            or set(routes) != set(ids)):
        raise ValueError("complete source run on the same full manifest required")
    cache_hashes = {}
    for row in rows:
        if routes[row["id"]].get("group_id") != row["image_sha256"]:
            raise ValueError("source image identity mismatch")
        if sha(row["image"]) != row["image_sha256"]:
            raise ValueError("manifest image SHA is not the actual file SHA256")
        path = args.source_run / "case-cache" / "compact_rows" / f"{fingerprint(row['id'])}.json"
        cached = json.loads(path.read_text())
        if cached.get("identity") != protocol["identity"]:
            raise ValueError("source cache identity mismatch")
        cache_hashes[row["id"]] = sha(path)
    spec = {**protocol["config"]["generalist"], "training_free_spatial": True,
            "deterministic_image_padding": True}
    if spec.get("backend") != "llava_med" or spec.get("tensor_bridge_checkpoint"):
        raise ValueError("first native CRES adapter requires frozen LLaVA-Med spatial bridge")
    if args.check_only:
        print(json.dumps({"preflight_passed": True, "n": len(rows), "labels_loaded": False,
                          "model_not_loaded": True, "options": options}))
        return
    import torch

    from merit_feddg.generalist_factory import generalist_provenance, load_generalist
    torch.set_num_threads(4)
    source_root = Path(__file__).resolve().parents[1]
    config = {"options": options, "arms": list((*ARMS, *ABLATIONS) if options["ablations"] else ARMS),
              "rows": rows, "manifest_sha256": sha(args.manifest), "manifest": str(args.manifest.resolve()),
              "source_protocol_sha256": sha(args.source_run / "protocol.json"),
              "source_run": str(args.source_run.resolve()), "source_caches": cache_hashes,
              "model": generalist_provenance(spec, args.artifacts), "generalist_spec": spec,
              "code": {str(q.relative_to(source_root)): sha(q)
                       for q in sorted((source_root / "merit_feddg").rglob("*.py"))},
              "runner_sha256": sha(__file__), "runtime": {"torch": torch.__version__,
              "cuda": torch.version.cuda, "threads": 4}, "inference_uses_labels": False}
    config = json.loads(json.dumps(config))
    identity = fingerprint(config)
    root = args.output / identity
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        frozen = root / "frozen.json"
        if frozen.exists() and json.loads(frozen.read_text()) != config:
            raise ValueError("frozen identity mismatch")
        atomic_json(frozen, config)
        print("ROOT", root.resolve(), flush=True)
        if not torch.cuda.is_available():
            raise RuntimeError("assigned CUDA device unavailable")
        torch.ones(1, device="cuda:0").sum().item()
        start = perf_counter()
        probe = load_generalist(spec, args.artifacts)
        probe.model.eval().requires_grad_(False)
        if list(probe.tensor_bridge.parameters()):
            raise ValueError("spatial bridge must have no learned parameters")
        atomic_json(root / f"startup-{len(list(root.glob('startup-*.json'))):03d}.json",
                    {"seconds": perf_counter() - start, "device": torch.cuda.get_device_name(0)})
        dest = root / "cases"
        dest.mkdir(exist_ok=True)
        completed = 0
        for row in rows:
            target = dest / f"{fingerprint(row['id'])}.json"
            if target.exists():
                stored = json.loads(target.read_text())
                if (stored.get("identity") != identity or stored.get("id") != row["id"]
                        or set(stored.get("arms", {})) != set(config["arms"])):
                    raise ValueError("invalid resumed row")
                validate_outputs(stored, config["arms"])
                continue
            if args.max_cases and completed >= args.max_cases:
                print("Scheduling stop; full manifest incomplete", flush=True)
                return
            path = args.source_run / "case-cache" / "compact_rows" / f"{fingerprint(row['id'])}.json"
            if sha(path) != cache_hashes[row["id"]] or sha(row["image"]) != row["image_sha256"]:
                raise ValueError("source changed after preflight")
            try:
                with torch.inference_mode():
                    record = control_case(probe, row, json.loads(path.read_text())["output"], protocol, options)
            except Exception as exc:
                atomic_json(root / f"failure-{fingerprint(row['id'])}.json",
                            {"id": row["id"], "type": type(exc).__name__, "message": str(exc)})
                raise
            atomic_json(target, {"identity": identity, **record})
            completed += 1
            print("DONE", row["id"], "seconds", round(record["wall_seconds"], 3), flush=True)
        if {q.stem for q in dest.glob("*.json")} != {fingerprint(i) for i in ids}:
            raise ValueError("incomplete or extra output IDs")
        atomic_json(root / "complete.json", {"identity": identity, "n": len(rows),
                                            "full_manifest_complete": True})


if __name__ == "__main__":
    main()
