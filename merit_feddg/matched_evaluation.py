"""Matched full-manifest evaluation. No split creation, fitting, or test tuning."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

from .capabilities import CapabilityResult, EvidenceItem
from .capability_runtime import NativeSession, ValueGenerationConfig
from .io import load_experiment_yaml
from .open_study import atomic_json, fingerprint, model_provenance


def load_manifest(path):
    rows = [json.loads(s) for s in Path(path).read_text(encoding="utf-8").splitlines() if s.strip()]
    ids = [r["id"] for r in rows]
    if not rows or len(set(ids)) != len(ids):
        raise ValueError("nonempty manifest with unique case IDs required")
    for row in rows:
        if any(k in row for k in ("answer", "answers", "reference", "references", "label")):
            raise ValueError("generation manifest must not contain answer labels")
        if not all(row.get(k) for k in ("image", "question", "image_sha256")):
            raise ValueError("each case needs image, question, and pixel/file identity")
    # Task-type metadata belongs to offline evaluation, never inference. Strip
    # it from the canonical manifest, cache key, routing, and every generation arm.
    return [{key: row[key] for key in ("id", "image", "question", "image_sha256")} for row in rows]


def cache_identity(config, rows, model_identity):
    source = Path(__file__).parent
    return fingerprint({"protocol": "matched-full-manifest-v1", "config": config,
                        "manifest": rows, "generalist": model_identity,
                        "implementation": {p.relative_to(source).as_posix():
                                           hashlib.sha256(p.read_bytes()).hexdigest()
                                           for p in sorted(source.rglob("*.py"))}})


def load_cached(path, identity):
    if not Path(path).exists():
        return None
    result = json.loads(Path(path).read_text(encoding="utf-8"))
    if result.get("identity") != identity:
        raise ValueError("cache belongs to a different model/config/manifest/code; refusing reuse")
    return result["output"]


class SharedExpertPool:
    """Persist native outputs so point/range/permission arms see the same tool run."""

    def __init__(self, pool, directory, identity):
        self.pool, self.directory, self.identity = pool, Path(directory), identity
        self.last_origin = "unknown"

    def _get(self, operation, expert, request):
        key = fingerprint([operation, expert, asdict(request)])
        path = self.directory / f"{key}.json"
        value = load_cached(path, self.identity)
        self.last_origin = "cached_native_output" if value is not None else "live_native_output"
        if value is None:
            value = getattr(self.pool, operation)(expert, request)
            value = asdict(value) if operation == "infer" else value
            atomic_json(path, {"identity": self.identity, "output": value})
        return copy.deepcopy(value)

    def infer(self, expert, request):
        value = self._get("infer", expert, request)
        return CapabilityResult(**{**value, "items": tuple(EvidenceItem(**v) for v in value["items"])})

    def behavior_probe(self, expert, request):
        value = self._get("behavior_probe", expert, request)
        if self.last_origin == "cached_native_output":
            value["cached_extra_model_calls"] = value.get("extra_model_calls", 0)
            value["extra_model_calls"] = 0
            value["cached_seconds"] = value.get("seconds", 0)
            value["seconds"] = 0
        value["result_origin"] = self.last_origin
        return value


def experiment_arms(decoder, protocol):
    """Declare matched arms without consulting case metadata or answer types."""
    if protocol == "vector":
        if decoder.evidence_style != "tensor" or decoder.token_budgeted_evidence or decoder.visual_views:
            raise ValueError("vector protocol requires tensor style, no text evidence, and no visual panels")
        names = {"generalist": "off", "tensor_all": "off", "tensor_gate": "visual_contrast"}
        return {name: replace(decoder, vector_gate=gate, block_tokens=decoder.max_new_tokens,
                              behavior_probe="off", uncertainty_from_probe=False)
                for name, gate in names.items()}
    if protocol != "text" or not decoder.token_budgeted_evidence or decoder.visual_views:
        raise ValueError("text protocol requires audited token-budgeted single-image evidence")
    methods = {"generalist": "uncertainty", "point": "scoped",
               "uncertainty": "uncertainty", "permissions": "permissions"}
    return {name: replace(decoder, evidence_style=style, evidence_top_k=10000,
                          block_tokens=decoder.max_new_tokens, vector_gate="off",
                          uncertainty_from_probe=name in {"uncertainty", "permissions"},
                          behavior_probe="audit" if name in {"uncertainty", "permissions"} else "off")
            for name, style in methods.items()}


def run(manifest, config_path, output_dir, *, artifacts="artifacts", protocol="text"):
    from .capability_experts import CapabilityPool
    from .capability_runtime import CapabilityRuntime
    from .capability_study import _filter_optional_experts, _route_records
    from .generalist_factory import generalist_provenance, load_generalist, resolve_generalist_spec

    original = load_manifest(manifest)
    config = load_experiment_yaml(config_path)
    config["generalist"] = resolve_generalist_spec(config["generalist"])
    config["generalist"]["deterministic_image_padding"] = True
    decoder = ValueGenerationConfig(**config["capability_value"]["generation"])
    arms = experiment_arms(decoder, protocol)
    methods = {name: asdict(arm) for name, arm in arms.items()}
    if protocol == "vector" and not config["generalist"].get("tensor_bridge_checkpoint"):
        raise ValueError("vector experiment requires a trained, base-matched tensor_bridge_checkpoint")
    specs, excluded = _filter_optional_experts(config["experts"], artifacts)
    specs.pop("source_cases", None)
    if protocol == "vector":
        for name, spec in list(specs.items()):
            supported = [v for v in spec.get("capabilities", [])
                         if v in {"classification", "segmentation", "detection"}]
            if supported:
                specs[name] = {**spec, "capabilities": supported}
            else:
                excluded[name] = {"reason": "unsupported_nontext_output_type"}
                del specs[name]
    if any(spec.get("native_precision_card") for spec in specs.values()):
        raise ValueError("this no-calibration benchmark does not enable domain precision cards")
    model_id = generalist_provenance(config["generalist"], artifacts)
    expert_ids = {name: model_provenance(spec, artifacts) for name, spec in specs.items()}
    # Bind cached predictions to actual bytes, not just caller-supplied image IDs.
    image_files = {path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                   for path in sorted({row["image"] for row in original})}
    identity = cache_identity({**config, "active_experts": specs, "methods": methods,
                               "expert_provenance": expert_ids,
                               "image_file_sha256": image_files}, original, model_id)
    root = Path(output_dir) / identity
    root.mkdir(parents=True, exist_ok=True)
    probe = load_generalist(config["generalist"], artifacts)
    if protocol == "vector":
        bridge = probe.tensor_bridge
        bridge.eval().requires_grad_(False)
        if bridge.gate.detach().item() == 0:
            raise ValueError("trained bridge has zero fusion strength; vector experiment would be a null intervention")
        if not set(specs).intersection(b["expert"] for b in bridge.contract.bindings):
            raise ValueError("no active expert has a registered native-output binding in the trained bridge")
    rows = [{"id": r["id"], "image": r["image"], "question": r["question"], "modality": "mixed",
             "capability": "classification", "task": "open_vqa", "domain": "official-test",
             "domain_kind": "official_dataset_split", "role": "target",
             "group_id": r["image_sha256"], "image_sha256": r["image_sha256"]} for r in original]
    rows, routes = _route_records(rows, config.get("routing", {}), lambda: probe,
                                  {"identity": identity}, root)
    atomic_json(root / "routing.json", routes)
    pool = CapabilityPool(specs, artifacts, source_records=())
    outputs = {name: {} for name in methods}
    try:
        for index, row in enumerate(rows, 1):
            # Reuse actual specialist predictions across all evidence arms.
            pool.reset_case()
            shared_pool = SharedExpertPool(pool, root / "expert-cache" / fingerprint(row["id"]), identity)
            for method, arm in arms.items():
                path = root / "case-cache" / method / f"{fingerprint(row['id'])}.json"
                cached = load_cached(path, identity)
                if cached is None:
                    prompt = row["question"] + "\n" + config.get("prompt_suffix", "Answer concisely from the image.")
                    session = NativeSession(probe, row["image"], prompt, row["question"], arm)
                    engine = CapabilityRuntime(session, shared_pool, row, specs, arm, None)
                    cached = engine.run("generalist" if method == "generalist" else "all_evidence")
                    cached["generation_config"] = asdict(arm)
                    atomic_json(path, {"identity": identity, "output": cached})
                outputs[method][row["id"]] = cached
            print(f"matched full manifest {index}/{len(rows)} {row['id']}", flush=True)
        for method, result in outputs.items():
            atomic_json(root / f"{method}.json", result)
        atomic_json(root / "protocol.json", {
            "identity": identity, "n": len(rows), "methods": list(methods), "config": config,
            "experiment_protocol": protocol, "arm_configs": methods,
            "bridge_requires_pretraining": protocol == "vector", "gate_fitted": False,
            "vector_gate_unit": "acquired_expert_result" if protocol == "vector" else None,
            "vector_gate_control": "same_size_image_channel_mean" if protocol == "vector" else None,
            "excluded": excluded, "baseline_regenerated": True, "dataset_partitioned": False,
            "references_loaded_for_generation": False, "calibration_or_policy_fitted": False,
            "answer_type_used_for_generation": False, "output_grammar": "unconstrained_for_all_questions",
            "limitations": ["No clinical efficacy claim until fixed reference evaluation.",
                            "Packing may change delivered subsets; compare transport before attributing channel effects.",
                            "Warm/cached expert timings are not comparable across sequential arms."]})
    finally:
        pool.clear()
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", default="configs/matched_permissions.yaml")
    parser.add_argument("--output", default="runs/matched-permissions")
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--protocol", choices=("text", "vector"), default="text")
    args = parser.parse_args()
    print(run(args.manifest, args.config, args.output, artifacts=args.artifacts, protocol=args.protocol))


if __name__ == "__main__":
    main()
