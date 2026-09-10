"""Matched full-manifest evaluation. No split creation, fitting, or test tuning."""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

from .capabilities import CapabilityResult, EvidenceItem, tool_descriptors
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
    normalized = []
    for row in rows:
        answer_type = str(row.get("answer_type", "")).strip().lower()
        if answer_type not in {"closed", "open"}:
            raise ValueError("each case needs answer_type closed/open for the frozen prompt contract")
        normalized.append({key: row[key] for key in ("id", "image", "question", "image_sha256")}
                          | {"answer_type": answer_type})
    return normalized


def generation_prompt(row, config):
    """Render the same answer-blind CE/OE prompts as ANCHOR anchor-ce-v1."""
    contract = config.get("prompt_contract", "legacy_suffix")
    question = str(row["question"]).strip()
    if contract == "anchor-ce-v1":
        if row["answer_type"] == "closed":
            return f"{question} Please answer Yes or No."
        return f"{question}\nGive only the short answer. Do not explain."
    if contract != "legacy_suffix":
        raise ValueError(f"unsupported prompt contract: {contract}")
    return question + "\n" + config.get("prompt_suffix", "Answer concisely from the image.")


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
    if protocol == "semantic_spatial":
        if decoder.evidence_style != "semantic" or not decoder.token_budgeted_evidence or decoder.visual_views:
            raise ValueError("semantic_spatial protocol requires token-budgeted semantic evidence")
        if decoder.vector_gate_probe_tokens != decoder.max_new_tokens:
            raise ValueError("semantic_spatial requires full answer gate probes")
        definitions = {"generalist": (False, "off"), "semantic_all": (False, "off"),
                       "hybrid_all": (True, "off"), "hybrid_contrast": (True, "visual_contrast"),
                       "hybrid_gate": (True, "multidimensional")}
        return {name: replace(decoder, semantic_spatial=spatial, vector_gate=gate,
                              block_tokens=decoder.max_new_tokens, uncertainty_from_probe=False,
                              behavior_probe="audit" if name == "hybrid_gate" else "off")
                for name, (spatial, gate) in definitions.items()}
    if protocol in {"vector", "spatial"}:
        if decoder.evidence_style != "tensor" or decoder.token_budgeted_evidence or decoder.visual_views:
            raise ValueError("vector protocol requires tensor style, no text evidence, and no visual panels")
        names = {"generalist": "off", "tensor_all": "off", "tensor_gate": "visual_contrast"}
        if protocol == "spatial":
            if decoder.vector_gate_probe_tokens != decoder.max_new_tokens:
                raise ValueError("spatial protocol requires the full answer budget for gate probes")
            names = {"generalist": "off", "spatial_equal": "off", "spatial_weighted": "off",
                     "spatial_gate": "visual_contrast"}
        return {name: replace(decoder, vector_gate=gate,
                              spatial_weighting="equal" if name == "spatial_equal" else decoder.spatial_weighting,
                              block_tokens=decoder.max_new_tokens,
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


def _finalize_shards(root, rows, methods, protocol_payload, shard_count):
    """Merge complete disjoint shards once; safe when workers finish together."""
    lock_path = root / "finalize.lock"
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        shard_roots = [root / "shards" / f"{i:04d}-of-{shard_count:04d}"
                       for i in range(shard_count)]
        required = [path / f"{method}.json" for path in shard_roots for method in methods]
        if not all(path.is_file() for path in required):
            return False
        expected = [row["id"] for row in rows]
        for method in methods:
            merged = {}
            for path in shard_roots:
                part = json.loads((path / f"{method}.json").read_text(encoding="utf-8"))
                overlap = set(merged).intersection(part)
                if overlap:
                    raise ValueError(f"duplicate cases across shards: {sorted(overlap)[:3]}")
                merged.update(part)
            if set(merged) != set(expected):
                raise ValueError(f"incomplete {method} shard union: {len(merged)} != {len(expected)}")
            atomic_json(root / f"{method}.json", {sample_id: merged[sample_id] for sample_id in expected})
        routing = {}
        for path in shard_roots:
            routing.update(json.loads((path / "routing.json").read_text(encoding="utf-8")))
        atomic_json(root / "routing.json", {sample_id: routing[sample_id] for sample_id in expected})
        atomic_json(root / "protocol.json", {**protocol_payload, "shards": shard_count,
                                              "shards_complete": True})
        return True


def run(manifest, config_path, output_dir, *, artifacts="artifacts", protocol="spatial",
        shard_index=0, shard_count=1):
    from .capability_experts import CapabilityPool
    from .capability_runtime import CapabilityRuntime
    from .capability_study import _filter_optional_experts, _route_records
    from .generalist_factory import generalist_provenance, load_generalist, resolve_generalist_spec

    original = load_manifest(manifest)
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must be in [0, shard_count)")
    config = load_experiment_yaml(config_path)
    config["generalist"] = resolve_generalist_spec(config["generalist"])
    config["generalist"]["deterministic_image_padding"] = True
    decoder = ValueGenerationConfig(**config["capability_value"]["generation"])
    arms = experiment_arms(decoder, protocol)
    methods = {name: asdict(arm) for name, arm in arms.items()}
    vector = protocol in {"vector", "spatial"}
    frozen_spatial = vector or protocol == "semantic_spatial"
    if frozen_spatial and (not config["generalist"].get("training_free_spatial")
                   or config["generalist"].get("tensor_bridge_checkpoint")):
        raise ValueError("vector/spatial experiment requires training_free_spatial, never a trained bridge")
    specs, excluded = _filter_optional_experts(config["experts"], artifacts)
    specs.pop("source_cases", None)
    if vector:
        for name, spec in list(specs.items()):
            supported = [v for v in spec.get("capabilities", [])
                         if v in {"classification", "segmentation", "detection"}]
            if "classification" in supported and not spec.get("spatial_support", False):
                supported.remove("classification")
                excluded[name] = {"reason": "classification_without_spatial_support"}
            if supported:
                specs[name] = {**spec, "capabilities": supported}
            else:
                excluded.setdefault(name, {"reason": "unsupported_nontext_output_type"})
                del specs[name]
    if any(spec.get("native_precision_card") for spec in specs.values()):
        raise ValueError("this no-calibration benchmark does not enable domain precision cards")
    prompt_by_id = {row["id"]: generation_prompt(row, config) for row in original}
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
    work_root = (root if shard_count == 1 else
                 root / "shards" / f"{shard_index:04d}-of-{shard_count:04d}")
    work_root.mkdir(parents=True, exist_ok=True)
    probe = load_generalist(config["generalist"], artifacts)
    if frozen_spatial:
        bridge = probe.tensor_bridge
        if not getattr(bridge, "training_free", False) or list(bridge.parameters()):
            raise ValueError("spatial experiment requires a parameter-free evidence operator")
    rows = [{"id": r["id"], "image": r["image"], "question": r["question"],
             "modality": "mixed",
             "capability": "classification", "task": "open_vqa", "domain": "official-test",
             "domain_kind": "official_dataset_split", "role": "target",
             "group_id": r["image_sha256"], "image_sha256": r["image_sha256"]} for r in original]
    rows = rows[shard_index::shard_count]
    rows, routes = _route_records(rows, config.get("routing", {}), lambda: probe,
                                  {"identity": identity}, work_root)
    atomic_json(work_root / "routing.json", routes)
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
                    prompt = prompt_by_id[row["id"]]
                    session = NativeSession(probe, row["image"], prompt, row["question"], arm)
                    engine = CapabilityRuntime(session, shared_pool, row, specs, arm, None)
                    cached = engine.run("generalist" if method == "generalist" else "all_evidence")
                    cached["generation_config"] = asdict(arm)
                    cached["input_modality"] = row["modality"]
                    cached["available_experts"] = sorted({d["expert"] for d in tool_descriptors(specs, row)})
                    atomic_json(path, {"identity": identity, "output": cached})
                outputs[method][row["id"]] = cached
            print(f"matched full manifest {index}/{len(rows)} {row['id']}", flush=True)
        for method, result in outputs.items():
            atomic_json(work_root / f"{method}.json", result)
        protocol_payload = {
            "identity": identity, "n": len(original), "methods": list(methods), "config": config,
            "experiment_protocol": protocol, "arm_configs": methods,
            "bridge_requires_pretraining": False, "gate_fitted": False,
            "collaboration_training_free": frozen_spatial,
            "vector_gate_unit": "acquired_expert_result" if frozen_spatial else None,
            "vector_gate_control": "same_size_image_channel_mean" if frozen_spatial else None,
            "semantic_channel": "existing_frozen_token_embeddings" if protocol == "semantic_spatial" else None,
            "excluded": excluded, "baseline_regenerated": True, "dataset_partitioned": False,
            "references_loaded_for_generation": False, "calibration_or_policy_fitted": False,
            "answer_type_used_for_generation": config.get("prompt_contract") == "anchor-ce-v1",
            "output_grammar": config.get("prompt_contract", "legacy_suffix"),
            "limitations": ["No clinical efficacy claim until fixed reference evaluation.",
                            "Packing may change delivered subsets; compare transport before attributing channel effects.",
                            "Warm/cached expert timings are not comparable across sequential arms."]}
        atomic_json(work_root / "protocol.json", {**protocol_payload,
                                                   "shard_index": shard_index,
                                                   "shard_count": shard_count,
                                                   "shard_n": len(rows)})
        if shard_count == 1:
            atomic_json(root / "protocol.json", {**protocol_payload, "shards": 1,
                                                  "shards_complete": True})
        else:
            _finalize_shards(root, original, methods, protocol_payload, shard_count)
    finally:
        pool.clear()
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", default="configs/matched_vector_gate.yaml")
    parser.add_argument("--output", default="runs/matched-spatial")
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--protocol", choices=("text", "vector", "spatial", "semantic_spatial"), default="spatial")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    print(run(args.manifest, args.config, args.output, artifacts=args.artifacts, protocol=args.protocol,
              shard_index=args.shard_index, shard_count=args.shard_count))


if __name__ == "__main__":
    main()
