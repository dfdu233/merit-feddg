"""Matched full-manifest evaluation. No split creation, fitting, or test tuning."""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

from .capabilities import CapabilityResult, EvidenceItem, tool_descriptors
from .capability_runtime import NativeSession, ValueGenerationConfig
from .io import load_experiment_yaml
from .open_study import atomic_json, fingerprint, model_provenance


def load_manifest(path, *, include_answer_type=True):
    rows = [json.loads(s) for s in Path(path).read_text(encoding="utf-8").splitlines() if s.strip()]
    ids = [r["id"] for r in rows]
    if not rows or len(set(ids)) != len(ids):
        raise ValueError("nonempty manifest with unique case IDs required")
    for row in rows:
        if any(k in row for k in ("answer", "answers", "reference", "references", "label")):
            raise ValueError("generation manifest must not contain answer labels")
        if not all(row.get(k) for k in ("image", "question", "image_sha256")):
            raise ValueError("each case needs image, question, and pixel/file identity")
    if not include_answer_type:
        return [{key: row[key] for key in ("id", "image", "question", "image_sha256")}
                | {"task": str(row.get("task") or "open_vqa")} for row in rows]
    normalized = []
    for row in rows:
        answer_type = str(row.get("answer_type", "")).strip().lower()
        task = str(row.get("task") or "open_vqa").strip().lower()
        if task not in {"open_vqa", "report_generation"}:
            raise ValueError("each case needs task open_vqa/report_generation")
        allowed = {"report"} if task == "report_generation" else {"closed", "open"}
        if answer_type not in allowed:
            raise ValueError(f"answer_type {answer_type!r} is incompatible with task {task!r}")
        normalized.append({key: row[key] for key in ("id", "image", "question", "image_sha256")}
                          | {"answer_type": answer_type, "task": task})
    return normalized


def generation_prompt(row, config):
    """Render the same answer-blind prompts as the corresponding ANCHOR task."""
    contract = config.get("prompt_contract", "legacy_suffix")
    question = str(row["question"]).strip()
    if contract in {"anchor-ce-v1", "anchor-task-v1"}:
        if row.get("task", "open_vqa") == "report_generation":
            if contract != "anchor-task-v1":
                raise ValueError("report generation requires anchor-task-v1")
            return question
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

    def __init__(self, pool, directory, identity, fallback_directories=()):
        self.pool, self.directory, self.identity = pool, Path(directory), identity
        self.fallback_directories = tuple((Path(path), donor_identity)
                                          for path, donor_identity in fallback_directories)
        self.last_origin = "unknown"

    def _get(self, operation, expert, request):
        key = fingerprint([operation, expert, asdict(request)])
        path = self.directory / f"{key}.json"
        value = load_cached(path, self.identity)
        self.last_origin = "cached_native_output" if value is not None else "live_native_output"
        if value is None:
            for fallback, donor_identity in self.fallback_directories:
                value = load_cached(fallback / f"{key}.json", donor_identity)
                if value is not None:
                    self.last_origin = "reused_compatible_native_output"
                    break
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
        if self.last_origin in {"cached_native_output", "reused_compatible_native_output"}:
            value["cached_extra_model_calls"] = value.get("extra_model_calls", 0)
            value["extra_model_calls"] = 0
            value["cached_seconds"] = value.get("seconds", 0)
            value["seconds"] = 0
        value["result_origin"] = self.last_origin
        return value


def experiment_arms(decoder, protocol):
    """Declare matched arms without consulting case metadata or answer types."""
    if protocol == "verified_packets":
        if decoder.evidence_style != "semantic" or decoder.native_entry_transport or decoder.semantic_spatial:
            raise ValueError("verified_packets requires intact semantic-only packets")
        definitions = {"generalist": (False, True), "semantic_all": (False, True),
                       "compact_rows": (True, False), "compact_all": (True, True),
                       "compact_verified": (True, True)}
        return {name: replace(decoder, compact_native=compact, compact_columns=columns,
            block_tokens=decoder.max_new_tokens, vector_gate="off", behavior_probe="off",
            claim_attribute_filter=False, uncertainty_from_probe=False)
            for name, (compact, columns) in definitions.items()}
    if protocol == "native_claims":
        if decoder.evidence_style != "semantic" or not decoder.token_budgeted_evidence or decoder.visual_views:
            raise ValueError("native_claims requires token-budgeted semantic evidence")
        if decoder.vector_gate_probe_tokens != decoder.max_new_tokens:
            raise ValueError("native_claims requires full answer gate probes")
        # Isolate packing, cheap filtering, spatial transport, and local gating.
        definitions = {
            "generalist": (False, False, False, "off"),
            "semantic_all": (False, False, False, "off"),
            "entry_all": (True, False, False, "off"),
            "entry_filtered": (True, True, False, "off"),
            "hybrid_all": (True, True, True, "off"),
            "hybrid_gate": (True, True, True, "claim_support")}
        return {name: replace(decoder, native_entry_transport=entries, claim_attribute_filter=filtered,
            semantic_spatial=spatial, vector_gate=gate, evidence_order="acquisition",
            block_tokens=decoder.max_new_tokens, uncertainty_from_probe=False, behavior_probe="off")
            for name, (entries, filtered, spatial, gate) in definitions.items()}
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
        shard_index=0, shard_count=1, reuse_generalist=None, reuse_expert_run=None):
    from .capability_experts import CapabilityPool
    from .capability_runtime import CapabilityRuntime
    from .capability_study import _filter_optional_experts, _route_records
    from .generalist_factory import generalist_provenance, load_generalist, resolve_generalist_spec

    config = load_experiment_yaml(config_path)
    uses_answer_contract = config.get("prompt_contract", "legacy_suffix") in {
        "anchor-ce-v1", "anchor-task-v1"
    }
    original = load_manifest(manifest, include_answer_type=protocol != "verified_packets" or uses_answer_contract)
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must be in [0, shard_count)")
    if protocol == "native_claims" and config.get("prompt_contract") != "anchor-ce-v1":
        raise ValueError("native_claims requires the frozen ANCHOR CE/OE prompt contract")
    if protocol == "verified_packets":
        if config.get("prompt_contract", "legacy_suffix") not in {
            "legacy_suffix", "anchor-ce-v1", "anchor-task-v1"
        }:
            raise ValueError("verified_packets requires a declared legacy or ANCHOR prompt contract")
        if not config.get("answer_verifiers"):
            raise ValueError("verified_packets requires explicitly configured frozen verifiers")
    elif reuse_generalist or reuse_expert_run:
        raise ValueError("cross-run reuse is restricted to verified_packets")
    config["generalist"] = resolve_generalist_spec(config["generalist"])
    config["generalist"]["deterministic_image_padding"] = True
    decoder = ValueGenerationConfig(**config["capability_value"]["generation"])
    arms = experiment_arms(decoder, protocol)
    methods = {name: asdict(arm) for name, arm in arms.items()}
    vector = protocol in {"vector", "spatial"}
    frozen_spatial = vector or protocol in {"semantic_spatial", "native_claims"}
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
    verifier_specs = config.get("answer_verifiers", {}) if protocol == "verified_packets" else {}
    verifier_provenance = {name: model_provenance(spec, artifacts) for name, spec in verifier_specs.items()}
    reused_generalist, reuse_generalist_audit = None, None
    if reuse_generalist:
        reuse_path = Path(reuse_generalist)
        payload = json.loads(reuse_path.read_text(encoding="utf-8"))
        reused_generalist = payload.get("outputs")
        expected_ids = [row["id"] for row in original]
        if (payload.get("schema") != "matched-generalist-reuse-v1"
                or payload.get("prompt_contract") != config.get("prompt_contract", "legacy_suffix")
                or payload.get("generalist_id") != config["generalist"]["id"]
                or not isinstance(reused_generalist, dict)
                or set(reused_generalist) != set(expected_ids)):
            raise ValueError("reused Generalist does not match protocol/model/full manifest")
        for sample_id in expected_ids:
            value = reused_generalist[sample_id]
            if not isinstance(value.get("text"), str) or not value["text"].strip() or not value.get("token_ids"):
                raise ValueError(f"invalid reused Generalist output: {sample_id}")
        reuse_generalist_audit = {"path": str(reuse_path.resolve()),
            "sha256": hashlib.sha256(reuse_path.read_bytes()).hexdigest(),
            "source": payload.get("source"), "n": len(reused_generalist)}
    fallback_expert_root, reuse_expert_audit, reused_routes = None, None, None
    if reuse_expert_run:
        donor_root = Path(reuse_expert_run)
        donor_protocol_path = donor_root / "protocol.json"
        donor_protocol = json.loads(donor_protocol_path.read_text(encoding="utf-8"))
        donor_routes = json.loads((donor_root / "routing.json").read_text(encoding="utf-8"))
        expected_ids = {row["id"] for row in original}
        donor_specs, donor_excluded = _filter_optional_experts(
            donor_protocol["config"]["experts"], artifacts)
        donor_specs.pop("source_cases", None)
        donor_ids = {name: model_provenance(spec, artifacts) for name, spec in donor_specs.items()}
        # Unsharded runs written before scheduling-only sharding have no
        # shards_complete field; their final protocol.json is emitted only
        # after all rows are finalized.  A present field must still be true.
        donor_complete = donor_protocol.get("shards_complete", donor_protocol.get("shards") is None)
        route_groups = [donor_routes[row["id"]].get("group_id") for row in original]
        routes_bind_groups = all(value is not None for value in route_groups)
        if (not donor_complete
                or donor_protocol.get("n") != len(original)
                or set(donor_routes) != expected_ids
                or donor_ids != expert_ids or donor_excluded != excluded
                or (any(value is not None for value in route_groups) and not routes_bind_groups)
                or (routes_bind_groups and any(value != row["image_sha256"]
                                                for value, row in zip(route_groups, original)))):
            raise ValueError("reused expert run does not match full manifest/routes/expert weights")
        fallback_expert_root = donor_root / "expert-cache"
        reuse_expert_audit = {"path": str(donor_root.resolve()), "identity": donor_protocol["identity"],
            "protocol_sha256": hashlib.sha256(donor_protocol_path.read_bytes()).hexdigest(),
            "native_requests_keyed": True, "expert_provenance_equal": True,
            "route_group_hashes_available": routes_bind_groups}
        reused_routes = donor_routes
    # Bind cached predictions to actual bytes, not just caller-supplied image IDs.
    image_files = {path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                   for path in sorted({row["image"] for row in original})}
    identity = cache_identity({**config, "active_experts": specs, "methods": methods,
                               "expert_provenance": expert_ids, "verifier_provenance": verifier_provenance,
                               "image_file_sha256": image_files, "reuse_generalist": reuse_generalist_audit,
                               "reuse_expert_run": reuse_expert_audit}, original, model_id)
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
             "capability": "classification", "task": r.get("task", "open_vqa"),
             "domain": "official-test",
             "domain_kind": "official_dataset_split", "role": "target",
             "group_id": r["image_sha256"], "image_sha256": r["image_sha256"]} for r in original]
    rows = rows[shard_index::shard_count]
    if reused_routes is None:
        rows, routes = _route_records(rows, config.get("routing", {}), lambda: probe,
                                      {"identity": identity}, work_root)
    else:
        routes = {row["id"]: reused_routes[row["id"]] for row in rows}
        rows = [{**row, "modality": routes[row["id"]]["modality"]} for row in rows]
    atomic_json(work_root / "routing.json", routes)
    pool = CapabilityPool(specs, artifacts, source_records=())
    outputs = {name: {} for name in methods}
    verifiers = {}
    try:
        for index, row in enumerate(rows, 1):
            # Reuse actual specialist predictions across all evidence arms.
            pool.reset_case()
            fallbacks = (() if fallback_expert_root is None else
                         ((fallback_expert_root / fingerprint(row["id"]), reuse_expert_audit["identity"]),))
            shared_pool = SharedExpertPool(pool, root / "expert-cache" / fingerprint(row["id"]),
                                           identity, fallbacks)
            for method, arm in arms.items():
                path = root / "case-cache" / method / f"{fingerprint(row['id'])}.json"
                cached = load_cached(path, identity)
                if cached is None:
                    if method == "generalist" and reused_generalist is not None:
                        cached = copy.deepcopy(reused_generalist[row["id"]])
                        cached["reuse_provenance"] = reuse_generalist_audit
                    elif method == "compact_verified":
                        from .answer_arbitration import arbitrate_output, load_verifier
                        candidate = outputs["compact_all"][row["id"]]
                        baseline = outputs["generalist"][row["id"]]
                        eligible = [(name, spec) for name, spec in verifier_specs.items()
                                    if row["modality"] in spec["modalities"]]
                        if len(eligible) > 1:
                            raise ValueError("declare one verifier per modality; no implicit selection")
                        verifier = None
                        verifier_load_seconds = 0.0
                        if eligible and baseline["text"] != candidate["text"]:
                            name, spec = eligible[0]
                            if name not in verifiers:
                                load_started = perf_counter()
                                verifiers[name] = load_verifier(spec, artifacts)
                                verifier_load_seconds = perf_counter() - load_started
                            verifier = verifiers[name]
                        sources = [specs[e["expert_id"]]["id"] for e in candidate.get("evidence", [])]
                        cached = arbitrate_output(baseline, candidate, image=row["image"],
                            question=row["question"], modality=row["modality"], verifier=verifier,
                            generalist_id=config["generalist"]["id"], source_model_ids=sources)
                        cached["verifier_initialization_seconds"] = verifier_load_seconds
                        cached["seconds"] += verifier_load_seconds
                    else:
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
            "collaboration_training_free": frozen_spatial or protocol == "verified_packets",
            "answer_arbitration": "external_frozen_image_text" if verifier_specs else None,
            "verifier_provenance": verifier_provenance,
            "reuse_generalist": reuse_generalist_audit,
            "reuse_expert_run": reuse_expert_audit,
            "vector_gate_unit": "native_entry" if protocol == "native_claims" else ("acquired_expert_result" if frozen_spatial else None),
            "vector_gate_control": "paired_local_blur_translation" if protocol == "native_claims" else ("same_size_image_channel_mean" if frozen_spatial else None),
            "semantic_channel": "existing_frozen_token_embeddings" if protocol in {"semantic_spatial", "native_claims"} else None,
            "excluded": excluded, "baseline_regenerated": reused_generalist is None,
            "dataset_partitioned": False,
            "references_loaded_for_generation": False, "calibration_or_policy_fitted": False,
            "answer_type_used_for_generation": config.get("prompt_contract") in {
                "anchor-ce-v1", "anchor-task-v1"
            },
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
    parser.add_argument("--protocol", choices=("text", "vector", "spatial", "semantic_spatial", "native_claims", "verified_packets"), default="spatial")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--reuse-generalist")
    parser.add_argument("--reuse-expert-run")
    args = parser.parse_args()
    print(run(args.manifest, args.config, args.output, artifacts=args.artifacts, protocol=args.protocol,
              shard_index=args.shard_index, shard_count=args.shard_count,
              reuse_generalist=args.reuse_generalist, reuse_expert_run=args.reuse_expert_run))


if __name__ == "__main__":
    main()
