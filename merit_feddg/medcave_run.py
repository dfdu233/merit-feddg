"""Opt-in MedCAVE free-answer runner; existing matched runners remain unchanged."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

from .medcave import (
    FrozenPolicy,
    SourceArtifactVerifier,
    admissible_descriptors,
    certificate_valid,
    certify,
    digest,
    replay_trajectory,
    run_case,
)


def read_json(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    """Identical writes may resume; no replacement of different artifacts."""
    from .open_study import atomic_json

    path = Path(path)
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"refusing to overwrite existing artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, value)


def manifest_rows(path, source):
    """Accept existing official TRAIN or strict source manifests; strip labels/types."""
    from .open_data import pixel_digest

    raw = [json.loads(v) for v in Path(path).read_text().splitlines() if v.strip()]
    rows = []
    forbidden = {
        "answer",
        "answers",
        "reference",
        "references",
        "label",
        "labels",
        "mask",
        "ground_truth",
        "report",
        "target",
    }
    allowed = {
        "id",
        "image",
        "question",
        "image_sha256",
        "answer_type",
        "task",
        "official_split",
        "official_index",
        "role",
        "group_id",
        "modality",
        "capability",
        "domain",
        "domain_kind",
        "patient_id",
        "study_id",
        "slide_id",
    }
    for r in raw:
        if forbidden & r.keys() or set(r) - allowed:
            raise ValueError("manifest contains forbidden/unknown inference fields")
        if source and not (r.get("official_split") == "train" or r.get("role") == "source"):
            raise ValueError("source stages require explicit TRAIN/source provenance")
        image = Path(r["image"])
        if not image.is_absolute():
            image = Path(path).parent / image
        pixels = pixel_digest(image)
        file_hash = hashlib.sha256(image.read_bytes()).hexdigest()
        if r["image_sha256"] not in {pixels, file_hash}:
            raise ValueError("image identity mismatch")
        group = (
            r.get("patient_id")
            or r.get("slide_id")
            or r.get("study_id")
            or r.get("group_id")
            or pixels
        )
        rows.append(
            {
                "id": r["id"],
                "image": str(image.resolve()),
                "question": r["question"],
                "image_sha256": pixels,
                "group_id": group,
                "modality": r.get("modality", "mixed"),
                "task": r.get("task", "open_vqa"),
                "domain": r.get("domain", "official-train" if source else "official-test"),
                "domain_kind": r.get("domain_kind", "dataset_split"),
                "role": "source" if source else "target",
                "capability": "classification",
            }
        )
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("nonempty unique IDs required")
    groups_by_image = {}
    for row in rows:
        previous = groups_by_image.setdefault(row["image_sha256"], row["group_id"])
        if previous != row["group_id"]:
            raise ValueError("same image assigned to different independent groups")
    return rows, raw


def offline_scores(trajectories, references, raw):
    """Closed correctness and open continuous token F1 kept separate."""
    from .contribution import answer_metrics

    refs = read_json(references)
    types = {r["id"]: r.get("answer_type", "open") for r in raw}
    scores = {}
    for t in trajectories:
        ref = refs[t["id"]]
        if not isinstance(ref, list) or not ref or any(not isinstance(x, str) for x in ref):
            raise ValueError("references require nonempty string lists")
        kind = types[t["id"]]
        if kind not in {"closed", "open"}:
            raise ValueError("VQA diagnostic cannot evaluate report generation")

        def score(output, kind=kind, ref=ref):
            text = output["text"]
            if kind == "closed":
                # Full normalized answer: no SLAKE non-yes -> no coercion.
                return answer_metrics(text, ref)["exact_match"]
            return answer_metrics(text, ref)["token_f1"]

        scores[t["id"]] = {
            "kind": kind,
            "baseline": score(t["baseline"]),
            "candidate": score(t["candidate"] or t["baseline"]),
            "final": score(t["final"]),
        }
    return scores


def describe(trajectories, scores):
    from collections import Counter

    result = {
        "n": len(trajectories),
        "accepted": sum(t["action"] == "accept" for t in trajectories),
        "logical_calls": sum(t["logical_calls"] for t in trajectories),
        "actual_tool_calls": sum(t["actual_tool_calls"] for t in trajectories),
        "cache_hits": sum(t["cache_hits"] for t in trajectories),
        "candidate_calls": sum(t["candidate_calls"] for t in trajectories),
        "probe_calls": sum(t["probe_calls"] for t in trajectories),
        "fallback_reasons": dict(Counter(t["reason"] for t in trajectories)),
        "mean_seconds": sum(t["seconds"] for t in trajectories) / len(trajectories),
        "harmful_accept_rate": None,
        "task_metrics": {},
    }
    fields = [
        v
        for t in trajectories
        for s in t["steps"]
        for v in s.get("verification", {}).get("fields", {}).values()
    ]
    result["unknown_field_fraction"] = (
        sum(v["status"] == "unknown" for v in fields) / len(fields) if fields else None
    )
    result["acceptance_coverage"] = result["accepted"] / len(trajectories)
    candidates = [t for t in trajectories if t.get("candidate") is not None]
    result["candidate_coverage"] = len(candidates) / len(trajectories)
    result["candidate_text_changes"] = sum(
        t["candidate"]["text"] != t["baseline"]["text"] for t in candidates
    )
    result["routing_calls"] = sum(t.get("routing_calls", 0) for t in trajectories)
    result["routing_cache_hits"] = sum(t.get("routing_cache_hit", False) for t in trajectories)
    result["timing_totals_seconds"] = {
        key: sum(t[key] for t in trajectories) if all(key in t for t in trajectories) else None
        for key in ("baseline_seconds", "routing_seconds", "model_load_seconds", "case_wall_seconds")
    }
    result["candidate_metrics"] = {}
    if scores:
        for kind in ("closed", "open"):
            s = [scores[t["id"]] for t in candidates if scores[t["id"]]["kind"] == kind]
            if s:
                result["candidate_metrics"][kind] = {
                    "n": len(s),
                    "metric": "normalized_exact" if kind == "closed" else "token_f1",
                    "baseline": sum(v["baseline"] for v in s) / len(s),
                    "candidate": sum(v["candidate"] for v in s) / len(s),
                    "improved": sum(v["candidate"] > v["baseline"] for v in s),
                    "declined": sum(v["candidate"] < v["baseline"] for v in s),
                }
        harms = sum(
            scores[t["id"]]["final"] < scores[t["id"]]["baseline"]
            for t in trajectories
            if t["action"] == "accept"
        )
        result["harmful_accept_rate"] = harms / result["accepted"] if result["accepted"] else None
        result["harm_and_accept_rate"] = harms / len(trajectories)
        for kind in ("closed", "open"):
            s = [v for v in scores.values() if v["kind"] == kind]
            if s:
                result["task_metrics"][kind] = {
                    "n": len(s),
                    "metric": "normalized_exact" if kind == "closed" else "token_f1",
                    "baseline": sum(v["baseline"] for v in s) / len(s),
                    "final": sum(v["final"] for v in s) / len(s),
                    "net_delta": sum(v["final"] - v["baseline"] for v in s) / len(s),
                    "rescue": sum(v["final"] > v["baseline"] for v in s),
                    "harm": sum(v["final"] < v["baseline"] for v in s),
                }
    return result


def prepare(config_path, manifest, artifacts, source):
    from .capability_runtime import ValueGenerationConfig
    from .generalist_factory import generalist_provenance
    from .io import load_experiment_yaml
    from .open_study import model_provenance

    config = load_experiment_yaml(config_path)
    if config.get("medcave_enabled") is not True:
        raise ValueError("explicit medcave_enabled required")
    policy = FrozenPolicy(**config.get("risk_policy", {}))
    generation = ValueGenerationConfig(**config["generation"])
    if (
        generation.visual_views
        or generation.vector_gate != "off"
        or generation.behavior_probe != "off"
        or generation.semantic_spatial
    ):
        raise ValueError("v1 uses original-image existing semantic channel; no hidden probes")
    if not generation.token_budgeted_evidence:
        raise ValueError("real tokenizer evidence delivery audit required")
    rows, raw = manifest_rows(manifest, source)
    specs = config["experts"]
    generalist = generalist_provenance(config["generalist"], artifacts)
    experts = {k: model_provenance(s, artifacts) for k, s in specs.items()}
    # Checkpoint source identity is independent of tool alias and capability.
    fingerprints = {
        k: digest({q: v for q, v in p.items() if q != "id"}) for k, p in experts.items()
    }
    code = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in Path(__file__).parent.glob("*.py")
    }
    binding_body = {
        "config": config,
        "generalist": generalist,
        "experts": experts,
        "implementation": code,
        "signal_schema": "candidate-bound-unknown-v1",
        "evaluator": "closed-normalized-exact-open-token-f1-v1",
        "grouping": "patient-slide-study-group-pixels; minimum-id-per-group",
        "qualification_files": {
            k: hashlib.sha256(Path(s["qualification_path"]).read_bytes()).hexdigest()
            for k, s in specs.items()
            if s.get("qualification_path")
        },
    }
    return config, policy, rows, raw, digest(binding_body), fingerprints, binding_body


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--stage",
        required=True,
        choices=["dry-run", "source-dev", "source-cal", "replay", "source-smoke", "evaluate"],
    )
    p.add_argument("--config", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--artifacts", default="artifacts")
    p.add_argument("--output", required=True)
    p.add_argument("--policy")
    p.add_argument("--certificate")
    p.add_argument("--references")
    p.add_argument("--replay-run")
    p.add_argument("--limit", type=int, default=2, help="source-smoke scheduling stop only")
    args = p.parse_args()
    source = args.stage != "evaluate"
    config, policy, rows, raw, binding, fingerprints, body = prepare(
        args.config, args.manifest, args.artifacts, source
    )
    cert = read_json(args.certificate) if args.certificate else None
    frozen = read_json(args.policy) if args.policy else None
    if frozen and frozen.get("binding") != binding:
        raise ValueError("policy implementation/model/config identity changed")
    if cert and cert.get("binding") != binding:
        raise ValueError("certificate incompatible with runtime")
    if args.stage in {"source-cal", "evaluate"} and not frozen:
        raise ValueError("source-dev frozen policy artifact required")
    if args.stage == "source-cal" and not args.references:
        raise ValueError("source-cal requires offline references")
    units = {
        "groups": sorted({r["group_id"] for r in rows}),
        "images": sorted({r["image_sha256"] for r in rows}),
    }
    if args.stage == "source-cal":
        for key, values in units.items():
            if set(values) & set(frozen["dev_units"][key]):
                raise ValueError("source-dev/cal overlap")
    if args.stage == "evaluate" and cert:
        for partition in ("dev_units", "cal_units"):
            for key, values in units.items():
                if set(values) & set(cert[partition][key]):
                    raise ValueError("calibration/evaluation leakage")
    identity = digest([binding, rows, args.stage, cert])
    root = Path(args.output) / identity
    preflight = {
        "binding": binding,
        "identity": identity,
        "n": len(rows),
        "output_root": str(root.resolve()),
        "acceptance_disabled": not certificate_valid(cert, binding),
        "patient_isolation": "unverified_when_patient_ids_absent",
        "calibration_only": True,
        "bindings": body,
    }
    write_new(root / "preflight.json", preflight)
    print(json.dumps({k: v for k, v in preflight.items() if k != "bindings"}, indent=2), flush=True)
    if args.stage == "dry-run":
        return
    if args.stage == "source-dev":
        write_new(
            root / "policy.json",
            {
                "binding": binding,
                "policy": asdict(policy),
                "dev_units": units,
                "selection": "prespecified_config; no_cal_data_search",
            },
        )
    if args.stage == "replay":
        if not args.replay_run:
            raise ValueError("--replay-run required")
        donor = Path(args.replay_run)
        available = []
        for row in rows:
            path = donor / "cases" / f"{digest(row['id'])}.json"
            t = read_json(path) if path.exists() else None
            available.append({'id': row['id'], **replay_trajectory(t, row, binding, policy)})
        write_new(
            root / "replay.json", {"control_flow_only": True, "accuracy": None, "rows": available}
        )
        print(
            json.dumps(
                {
                    "exact": sum(v["status"] == "exact_trajectory_available" for v in available),
                    "replay_unavailable": sum(
                        v["status"] == "replay_unavailable" for v in available
                    ),
                }
            )
        )
        return
    from .capability_experts import CapabilityPool
    from .capability_routing import infer_image_type
    from .capability_runtime import NativeSession, NativeState, ValueGenerationConfig
    from .generalist_factory import load_generalist
    from .matched_evaluation import SharedExpertPool
    from .open_study import atomic_json

    decoder = ValueGenerationConfig(**config["generation"])
    if args.stage == "source-smoke" and args.limit < 1:
        raise ValueError("positive source-smoke limit required")
    selected = rows[: args.limit] if args.stage == "source-smoke" else rows
    probe = None
    # infer_image_type uses only pixels and a fixed prompt, never the question.
    # Scoped to this execution/model binding; never cache answers or patient facts.
    routing_cache = {}
    verifier = SourceArtifactVerifier(config["experts"], fingerprints)
    outputs = []
    for original in selected:
        case_path = root / "cases" / f"{digest(original['id'])}.json"
        if case_path.exists():
            t = read_json(case_path)
            if t["binding"] != binding or t["input_hash"] != digest(original):
                raise ValueError("incompatible cached trajectory")
            outputs.append(t)
            continue
        if probe is None:
            load_started = perf_counter()
            probe = load_generalist(config["generalist"], args.artifacts)
            pool = SharedExpertPool(
                CapabilityPool(config["experts"], args.artifacts, source_records=()),
                root / "expert-cache",
                identity,
            )
            model_load_seconds = perf_counter() - load_started
        else:
            model_load_seconds = 0.0
        case_started = perf_counter()
        row = copy.deepcopy(original)
        route = None
        routing_cache_hit = False
        routing_started = perf_counter()
        if row["modality"] == "mixed":
            key = row["image_sha256"]
            routing_cache_hit = key in routing_cache
            if not routing_cache_hit:
                routing_cache[key] = infer_image_type(probe, row)
            route = copy.deepcopy(routing_cache[key])
            row["modality"] = route["modality"]
        routing_seconds = perf_counter() - routing_started
        prompt = row["question"] + "\n" + config["prompt_suffix"]

        def generate(items, probe=probe, row=row, prompt=prompt):
            session = NativeSession(probe, row["image"], prompt, row["question"], decoder)
            block = session.propose(NativeState(items=items), decoder.max_new_tokens)
            return {
                "text": session.decode(block.tokens).strip(),
                "token_ids": list(block.tokens),
                "generation_config": asdict(decoder),
                "evidence_transport": copy.deepcopy(session.last_transport),
            }

        t = run_case(
            row,
            binding=binding,
            policy=policy,
            descriptors=admissible_descriptors(config["experts"], row, config.get("utilities", {})),
            fingerprints=fingerprints,
            pool=pool,
            generate=generate,
            certificate=cert,
            phase=args.stage,
            verifier=verifier,
        )
        t["input_hash"] = digest(original)
        t["routing"] = route
        t["routing_calls"] = int(route is not None and not routing_cache_hit)
        t["routing_cache_hit"] = routing_cache_hit
        t["routing_seconds"] = routing_seconds
        t["model_load_seconds"] = model_load_seconds
        t["case_wall_seconds"] = perf_counter() - case_started
        atomic_json(case_path, t)
        outputs.append(t)
        print(f"completed {len(outputs)}/{len(selected)}", flush=True)
    complete = len(outputs) == len(rows) and all(t["complete"] for t in outputs)
    write_new(
        root / "protocol.json",
        {
            "identity": identity,
            "binding": binding,
            "n": len(rows),
            "completed": len(outputs),
            "shards_complete": complete,
            "phase": args.stage,
            "policy": asdict(policy),
            "calibration_only": True,
        },
    )
    # Labels become available only after all requested generations have finished.
    scores = offline_scores(outputs, args.references, raw) if args.references else {}
    report = describe(outputs, scores)
    write_new(root / "diagnostics.json", describe(outputs, {}))
    if scores:
        reference_hash = hashlib.sha256(Path(args.references).read_bytes()).hexdigest()
        write_new(root / f"evaluation-{reference_hash}.json",
                  {"reference_sha256": reference_hash, "summary": report, "per_case": scores,
                   "full_manifest_complete": complete})
    if args.stage == "source-cal":
        if not complete:
            raise ValueError("cannot certify runtime failures/incomplete trajectories")
        write_new(
            root / "certificate.json",
            certify(outputs, scores, binding, policy, frozen["dev_units"]),
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
