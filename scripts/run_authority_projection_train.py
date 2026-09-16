"""Free-text capability-authority projection on the frozen XRV TRAIN pilot.

This experiment does not change the generalist, specialist, prompt contract, or
previous uncertainty run. It reuses the already generated full-text candidate
answers from the fixed 12-case TRAIN pilot, scores each complete text under the
same frozen LLaVA-Med context, then applies a minimum-change finite-pool
projection only on the one XRV-native binary finding variable.

No references are loaded here. This is a mechanism experiment, not a test score.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from run_channel_soft_canary import visible
from run_uncertainty_train import context, evidence, gpu_check, resources

from merit_feddg.authority_projection import (
    NEGATIVE,
    POSITIVE,
    UNKNOWN,
    binary_finding_group,
    normalized_pool_distribution,
    project_free_text_candidates,
    stable_select,
    transport_diagnostic,
    xrv_operating_coordinate,
)
from merit_feddg.capabilities import EvidenceItem
from merit_feddg.capability_runtime import ValueGenerationConfig
from merit_feddg.generalist_factory import load_generalist
from merit_feddg.open_study import atomic_json, fingerprint

CANDIDATE_ARMS = (
    "incumbent",
    "scope_text",
    "fixed_blend",
    "fixed_cad",
    "acd",
    "source_only",
    "source_acd",
)
ALIASES = {
    "Effusion": ("pleural effusion", "effusion", "pleural fluid"),
    "Cardiomegaly": ("cardiomegaly", "cardiac enlargement", "enlarged heart"),
    "Pneumothorax": ("pneumothorax", "collapsed lung"),
}


def _candidate_pool(source_case):
    """Unique complete texts; proposal provenance is retained but never scored as truth."""
    pooled = []
    by_tokens = {}
    for arm in CANDIDATE_ARMS:
        candidate = source_case["arms"].get(arm)
        if not candidate:
            continue
        tokens = tuple(int(token) for token in candidate.get("token_ids", ()))
        text = str(candidate.get("text", "")).strip()
        if not tokens or not text:
            continue
        if tokens in by_tokens:
            pooled[by_tokens[tokens]]["source_arms"].append(arm)
            continue
        by_tokens[tokens] = len(pooled)
        pooled.append({"text": text, "token_ids": list(tokens), "source_arms": [arm]})
    if not pooled:
        raise RuntimeError("source run provided no full-text candidates")
    return pooled


def _json_projection(result):
    cleaned = dict(result)
    for key in ("base_probabilities", "projected_probabilities"):
        if key in cleaned:
            cleaned[key] = [float(value) for value in cleaned[key]]
    return cleaned


def _official_operating_points():
    import torchxrayvision as xrv

    metadata = xrv.models.model_urls["densenet121-res224-all"]
    labels = list(metadata["labels"])
    thresholds = list(metadata["op_threshs"])
    mapping = {}
    for label, threshold in zip(labels, thresholds):
        if label and np.isfinite(threshold):
            mapping[label] = float(threshold)
    return mapping


def _reconstruct_sessions(probe, row, historical, protocol, source_case, source):
    cfg = ValueGenerationConfig(**historical["generation_config"])
    native = tuple(EvidenceItem(**item) for item in historical["evidence"])
    initial, transport = context(probe, row, protocol, cfg, native)
    original_ids = visible(transport)
    kept = tuple(
        item for item in native if (item.expert_id, item.evidence_id) in original_ids
    )
    base, base_transport = context(probe, row, protocol, cfg, kept)
    focused, focused_transport = context(
        probe,
        row,
        protocol,
        cfg,
        kept + (evidence(row["id"], source, True),),
    )
    expected = original_ids | {("xrv_native_uncertainty", "xrv-native:" + row["id"])}
    if visible(base_transport) != original_ids:
        raise RuntimeError("base evidence transport changed")
    if visible(focused_transport) != expected:
        raise RuntimeError("focused XRV evidence transport changed")
    if source_case["historical_token_parity"] is not True:
        raise RuntimeError("source case lacks historical incumbent parity")
    return initial, base, focused


def _run_case(probe, row, historical, protocol, source_case, source, threshold):
    _, base, focused = _reconstruct_sessions(
        probe, row, historical, protocol, source_case, source
    )
    pool = _candidate_pool(source_case)
    aliases = ALIASES[source["label"]]
    groups = [binary_finding_group(candidate["text"], aliases) for candidate in pool]

    base_scores = [
        base.sequence_mean_logp((), tuple(candidate["token_ids"])) for candidate in pool
    ]
    text_scores = [
        focused.sequence_mean_logp((), tuple(candidate["token_ids"])) for candidate in pool
    ]
    coordinate = xrv_operating_coordinate(source["probability"], threshold)
    projection = project_free_text_candidates(
        [candidate["text"] for candidate in pool],
        base_scores,
        groups,
        coordinate,
    )
    base_probabilities = normalized_pool_distribution(base_scores)
    text_probabilities = normalized_pool_distribution(text_scores)
    text_index = stable_select(text_probabilities, base_probabilities)

    for candidate, group, base_score, text_score, p0, pe in zip(
        pool, groups, base_scores, text_scores, base_probabilities, text_probabilities
    ):
        candidate.update(
            semantic_group=group,
            base_mean_logp=float(base_score),
            text_mean_logp=float(text_score),
            base_pool_probability=float(p0),
            text_pool_probability=float(pe),
        )

    return {
        "id": row["id"],
        "source": {
            "label": source["label"],
            "raw_sigmoid": float(source["probability"]),
            "official_operating_point": float(threshold),
            "native_decision_coordinate": float(coordinate),
            "calibrated_probability": False,
        },
        "candidate_pool": pool,
        "candidate_pool_arms": list(CANDIDATE_ARMS),
        "projection": _json_projection(projection),
        "transport": transport_diagnostic(base_scores, text_scores, groups, coordinate),
        "outputs": {
            "incumbent": source_case["arms"]["incumbent"]["text"],
            "scope_text": source_case["arms"]["scope_text"]["text"],
            "base_rerank": pool[projection["base_index"]]["text"],
            "text_rerank": pool[text_index]["text"],
            "authority_projection": projection["selected_text"],
        },
        "selected": {
            "base_index": int(projection["base_index"]),
            "text_index": int(text_index),
            "authority_index": int(projection["selected_index"]),
            "base_group": groups[projection["base_index"]],
            "text_group": groups[text_index],
            "authority_group": groups[projection["selected_index"]],
        },
        "mapped_group_counts": {
            POSITIVE: groups.count(POSITIVE),
            NEGATIVE: groups.count(NEGATIVE),
            UNKNOWN: groups.count(UNKNOWN),
        },
        "free_text_output": True,
        "references_loaded": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--max-cases", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("output must be a fresh directory")
    if args.max_cases < 0:
        raise ValueError("max-cases must be nonnegative")

    source_frozen = json.loads((args.source_run / "frozen.json").read_text())
    pilot = json.loads((args.source_run / "pilot-complete.json").read_text())
    if not pilot.get("pilot_complete") or pilot.get("full_manifest_complete"):
        raise RuntimeError("expected the completed fixed TRAIN pilot, not a full/test run")
    source_data = json.loads((args.source_run / "sources.json").read_text())
    source_case_files = {
        path.stem: path for path in (args.source_run / "cases").glob("*.json")
    }
    selected = list(source_frozen["selected"])
    selected_ids = [item["id"] for item in selected]
    if set(source_case_files) != set(selected_ids):
        raise RuntimeError("source pilot case set is incomplete or changed")

    protocol, rows, historical = resources()
    lookup = {row["id"]: row for row in rows}
    if any(key not in lookup or key not in historical for key in selected_ids):
        raise RuntimeError("source pilot does not align with the frozen TRAIN resources")

    operating_points = _official_operating_points()
    labels = {source_data["cases"][key]["label"] for key in selected_ids}
    if not labels <= set(operating_points) or not labels <= set(ALIASES):
        raise RuntimeError("native label lacks a declared operating point or semantic aliases")

    config = {
        "schema": "capability-authority-free-text-v1",
        "source_run": str(args.source_run.resolve()),
        "source_identity": source_data["identity"],
        "selected": selected,
        "candidate_arms": list(CANDIDATE_ARMS),
        "projection": "finite-pool minimum-KL group marginal projection",
        "unknown_policy": "preserve exact base unknown mass",
        "within_group_policy": "preserve exact base relative odds",
        "source_coordinate": "official XRV operating-point normalization; not calibrated",
        "free_text_output": True,
        "fit": False,
        "test_used": False,
        "references_loaded_during_generation": False,
    }
    identity = fingerprint(config)
    args.output.mkdir(parents=True)
    (args.output / "cases").mkdir()
    atomic_json(args.output / "frozen.json", config)

    gpu_check(args.gpu_uuid)
    started = time.perf_counter()
    probe = load_generalist(protocol["config"]["generalist"], "artifacts")
    probe.model.eval().requires_grad_(False)
    atomic_json(
        args.output / "load.json",
        {"seconds": time.perf_counter() - started, "gpu_uuid": args.gpu_uuid},
    )

    completed = 0
    with torch.inference_mode():
        for item in selected:
            if args.max_cases and completed >= args.max_cases:
                break
            key = item["id"]
            source_case = json.loads(source_case_files[key].read_text())
            source = source_data["cases"][key]
            result = _run_case(
                probe,
                lookup[key],
                historical[key],
                protocol,
                source_case,
                source,
                operating_points[source["label"]],
            )
            atomic_json(
                args.output / "cases" / f"{key}.json",
                {"identity": identity, **result},
            )
            completed += 1
            print(
                "DONE",
                key,
                "pool",
                len(result["candidate_pool"]),
                "groups",
                result["mapped_group_counts"],
                "projection",
                result["projection"]["status"],
                flush=True,
            )

    atomic_json(
        args.output / "generation-complete.json",
        {
            "identity": identity,
            "completed": completed,
            "expected": len(selected),
            "pilot_complete": completed == len(selected),
            "partial": completed != len(selected),
        },
    )


if __name__ == "__main__":
    main()
