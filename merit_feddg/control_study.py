"""Matched frozen spatial CRES experiment; labels belong only to offline scoring."""
from __future__ import annotations

import copy
import math
from dataclasses import asdict, replace
from time import perf_counter

from .capabilities import EvidenceItem
from .capability_runtime import NativeSession, NativeState, ValueGenerationConfig
from .control_evidence import decode_controlled, spatial_controls
from .matched_evaluation import generation_prompt
from .soft_guidance import decode

ARMS = ("generalist", "compact", "deletion", "native_fixed", "null_fixed", "cres")
ABLATIONS = ("kl_only", "residual_unbounded")
UNIFORM_SUFFIX = "Give only the short answer. Do not explain."


def validate_outputs(record, names):
    arms = record.get("arms", {})
    if set(arms) != set(names):
        raise ValueError("incomplete arm outputs")
    for value in arms.values():
        tokens = value.get("token_ids")
        if (not isinstance(value.get("text"), str) or not value["text"].strip()
                or not isinstance(tokens, list) or not 1 <= len(tokens) <= 64
                or any(type(t) is not int or t < 0 for t in tokens)
                or not isinstance(value.get("seconds"), (int, float))
                or not math.isfinite(value["seconds"]) or value["seconds"] < 0):
            raise ValueError("invalid/empty arm output")


def prompt_config(config, contract):
    if contract == "uniform":
        return {**config, "prompt_contract": "legacy_suffix", "prompt_suffix": UNIFORM_SUFFIX}
    if contract != "historical":
        raise ValueError("declare uniform or historical prompt contract")
    return config


def reuse(value, reason):
    return {"text": value["text"], "token_ids": list(value["token_ids"]), "seconds": 0.0,
            "guidance_applied": False, "reused_control": True, "reason": reason,
            "score_calls": 0, "steps": []}


def control_case(probe, row, historical, protocol, options):
    """Generate fresh comparators. Never compare a changed prompt to old scores."""
    import torch
    from PIL import Image

    started = perf_counter()
    if any(k in row for k in ("answer", "answers", "reference", "references", "label")):
        raise ValueError("inference row contains labels")
    source_cfg = ValueGenerationConfig(**historical["generation_config"])
    if (source_cfg.semantic_spatial or source_cfg.vector_gate != "off"
            or source_cfg.evidence_style != "semantic" or not source_cfg.compact_native):
        raise ValueError("expected ungated compact native source cache")
    cfg = replace(source_cfg, max_new_tokens=64, block_tokens=64,
                  behavior_probe="off", uncertainty_from_probe=False)
    native = tuple(EvidenceItem(**i) for i in historical["evidence"])
    # Bind cached expert outputs to the original question/delivery, before any
    # declared prompt change. This is rendering only, not a historical decode.
    old = NativeSession(probe, row["image"], generation_prompt(row, protocol["config"]),
                        row["question"], source_cfg)
    old.context(NativeState(items=native))
    recorded = next((event["evidence_transport"] for event in reversed(historical.get("trace", []))
                     if event.get("event") == "decode" and event.get("evidence_transport")), None)
    if not recorded:
        raise ValueError("source cache lacks prompt/evidence binding audit")
    for key in ("prompt_sha256", "evidence_sha256", "presented", "omitted", "context"):
        if key not in recorded or recorded[key] != old.last_transport.get(key):
            raise ValueError("source cache delivery mismatch: " + key)

    config = prompt_config(protocol["config"], options["prompt_contract"])
    prompt = generation_prompt(row, config)
    text = NativeSession(probe, row["image"], prompt, row["question"], cfg)
    with torch.inference_mode():
        before = perf_counter()
        block = text.propose(NativeState(items=native), cfg.max_new_tokens)
    arms = {"compact": {"text": text.decode(block.tokens).strip(), "token_ids": list(block.tokens),
                        "seconds": perf_counter() - before}}
    eos = probe.tokenizer.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    generalist = probe.new_answer_session(row["image"], prompt)
    arms["generalist"] = decode(generalist, None, alpha=0, max_tokens=64, eos_ids=eos_ids)
    visible = {(i["expert_id"], i["evidence_id"]) for i in text.last_transport["presented"]}
    presented = tuple(i for i in native if (i.expert_id, i.evidence_id) in visible)
    masks = tuple(i for i in presented if i.capability == "segmentation")
    other = tuple(i for i in presented if i.capability != "segmentation")
    packet = probe.tensor_packet(masks, row["image"], question=row["question"], weighting="equal")
    result = {"id": row["id"], "arms": arms, "generation_config": asdict(cfg),
              "text_transport": text.last_transport, "new_expert_calls": 0,
              "applicable": bool(len(packet)), "retained_other_sources": [i.expert_id for i in other],
              "source_delivery_verified": True, "historical_predictions_are_comparators": False}
    names = (*ARMS, *ABLATIONS) if options["ablations"] else ARMS
    if not len(packet):
        for name in names:
            if name not in arms:
                arms[name] = reuse(arms["compact"], "no_presented_spatial_operator")
        result["controls_available"] = False
    else:
        context = NativeSession(probe, row["image"], prompt, row["question"], cfg)
        image, base_prompt = context.context(NativeState(items=other))
        plain = probe.new_answer_session(image, base_prompt)
        spatial = probe.new_tensor_answer_session(image, base_prompt, masks,
                                                  question=row["question"], weighting="equal")
        arms["deletion"] = decode(plain, None, alpha=0, max_tokens=64, eos_ids=eos_ids)
        with torch.inference_mode():
            before = perf_counter()
            direct = plain.propose((), count=1, length=64)[0]
        result["parity_check_seconds"] = perf_counter() - before
        if list(direct.tokens) != arms["deletion"]["token_ids"]:
            raise RuntimeError("fresh zero-strength token parity failed")
        result["zero_parity"] = True
        arms["native_fixed"] = decode(plain, spatial, alpha=options["fixed_strength"],
                                      max_tokens=64, eos_ids=eos_ids)
        result["native_operator_audit"] = dict(probe.tensor_bridge.last_audit)
        with Image.open(row["image"]) as source_image:
            controls, result["control_audit"] = spatial_controls(packet, source_image.size)
        sessions = []
        for control in controls:
            # Replay sessions have immutable input tensors and no per-prefix KV
            # state; only the packet used by their scoped projector hook changes.
            session = copy.copy(spatial)
            session.tensor_packet = control
            sessions.append(session)
        result["controls_available"] = bool(sessions)
        if sessions:
            arms["null_fixed"] = decode(plain, sessions[0], alpha=options["fixed_strength"],
                                        max_tokens=64, eos_ids=eos_ids)
        else:
            arms["null_fixed"] = reuse(arms["deletion"], "degenerate_control_family")
        kwargs = {"max_tokens": 64, "eos_ids": eos_ids, "max_strength": options["max_strength"],
                  "kl_budget": options["kl_budget"]}
        arms["cres"] = decode_controlled(plain, spatial, sessions, **kwargs)
        if options["ablations"]:
            arms["kl_only"] = decode_controlled(plain, spatial, (), mode="raw", **kwargs)
            arms["residual_unbounded"] = decode_controlled(
                plain, spatial, sessions, **{**kwargs, "kl_budget": 1e6})
    validate_outputs(result, names)
    result["wall_seconds"] = perf_counter() - started
    return result
