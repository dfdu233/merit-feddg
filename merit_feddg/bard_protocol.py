"""Matched-evaluation adapter for BARD isolated expert decoding."""
from __future__ import annotations

from dataclasses import asdict
from time import perf_counter

from .bard import BARDConfig, decode_bard

BARD_METHODS = ("isolated_mean", "isolated_geomedian", "bard")


def acquire_expert_groups(runtime):
    """Acquire a frozen budget of tools from clean independent states.

    Every native request is executed from an empty answer/evidence state, so one
    specialist cannot influence another specialist's request.  Multiple
    capabilities from the same expert_id are grouped as one Byzantine node to
    avoid double-counting correlated outputs from one underlying model.
    """
    from .capability_runtime import NativeState

    initial = NativeState()
    descriptors = runtime.descriptors(initial)[: runtime.config.max_expert_calls]
    groups, events, order = {}, [], []
    started = perf_counter()
    for descriptor in descriptors:
        expert = descriptor["expert"]
        if expert not in groups:
            groups[expert] = []
            order.append(expert)
        state, event = runtime.execute(initial, descriptor)
        events.append(event)
        existing = {
            (item.expert_id, item.evidence_id) for item in groups[expert]
        }
        for item in state.items:
            key = (item.expert_id, item.evidence_id)
            if key not in existing:
                groups[expert].append(item)
                existing.add(key)
    groups = {
        expert: tuple(groups[expert])
        for expert in order
        if groups[expert]
    }
    return {
        "groups": groups,
        "events": events,
        "native_requests": len(descriptors),
        "seconds": perf_counter() - started,
        "selection": (
            "first compatible descriptors under frozen max_expert_calls; "
            "grouped by expert_id"
        ),
        "answer_labels_used": False,
    }


def build_isolated_sessions(native_session, groups):
    """Construct one receiver context per expert and audit actual delivery."""
    from .capability_runtime import NativeState

    config = native_session.config
    if (
        config.evidence_style != "semantic"
        or config.semantic_spatial
        or config.visual_views
        or not config.token_budgeted_evidence
    ):
        raise ValueError(
            "BARD v1 requires token-budgeted semantic-only isolated branches"
        )
    probe = native_session.probe
    base = probe.new_answer_session(native_session.image, native_session.prompt)
    sessions, audits = {}, {}
    for expert, items in groups.items():
        # Reuse the normal matched-evaluation packer, but never put packets from
        # two experts in one receiver branch.
        temporary = type(native_session)(
            probe,
            native_session.image,
            native_session.prompt,
            native_session.question,
            config,
        )
        image, prompt = temporary.context(NativeState(items=tuple(items)))
        presented = {
            (value["expert_id"], value["evidence_id"])
            for value in temporary.last_transport.get("presented", [])
        }
        visible = tuple(
            item
            for item in items
            if (item.expert_id, item.evidence_id) in presented
        )
        audits[expert] = {
            "requested_items": len(items),
            "presented_items": len(visible),
            "transport": temporary.last_transport,
        }
        if not visible:
            continue
        if {item.expert_id for item in visible} != {expert}:
            raise ValueError("isolated branch contains evidence from another expert")
        sessions[expert] = probe.new_answer_session(image, prompt)
    return base, sessions, audits


def run_bard_method(native_session, acquisition, bard_config, method):
    """Run one matched isolated-expert ablation using shared native acquisitions."""
    if method not in BARD_METHODS:
        raise ValueError("unknown BARD matched arm")
    config = BARDConfig(**bard_config)
    base, experts, transport = build_isolated_sessions(
        native_session, acquisition["groups"]
    )
    mapping = {
        "isolated_mean": ("mean", False),
        "isolated_geomedian": ("geometric_median", False),
        "bard": ("geometric_median", True),
    }
    aggregation, bounded = mapping[method]
    started = perf_counter()
    result = decode_bard(
        base,
        experts,
        max_tokens=native_session.config.max_new_tokens,
        config=config,
        aggregation=aggregation,
        bounded_commit=bounded,
        fault_probe=method == "bard",
    )
    decode_seconds = perf_counter() - started
    evidence = [
        asdict(item)
        for values in acquisition["groups"].values()
        for item in values
    ]
    return {
        **result,
        "finished": bool(
            result["token_ids"] and result["token_ids"][-1] in base.eos_ids
        ),
        "seconds": decode_seconds,
        # Native specialist calls are shared across the matched arms.  This
        # field records the acquisition workload, not repeated calls per arm.
        "expert_calls": acquisition["native_requests"],
        "controller_calls": 0,
        "probe_model_calls": 0,
        "probe_seconds": 0.0,
        "controller_output_tokens": 0,
        "evidence": evidence,
        "adopted_evidence_count": sum(
            value["presented_items"] > 0 for value in transport.values()
        ),
        "presented_evidence_count": sum(
            value["presented_items"] for value in transport.values()
        ),
        "isolated_transport": transport,
        "shared_acquisition": {
            "native_requests": acquisition["native_requests"],
            "seconds": acquisition["seconds"],
            "selection": acquisition["selection"],
            "events": acquisition["events"],
        },
        "byzantine_model": {
            "fault_budget": config.fault_budget,
            "required_nodes_for_commit": (
                3 * config.fault_budget + 1
                if config.fault_budget
                else 1
            ),
            "node": "expert_id",
            "medical_correctness_guaranteed": False,
        },
    }
