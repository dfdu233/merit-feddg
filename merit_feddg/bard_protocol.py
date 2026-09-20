"""Matched-evaluation adapter for BARD isolated expert decoding."""
from __future__ import annotations

from dataclasses import asdict
from time import perf_counter

from .bard import BARDConfig, bard_step, decode_bard, decode_bard_incremental
from .bard import decode_bard_bundle as decode_bundle
from .evidence_decode import _log_probs

BARD_METHODS = ("isolated_mean", "isolated_geomedian", "bard")


def select_bard_descriptors(candidates, specs, max_calls):
    """Fixed answer-blind BARD acquisition schedule over compatible descriptors."""
    if type(max_calls) is not int or max_calls < 1:
        raise ValueError("max_calls must be a positive integer")

    def fault_node(descriptor):
        expert = descriptor["expert"]
        group = specs[expert].get("fault_group", expert)
        if not isinstance(group, str) or not group.strip():
            raise ValueError(f"expert {expert}: fault_group must be a nonempty string")
        return group.strip()

    primary_visual, primary_knowledge, repeated = [], [], []
    seen = set()
    for descriptor in candidates:
        node = fault_node(descriptor)
        if node in seen:
            repeated.append(descriptor)
            continue
        seen.add(node)
        if descriptor["capability"] == "retrieval":
            primary_knowledge.append(descriptor)
        else:
            primary_visual.append(descriptor)
    return (primary_visual + primary_knowledge + repeated)[:max_calls]


def acquire_expert_groups(runtime):
    """Acquire a frozen budget of tools from clean independent states.

    Every native request is executed from an empty answer/evidence state, so one
    specialist cannot influence another specialist's request.  Multiple
    capabilities from the same expert_id are grouped as one Byzantine node to
    avoid double-counting correlated outputs from one underlying model.
    """
    from .capability_runtime import NativeState

    initial = NativeState()
    candidates = runtime.descriptors(initial)
    descriptors = select_bard_descriptors(
        candidates, runtime.specs, runtime.config.max_expert_calls
    )

    def fault_node(descriptor):
        expert = descriptor["expert"]
        return str(runtime.specs[expert].get("fault_group", expert)).strip()
    groups, events, order, members = {}, [], [], {}
    started = perf_counter()
    for descriptor in descriptors:
        expert = descriptor["expert"]
        node = fault_node(descriptor)
        if node not in groups:
            groups[node] = []
            members[node] = []
            order.append(node)
        if expert not in members[node]:
            members[node].append(expert)
        state, event = runtime.execute(initial, descriptor)
        events.append({**event, "fault_group": node})
        existing = {
            (item.expert_id, item.evidence_id) for item in groups[node]
        }
        for item in state.items:
            key = (item.expert_id, item.evidence_id)
            if key not in existing:
                groups[node].append(item)
                existing.add(key)
    groups = {
        node: tuple(groups[node])
        for node in order
        if groups[node]
    }
    members = {node: members[node] for node in groups}
    return {
        "groups": groups,
        "events": events,
        "native_requests": len(descriptors),
        "seconds": perf_counter() - started,
        "selection": (
            "compatible descriptors under frozen max_expert_calls; distinct "
            "declared patient-image fault_group nodes first, then retrieval knowledge "
            "nodes, then repeated capabilities"
        ),
        "fault_group_members": members,
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
            "BARD requires token-budgeted semantic packets with branch-local native spatial transport"
        )
    probe = native_session.probe
    base = probe.new_answer_session(native_session.image, native_session.prompt)
    if hasattr(base, "prime_vision_cache"):
        base.prime_vision_cache()
    sessions, audits = {}, {}
    for node, items in groups.items():
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
        audits[node] = {
            "requested_items": len(items),
            "presented_items": len(visible),
            "transport": temporary.last_transport,
        }
        if not visible:
            continue
        visible_experts = sorted({item.expert_id for item in visible})

        spatial_records = 0
        spatial_rejected = []
        if hasattr(probe, "tensor_bridge") and hasattr(probe, "tensor_packet"):
            packet = probe.tensor_packet(
                visible,
                native_session.image,
                question=native_session.question,
                weighting=config.spatial_weighting,
            )
            spatial_records = len(packet)
            spatial_rejected = list(packet.rejected)
            if spatial_records:
                sessions[node] = probe.new_tensor_answer_session(
                    native_session.image,
                    prompt,
                    visible,
                    question=native_session.question,
                    weighting=config.spatial_weighting,
                )
            else:
                sessions[node] = probe.new_answer_session(image, prompt)
        else:
            sessions[node] = probe.new_answer_session(image, prompt)
        if hasattr(sessions[node], "share_vision_cache_from"):
            sessions[node].share_vision_cache_from(base)
        audits[node].update(
            fault_group=node,
            member_experts=visible_experts,
            receiver_channel=(
                "semantic_plus_native_spatial" if spatial_records else "semantic"
            ),
            native_spatial_records=spatial_records,
            native_spatial_rejected=spatial_rejected,
            shared_vision_features=bool(
                getattr(sessions[node], "_cached_vision_features", None) is not None
            ),
        )
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
            "fault_group_members": acquisition.get("fault_group_members", {}),
        },
        "byzantine_model": {
            "declared_fault_budget": config.fault_budget,
            "centralized_condition": "f < n/2",
            "single_expert_policy": config.single_expert_policy,
            "two_expert_policy": config.pair_policy,
            "node": "fault_group",
            "medical_correctness_guaranteed": False,
        },
    }


def run_bard_bundle(
    native_session,
    acquisition,
    bard_config,
    *,
    parity_tokens=None,
):
    """Compute matched arms; BARD may use KV only after replay parity succeeds."""
    config = BARDConfig(**bard_config)
    base, experts, transport = build_isolated_sessions(
        native_session, acquisition["groups"]
    )
    started = perf_counter()
    parity = {
        "fast_path_allowed": False,
        "reason": "receiver_mode_replay",
    }
    if config.receiver_mode == "auto" and parity_tokens:
        parity = validate_incremental_parity(
            base,
            experts,
            parity_tokens,
            max_steps=config.incremental_parity_steps,
            tolerance=config.incremental_logprob_tolerance,
            bard_config=config,
        )

    if parity.get("fast_path_allowed"):
        replay_started = perf_counter()
        bundle = decode_bundle(
            base,
            experts,
            max_tokens=native_session.config.max_new_tokens,
            config=config,
            methods=("isolated_mean", "isolated_geomedian"),
        )
        replay_seconds = perf_counter() - replay_started
        base_stream = base.new_incremental_stream()
        expert_streams = {
            name: session.new_incremental_stream()
            for name, session in experts.items()
        }
        bard_started = perf_counter()
        bundle["bard"] = decode_bard_incremental(
            base_stream,
            expert_streams,
            max_tokens=native_session.config.max_new_tokens,
            config=config,
            fault_probe=True,
        )
        bard_seconds = perf_counter() - bard_started
        bundle["bard"]["incremental_parity"] = parity
        bundle["bard"]["fast_path_activated"] = True
    else:
        bundle = decode_bundle(
            base,
            experts,
            max_tokens=native_session.config.max_new_tokens,
            config=config,
        )
        replay_seconds = perf_counter() - started
        bard_seconds = 0.0
        bundle["bard"]["incremental_parity"] = parity
        bundle["bard"]["fast_path_activated"] = False

    decode_seconds = perf_counter() - started
    evidence = [
        asdict(item)
        for values in acquisition["groups"].values()
        for item in values
    ]
    shared = {
        "native_requests": acquisition["native_requests"],
        "seconds": acquisition["seconds"],
        "selection": acquisition["selection"],
        "events": acquisition["events"],
        "fault_group_members": acquisition.get("fault_group_members", {}),
    }
    outputs = {}
    for method, result in bundle.items():
        outputs[method] = {
            **result,
            "finished": bool(
                result["token_ids"] and result["token_ids"][-1] in base.eos_ids
            ),
            "seconds": (
                bard_seconds if method == "bard" and parity.get("fast_path_allowed")
                else replay_seconds
            ),
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
            "shared_acquisition": shared,
            "byzantine_model": {
                "declared_fault_budget": config.fault_budget,
                "centralized_condition": "f < n/2",
                "single_expert_policy": config.single_expert_policy,
                "two_expert_policy": config.pair_policy,
                "node": "fault_group",
                "medical_correctness_guaranteed": False,
            },
            "bundle_decode_seconds": decode_seconds,
            "bundle_replay_ablation_seconds": replay_seconds,
            "bard_receiver_seconds": bard_seconds,
            "bundle_amortized_across_methods": list(BARD_METHODS),
        }
    return outputs


def _normalized_score_error(reference, candidate):
    ref, ref_mask = _log_probs(reference)
    cand, cand_mask = _log_probs(candidate)
    if ref.shape != cand.shape or not np.array_equal(ref_mask, cand_mask):
        return float("inf")
    return float(np.max(np.abs(ref[ref_mask] - cand[cand_mask])))


def validate_incremental_parity(
    base_session,
    expert_sessions,
    prefix_tokens,
    *,
    max_steps=8,
    tolerance=0.02,
    bard_config=None,
):
    """Require branch-distribution and BARD-decision parity before persistent KV."""
    if type(max_steps) is not int or max_steps < 1:
        raise ValueError("max_steps must be positive")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    config = bard_config or BARDConfig()
    prefix_tokens = [int(token) for token in prefix_tokens[:max_steps]]
    sessions = {"generalist": base_session, **dict(expert_sessions)}
    if not all(hasattr(session, "new_incremental_stream") for session in sessions.values()):
        return {
            "fast_path_allowed": False,
            "reason": "incremental_stream_unavailable",
            "prefix_tokens": prefix_tokens,
        }

    try:
        streams = {
            name: session.new_incremental_stream()
            for name, session in sessions.items()
        }
    except (ValueError, TypeError, RuntimeError, AttributeError) as exc:
        return {
            "fast_path_allowed": False,
            "reason": f"stream_initialization_failed:{type(exc).__name__}:{exc}",
            "prefix_tokens": prefix_tokens,
        }

    prefix = []
    rows = []
    max_error = 0.0
    all_branch_argmax = True
    all_decisions_equal = True
    expert_names = list(expert_sessions)
    for step in range(len(prefix_tokens) + 1):
        replay = {
            name: session.next_scores(prefix)
            for name, session in sessions.items()
        }
        cached = {
            name: stream.current_scores()
            for name, stream in streams.items()
        }
        errors = {
            name: _normalized_score_error(replay[name], cached[name])
            for name in sessions
        }
        max_error = max(max_error, *errors.values())
        branch_equal = all(
            int(np.argmax(replay[name])) == int(np.argmax(cached[name]))
            for name in sessions
        )
        replay_token, replay_audit = bard_step(
            replay["generalist"],
            [replay[name] for name in expert_names],
            config,
            aggregation=config.aggregation,
            bounded_commit=True,
        )
        cached_token, cached_audit = bard_step(
            cached["generalist"],
            [cached[name] for name in expert_names],
            config,
            aggregation=config.aggregation,
            bounded_commit=True,
        )
        decision_equal = (
            replay_token == cached_token
            and replay_audit["committed"] == cached_audit["committed"]
            and replay_audit["consensus_mode"] == cached_audit["consensus_mode"]
        )
        all_branch_argmax &= branch_equal
        all_decisions_equal &= decision_equal
        rows.append({
            "step": step,
            "branch_argmax_equal": branch_equal,
            "bard_decision_equal": decision_equal,
            "replay_selected_token": int(replay_token),
            "incremental_selected_token": int(cached_token),
            "max_normalized_logprob_error": max(errors.values()),
            "branch_errors": errors,
        })
        if max_error > tolerance or not branch_equal or not decision_equal:
            return {
                "fast_path_allowed": False,
                "reason": (
                    "normalized_logprob_mismatch"
                    if max_error > tolerance
                    else "bard_or_branch_decision_mismatch"
                ),
                "prefix_tokens": prefix_tokens,
                "steps": rows,
                "max_normalized_logprob_error": max_error,
                "tolerance": tolerance,
            }
        if step < len(prefix_tokens):
            token = prefix_tokens[step]
            prefix.append(token)
            for stream in streams.values():
                stream.commit(token)

    return {
        "fast_path_allowed": True,
        "reason": "replay_parity_passed",
        "prefix_tokens": prefix_tokens,
        "steps": rows,
        "all_branch_argmax_equal": all_branch_argmax,
        "all_bard_decisions_equal": all_decisions_equal,
        "max_normalized_logprob_error": max_error,
        "tolerance": tolerance,
    }


def run_bard_incremental_method(
    native_session,
    acquisition,
    bard_config,
    *,
    parity_tokens,
    parity_steps=8,
):
    """Run persistent-KV BARD only after branch-complete replay parity succeeds."""
    config = BARDConfig(**bard_config)
    base, experts, transport = build_isolated_sessions(
        native_session, acquisition["groups"]
    )
    parity = validate_incremental_parity(
        base, experts, parity_tokens, max_steps=parity_steps
    )
    if not parity["fast_path_allowed"]:
        raise RuntimeError(
            "persistent-KV BARD parity failed; keep the production replay decoder"
        )
    base_stream = base.new_incremental_stream()
    expert_streams = {
        name: session.new_incremental_stream()
        for name, session in experts.items()
    }
    started = perf_counter()
    result = decode_bard_incremental(
        base_stream,
        expert_streams,
        max_tokens=native_session.config.max_new_tokens,
        config=config,
        fault_probe=True,
    )
    decode_seconds = perf_counter() - started
    return {
        **result,
        "finished": bool(
            result["token_ids"] and result["token_ids"][-1] in base.eos_ids
        ),
        "seconds": decode_seconds,
        "expert_calls": acquisition["native_requests"],
        "controller_calls": 0,
        "probe_model_calls": 0,
        "probe_seconds": 0.0,
        "controller_output_tokens": 0,
        "evidence": [
            asdict(item)
            for values in acquisition["groups"].values()
            for item in values
        ],
        "isolated_transport": transport,
        "incremental_parity": parity,
        "fast_path_activated": True,
        "medical_correctness_guaranteed": False,
    }

