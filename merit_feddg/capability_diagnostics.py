"""Source-only, same-prefix tests of capability, communication and composition.

No policy is fitted here and no target answers are inspected. Comparisons share
raw tool outputs, so a presentation comparison does not rerun an expert. The
best observed branch is an optimistic diagnostic, never an evaluated policy.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, replace

import numpy as np

from .capabilities import EvidenceItem
from .capability_features import action_key, state_kind
from .capability_runtime import CapabilityRuntime, NativeSession, NativeState
from .evidence_need import evidence_memory
from .experts.base import load_rgb
from .open_study import atomic_json, fingerprint


class _DuplicateSession(NativeSession):
    """Two-image format control. Not a deployed tool or fake segmentation mask."""

    def context(self, state):
        _, prompt = super().context(state)
        image = load_rgb(self.image)
        prompt += ("\nImage 1 is the unchanged original. Image 2 is an identical copy of "
                   "Image 1 for a format control; it supplies no additional medical evidence.")
        self.view_metadata = [{"view_kind": "duplicate_original_control", "sources": [],
                               "ground_truth": False}]
        return [self.image, image.copy()], prompt


def _variant_runtime(runtime, *, style="native", query="question", views=0,
                     duplicate=False, answers=False):
    config = replace(runtime.config, evidence_style=style, request_style=query,
                     visual_views=views, retrieval_answer_context=answers)
    session_class = _DuplicateSession if duplicate else NativeSession
    old = runtime.session
    session = session_class(old.probe, old.image, old.prompt, old.question, config)
    return CapabilityRuntime(session, runtime.pool, runtime.row, runtime.specs, config)


def _raw_state(after, event, previous):
    # Preserve every validated native item for a fair presentation comparison,
    # including observations omitted by one particular presentation budget.
    raw = tuple(EvidenceItem(**item) for item in event.get("native_evidence", []))
    return replace(after, items=raw + previous.items)


def collect_diagnostic_case(runtime, references, scorer, *, pairs=(), continuations=True):
    if runtime.row["role"] != "source":
        raise ValueError("diagnostics may only run on source cases")
    if (not isinstance(pairs, (list, tuple)) or any(
        not isinstance(pair, (list, tuple)) or len(pair) != 2 or pair[0] == pair[1]
        or any(not isinstance(name, str) or name not in runtime.specs for name in pair)
        for pair in pairs
    )):
        raise ValueError("pairs must name two distinct configured experts in execution order")
    if len({tuple(pair) for pair in pairs}) != len(pairs):
        raise ValueError("duplicate pair declarations")
    if pairs and runtime.config.max_expert_calls < 2:
        raise ValueError("pair diagnostics require a two-call budget")
    plain = _variant_runtime(runtime)
    initial = NativeState()
    baseline = plain.complete(initial)
    blocked = plain.run("block_none")
    if baseline["token_ids"] != blocked["token_ids"]:
        raise RuntimeError("block-NONE differs from baseline; stop before evidence diagnostics")
    states = [initial]
    end = runtime.config.block_tokens
    if continuations and len(baseline["token_ids"]) > end:
        states.append(replace(initial, prefix=tuple(baseline["token_ids"][:end])))
    branches, compositions, events, skipped_pairs = [], [], [], []
    for state in states:
        state_id = fingerprint(asdict(state))
        base = baseline if not state.prefix else plain.complete(state)
        base_score = scorer(runtime.row, base, references)
        descriptors = plain.descriptors(state)
        lookup = {d["expert"]: d for d in descriptors}
        if len(lookup) != len(descriptors):
            raise ValueError("diagnostic profile requires one declared scope per expert")
        executed, outputs = {}, {}

        def execute(start, descriptor, query, executed=executed):
            key = (fingerprint(asdict(start)), action_key(runtime.row, descriptor), query)
            if key not in executed:
                engine = _variant_runtime(runtime, query=query)
                after, event = engine.execute(start, descriptor)
                if str(event.get("reason", "")).startswith("runtime_error:"):
                    raise RuntimeError(f"Source diagnostic tool failed: {event['reason']}")
                events.append(event)
                executed[key] = (_raw_state(after, event, start), event)
            return executed[key]

        def observe(name, after, *, style="native", query="question", views=0,
                    duplicate=False, answers=False, tools=(), tool_events=(),
                    state=state, state_id=state_id, base=base, base_score=base_score,
                    outputs=outputs):
            engine = _variant_runtime(runtime, style=style, query=query, views=views,
                                      duplicate=duplicate, answers=answers)
            output = engine.complete(after)
            if output["token_ids"][:len(state.prefix)] != list(state.prefix):
                raise RuntimeError("diagnostic changed committed prefix")
            score = scorer(runtime.row, output, references)
            memory = evidence_memory(after.items, runtime.row["question"], engine.config)
            branch = {
                "name": name, "state_id": state_id, "state_kind": state_kind(state),
                "prefix_tokens": list(state.prefix), "tools": list(tools),
                "request_style": query, "presentation": style, "duplicate_control": duplicate,
                "without": base, "with": output, "base_quality": base_score,
                "quality": score, "gain": score - base_score, "presented_memory": memory,
                "native_evidence": [asdict(item) for item in after.items],
                "tool_events": list(tool_events),
                "has_visible_evidence": bool(memory or any(
                    view.get("sources") for view in output.get("visual_evidence", [])
                )),
                "timing_note": "presentation replay reuses tool output; not online method latency",
            }
            branches.append(branch)
            outputs[name] = branch
            return branch

        observe("format:duplicate_original", state, duplicate=True)
        for descriptor in descriptors:
            name, capability = descriptor["expert"], descriptor["capability"]
            # Only generative adapters get a changed query. Fixed-catalog tools
            # ignore query; retrieval should continue embedding the exact question.
            queries = ("question", "need") if capability == "generation" else ("question",)
            for query in queries:
                after, event = execute(state, descriptor, query)
                stem = f"single:{name}:{query}"
                common = {"query": query, "tools": (name,), "tool_events": (event,)}
                observe(stem + ":native_text", after, **common)
                observe(stem + ":scoped_text", after, style="scoped", **common)
                if capability == "retrieval":
                    observe(stem + ":scoped_text_answers", after, style="scoped", answers=True, **common)
                if capability in {"segmentation", "detection"}:
                    observe(stem + ":native_overlay", after, views=1, **common)
                    observe(stem + ":scoped_overlay", after, style="scoped", views=1, **common)
                    observe(stem + ":scoped_text_duplicate", after, style="scoped",
                            duplicate=True, **common)
        for first, second in pairs:
            if first not in lookup or second not in lookup:
                skipped_pairs.append({"state_id": state_id, "pair": [first, second],
                                      "reason": "incompatible_modality_task_or_scope"})
                continue
            # Match singleton request/presentation for meaningful interaction.
            qa = "need" if lookup[first]["capability"] == "generation" else "question"
            qb = "need" if lookup[second]["capability"] == "generation" else "question"
            after_a, event_a = execute(state, lookup[first], qa)
            after_ab, event_b = execute(after_a, lookup[second], qb)
            paired = observe(f"pair:{first}->{second}:scoped_text", after_ab,
                             style="scoped", query=qb, tools=(first, second),
                             tool_events=(event_a, event_b))
            a = outputs[f"single:{first}:{qa}:scoped_text"]
            b = outputs[f"single:{second}:{qb}:scoped_text"]
            compositions.append({
                "state_id": state_id, "state_kind": state_kind(state), "pair": [first, second],
                "joint_gain": paired["gain"], "first_gain": a["gain"], "second_alone_gain": b["gain"],
                "second_given_first_gain": paired["quality"] - a["quality"],
                "interaction": paired["gain"] - a["gain"] - b["gain"],
                "beats_best_single": paired["quality"] > max(a["quality"], b["quality"]),
                "both_executed": event_a["executed"] and event_b["executed"],
                "interpretation": "joint evidence presentation, NOT ROI handoff or medical causality",
            })
    return {"role": "source", "sample_id": runtime.row["id"], "domain": runtime.row["domain"],
            "group_id": runtime.row["group_id"], "modality": runtime.row["modality"],
            "domain_kind": runtime.row["domain_kind"], "baseline": baseline,
            "branches": branches, "compositions": compositions, "tool_events": events,
            "skipped_pairs": skipped_pairs, "block_none_exact": True,
            "target_generations": 0}


def diagnostic_summary(cases):
    """Equal independent-group weights; repeated prefixes never increase support."""
    grouped, pairs = defaultdict(list), defaultdict(list)
    for case in cases:
        if case.get("role") != "source":
            raise ValueError("only source diagnostics can be summarized")
        for branch in case["branches"]:
            grouped[(branch["name"], case["domain"], branch["state_kind"])].append(
                (case["group_id"], branch["gain"]))
        for pair in case["compositions"]:
            pairs[("->".join(pair["pair"]), case["domain"], pair["state_kind"])].append(
                (case["group_id"], pair))

    def means(entries):
        groups = defaultdict(list)
        for group, value in entries:
            groups[group].append(value)
        return np.asarray([np.mean(values) for values in groups.values()], dtype=float)

    rows = []
    for (name, domain, kind), entries in sorted(grouped.items()):
        values = means(entries)
        rows.append({"name": name, "domain": domain, "state_kind": kind,
                     "independent_groups": len(values),
                     "branch_count": len(entries), "mean_gain": float(values.mean()),
                     "lexical_improved_groups": int((values > 0).sum()),
                     "lexical_harmed_groups": int((values < 0).sum())})
    pair_rows = []
    for (name, domain, kind), entries in sorted(pairs.items()):
        result = {"pair": name, "domain": domain, "state_kind": kind,
                  "independent_groups": len({group for group, _ in entries})}
        for key in ("joint_gain", "second_given_first_gain", "interaction", "beats_best_single"):
            result[key] = float(means([(g, p[key]) for g, p in entries]).mean())
        pair_rows.append(result)
    return {"presentation_by_domain": rows, "pairs_by_domain": pair_rows,
            "source_cases": len(cases), "target_generations": 0,
            "domain_kinds": sorted({case["domain_kind"] for case in cases}),
            "policy_fitted": False,
            "warning": "Descriptive source-only lexical diagnostics, NOT hallucination reduction or DG proof."}


def write_diagnostics(root, cases, rows, routes):
    root.mkdir(parents=True, exist_ok=True)
    report = diagnostic_summary(cases)
    atomic_json(root / "source-diagnostics.json", cases)
    atomic_json(root / "diagnostic-summary.json", report)
    lookup = {row["id"]: row for row in rows}
    annotations, keys, routing = [], [], []
    for case in cases:
        row = lookup[case["sample_id"]]
        routing.append({"sample_id": row["id"], "image": row["image"],
                        "audited_image_type": None, "audit_note": None})
        for branch in case["branches"]:
            identity = fingerprint({"id": row["id"], "state": branch["state_id"],
                                    "name": branch["name"]})[:20]
            annotations.append({
                "annotation_id": identity, "image": row["image"], "question": row["question"],
                "response": branch["with"]["text"], "observations": branch["presented_memory"],
                "visual_evidence": branch["with"].get("visual_evidence", []),
                "observation_correctness": None, "question_relevance": None,
                "answer_factuality": None, "unsupported_claims": None, "clinical_harm": None,
                "note": "Method hidden; tool identity in native observations is NOT blinded.",
            })
            keys.append({"annotation_id": identity, "sample_id": row["id"],
                         "name": branch["name"], "state_id": branch["state_id"]})
        # Baseline must be rated too; lexical gains alone cannot establish rescue/harm.
        identity = fingerprint({"id": row["id"], "name": "generalist"})[:20]
        annotations.append({"annotation_id": identity, "image": row["image"],
                            "question": row["question"], "response": case["baseline"]["text"],
                            "observations": [], "visual_evidence": [],
                            "observation_correctness": None, "question_relevance": None,
                            "answer_factuality": None, "unsupported_claims": None,
                            "clinical_harm": None})
        keys.append({"annotation_id": identity, "sample_id": row["id"], "name": "generalist"})
    atomic_json(root / "evidence-audit.json", sorted(annotations, key=lambda item: item["annotation_id"]))
    atomic_json(root / "evidence-audit-key-private.json", keys)
    atomic_json(root / "routing-audit.json", routing)
    atomic_json(root / "routing-predictions-private.json", routes)
    lines = ["# Source capability diagnostics", "", report["warning"], "",
             "| Presentation | Source domain | State | Independent groups | Mean paired gain |",
             "| --- | --- | --- | ---: | ---: |"]
    for row in report["presentation_by_domain"]:
        lines.append(f"| {row['name']} | {row['domain']} | {row['state_kind']} | {row['independent_groups']} | "
                     f"{row['mean_gain']:+.5f} |")
    lines += ["", "Pairs are joint evidence presentations, not proof of native tool-to-tool transfer.",
              "No fitted gate, target generation, clinical safety guarantee or automatic go/no-go verdict.",
              "See evidence-audit.json for separate correctness, relevance and factuality review."]
    (root / "diagnostic-summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"summary": str(root / "diagnostic-summary.json"), **report}
