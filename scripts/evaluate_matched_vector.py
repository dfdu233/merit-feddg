#!/usr/bin/env python3
"""Offline fixed-reference audit for a completed matched vector experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

from merit_feddg.contribution import answer_metrics
from merit_feddg.open_study import atomic_json

LEADING_BINARY = re.compile(r"^\s*[-*'\"`(\[]*\s*(?:answer\s*:\s*)?(yes|no)\b", re.IGNORECASE)


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--references", required=True, type=Path)
    parser.add_argument("--anchor-root", default="/home/dbw/ANCHOR", type=Path)
    parser.add_argument("--source-audit", type=Path)
    parser.add_argument("--training-report", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.anchor_root))
    from anchor.corrected_sgta.evaluate_medheval_answers import PROTOCOL_VERSION, evaluate_rows
    from anchor.medeval.evaluate_mixed_vqa_table import answer_token_recall

    manifest = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    references = json.loads(args.references.read_text())
    protocol = json.loads((args.run / "protocol.json").read_text())
    names = tuple(protocol["methods"])
    all_name = next((n for n in ("compact_all", "hybrid_all", "spatial_weighted", "tensor_all") if n in names), None)
    gate_name = next((n for n in ("compact_verified", "hybrid_gate", "spatial_gate", "tensor_gate") if n in names), None)
    if not {"generalist", all_name, gate_name}.issubset(names):
        raise ValueError("unsupported matched evidence protocol")
    outputs = {name: json.loads((args.run / f"{name}.json").read_text()) for name in names}
    ids = [row["id"] for row in manifest]
    if any(set(outputs[name]) != set(ids) for name in names) or set(references) != set(ids):
        raise ValueError("manifest, references and all method outputs must align exactly")

    evaluated, per_case = {}, {}
    for name in names:
        rows = [{"qid": row["id"], "question": row["question"],
                 "answer": references[row["id"]][0], "answer_type": row["answer_type"],
                 "text": outputs[name][row["id"]]["text"],
                 "image_sha256": row["image_sha256"]} for row in manifest]
        report = evaluate_rows(rows)
        detail = {row["question_id"]: row for row in report["details"]}
        generic = [answer_metrics(outputs[name][sample_id]["text"], references[sample_id])
                   for sample_id in ids]
        strict, content, opened = {}, {}, {}
        for row in manifest:
            sample_id, ref = row["id"], references[row["id"]][0]
            if row["answer_type"] == "closed":
                expected = "yes" if str(ref).strip().casefold() == "yes" else "no"
                match = LEADING_BINARY.search(outputs[name][sample_id]["text"])
                strict[sample_id] = float(bool(match and match.group(1).casefold() == expected))
                content[sample_id] = float(detail[sample_id]["correct"])
            else:
                value = answer_token_recall(outputs[name][sample_id]["text"], ref)
                strict[sample_id] = content[sample_id] = opened[sample_id] = value
        closed_ids = [row["id"] for row in manifest if row["answer_type"] == "closed"]
        evaluated[name] = {
            "n": len(ids), "generic_exact_match": statistics.mean(x["exact_match"] for x in generic),
            "generic_token_f1": statistics.mean(x["token_f1"] for x in generic),
            "strict_mixed": statistics.mean(strict.values()),
            "strict_closed_accuracy": statistics.mean(strict[i] for i in closed_ids),
            "strict_closed_parseable": sum(bool(LEADING_BINARY.search(outputs[name][i]["text"]))
                                           for i in closed_ids),
            "open_answer_token_recall": statistics.mean(opened.values()),
            "content_aware_mixed_diagnostic": statistics.mean(content.values()),
            "content_aware_closed_accuracy_diagnostic": statistics.mean(content[i] for i in closed_ids),
            "empty_answers": sum(not outputs[name][i]["text"].strip() for i in ids),
            "sum_engine_seconds": sum(outputs[name][i]["seconds"] for i in ids),
            "mean_engine_seconds": statistics.mean(outputs[name][i]["seconds"] for i in ids),
        }
        per_case[name] = {"strict": strict, "content": content}

    paired = {}
    for metric in ("strict", "content"):
        paired[metric] = {}
        for name in names[1:]:
            deltas = [per_case[name][metric][i] - per_case["generalist"][metric][i] for i in ids]
            paired[metric][name] = {"mean_delta": statistics.mean(deltas),
                                    "improved": sum(v > 0 for v in deltas),
                                    "harmed": sum(v < 0 for v in deltas),
                                    "unchanged": sum(v == 0 for v in deltas),
                                    "exact_text_same": sum(outputs[name][i]["text"] ==
                                                           outputs["generalist"][i]["text"] for i in ids)}

    intervention_diagnostics = {}
    for name in names[1:]:
        changed_ids = [i for i in ids if outputs[name][i]["text"] != outputs["generalist"][i]["text"]]
        intervention_diagnostics[name] = {
            "changed_text_coverage": len(changed_ids) / len(ids),
            "changed_cases": len(changed_ids),
            "mean_score_shortfall_on_changed": statistics.mean(
                1-per_case[name]["strict"][i] for i in changed_ids) if changed_ids else None,
            "baseline_shortfall_on_same_cases": statistics.mean(
                1-per_case["generalist"]["strict"][i] for i in changed_ids) if changed_ids else None,
            "harm_fraction_on_changed": statistics.mean(
                per_case[name]["strict"][i] < per_case["generalist"]["strict"][i]
                for i in changed_ids) if changed_ids else None,
            "clinical_risk_estimated": False}

    transport = {}
    channel_cases = {name: set() for name in ("cxr_findings", "cxr_anatomy", "biomed_anatomy")}
    for name in names:
        calls, adopted, presented, gate_reasons = Counter(), Counter(), Counter(), Counter()
        runtime_errors, spatial_items, spatial_structures = [], 0, 0
        for sample_id in ids:
            seen = set()
            for event in outputs[name][sample_id]["trace"]:
                if event.get("event") == "tool":
                    expert = event["expert"]
                    calls[expert] += 1
                    adopted[expert] += int(bool(event.get("adopted")))
                    # Channel effects describe cases where that channel actually
                    # entered generation, not applicability-rejected tool calls.
                    if name == all_name and event.get("adopted"):
                        channel_cases.setdefault(expert, set()).add(sample_id)
                    if event.get("reason", "").startswith("runtime_error:"):
                        runtime_errors.append({"id": sample_id, "expert": expert,
                                               "reason": event["reason"]})
                    if expert == "cxr_anatomy":
                        for item in event.get("native_evidence", []):
                            structures = item.get("payload", {}).get("structures", [])
                            spatial_items += 1
                            spatial_structures += len(structures)
                    if event.get("vector_gate"):
                        gate_reasons[event["vector_gate"]["reason"]] += 1
                if event.get("event") == "decode":
                    seen.update((item["expert_id"], item["evidence_id"])
                                for item in event.get("evidence_transport", {}).get("presented", []))
            for expert in {expert for expert, _ in seen}:
                presented[expert] += 1
        transport[name] = {"expert_calls": sum(calls.values()), "calls_by_expert": dict(calls),
                           "adopted_by_expert": dict(adopted),
                           "presented_cases_by_expert": dict(presented),
                           "cases_with_any_presented": sum(outputs[name][i].get(
                               "presented_evidence_count", 0) > 0 for i in ids),
                           "spatial_evidence_items_executed": spatial_items,
                           "spatial_structures_executed": spatial_structures,
                           "runtime_errors": runtime_errors, "gate_reasons": dict(gate_reasons)}

    channel_effects = {}
    for expert, selected in channel_cases.items():
        selected = sorted(selected)
        deltas = [per_case[all_name]["content"][i] -
                  per_case["generalist"]["content"][i] for i in selected]
        channel_effects[expert] = {"n": len(selected),
                                   "text_changed": sum(outputs[all_name][i]["text"] !=
                                                       outputs["generalist"][i]["text"] for i in selected),
                                   "mean_content_delta": statistics.mean(deltas) if deltas else None,
                                   "improved": sum(v > 0 for v in deltas),
                                   "harmed": sum(v < 0 for v in deltas)}

    gate_events = [event["vector_gate"] for sample_id in ids
                   for event in outputs[gate_name][sample_id]["trace"]
                   if event.get("event") == "tool" and event.get("vector_gate")]
    finite_gains = [event["gain"] for event in gate_events if "gain" in event]
    entry_events = [entry for event in gate_events for entry in event.get("entries", [])]
    entry_audit = {
        "calls": len(entry_events), "accepted": sum(e["accepted"] for e in entry_events),
        "reasons": dict(Counter(e["reason"] for e in entry_events)),
        "accepted_without_visual_verification": sum(e["accepted"] and e.get("visual_status") == "unknown"
                                                    for e in entry_events),
        "local_removal_gains": [e["local_removal_gain"] for e in entry_events if "local_removal_gain" in e]}

    revision_events = [outputs[gate_name][i]["answer_arbitration"] for i in ids
                       if "answer_arbitration" in outputs[gate_name][i]]
    no_tool = [i for i in ids if outputs[all_name][i]["expert_calls"] == 0]
    changed = [i for i in ids if outputs[all_name][i]["text"] !=
               outputs["generalist"][i]["text"]]
    payload = {
        "protocol_identity": json.loads((args.run / "protocol.json").read_text())["identity"],
        "complete": True, "n": len(ids), "scores": evaluated, "paired_vs_generalist": paired,
        "transport": transport, "evidence_by_expert_content_diagnostic": channel_effects,
        "native_entry_gate": entry_audit,
        "answer_arbitration": {"decisions": dict(Counter(e["decision"] for e in revision_events)),
            "reasons": dict(Counter(e["reason"] for e in revision_events)),
            "score_calls": sum(e["score_calls"] for e in revision_events),
            "seconds": sum(e["seconds"] for e in revision_events),
            "margins": [e["margins"] for e in revision_events if "margins" in e]},
        "intervention_diagnostics": intervention_diagnostics,
        "gate": {"calls": len(gate_events), "accepted": sum(e["accepted"] for e in gate_events),
                 "acceptance_rate": (statistics.mean(e["accepted"] for e in gate_events) if gate_events else 0.0),
                 "reasons": dict(Counter(e["reason"] for e in gate_events)),
                 "calls_reaching_visual_contrast": len(finite_gains),
                 "finite_gains": finite_gains,
                 "candidate_generated_tokens": sum(e["candidate_generated_tokens"] for e in gate_events),
                 "verifier_queries": sum(e["verifier_queries"] for e in gate_events),
                 "sum_seconds": sum(e["seconds"] for e in gate_events)},
        "no_tool_parity": {"no_tool_cases": len(no_tool),
                           "all_arms_exact_text_parity": sum(all(outputs[name][i]["text"] ==
                                                                  outputs["generalist"][i]["text"]
                                                                  for name in names) for i in no_tool)},
        "changed_examples": [{"id": i, "reference": references[i],
                              "generalist": outputs["generalist"][i]["text"],
                              all_name: outputs[all_name][i]["text"],
                              gate_name: outputs[gate_name][i]["text"]} for i in changed],
        "fixed_evaluator": {"protocol": PROTOCOL_VERSION,
                            "source_sha256": file_sha256(args.anchor_root /
                                "anchor/corrected_sgta/evaluate_medheval_answers.py")},
        "inputs": {"manifest_sha256": file_sha256(args.manifest),
                   "references_sha256": file_sha256(args.references)},
        "limitations": [
            "Zero gate acceptance is a null intervention and cannot be counted as success.",
            "Lexical and target-blind content-aware metrics are not clinician adjudication.",
            "Per-expert subsets overlap and differ in difficulty; channel deltas are descriptive.",
            "Sequential shared-cache engine seconds are not independent cold-start latency.",
        ],
    }
    if args.source_audit:
        payload["source_audit"] = json.loads(args.source_audit.read_text())
    if args.training_report:
        payload["training"] = json.loads(args.training_report.read_text())
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
