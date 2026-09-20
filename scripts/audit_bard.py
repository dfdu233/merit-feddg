"""Label-free audit for a completed matched BARD run.

This script never reads references.  It summarizes whether the bounded-fault
mechanism was actually exercised before any task-score interpretation.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def summarize_bard(outputs):
    if not isinstance(outputs, dict) or not outputs:
        raise ValueError("nonempty BARD output mapping required")
    node_counts = Counter()
    structural = departures = evaluated_steps = 0
    probes = retained = fallback = 0
    reasons = Counter()
    for sample_id, output in outputs.items():
        if not isinstance(sample_id, str) or not isinstance(output, dict):
            raise TypeError("invalid BARD output record")
        branches = output.get("expert_branches", [])
        node_counts[len(branches)] += 1
        structural += int(bool(output.get("structural_fallback")))
        sample_departed = False
        for event in output.get("trace", []):
            if not isinstance(event, dict):
                raise TypeError("invalid BARD trace event")
            evaluated_steps += 1
            reasons[str(event.get("reason", "missing"))] += 1
            sample_departed |= bool(event.get("committed"))
            probe = event.get("fault_probe")
            if probe:
                for result in probe.get("single_faults", []):
                    probes += 1
                    retained += int(bool(result.get("same_as_clean")))
                    fallback += int(bool(result.get("fallback_to_base")))
        departures += int(sample_departed)
    n = len(outputs)
    return {
        "schema": "bard-label-free-audit-v1",
        "n": n,
        "independent_expert_nodes": {
            str(key): value for key, value in sorted(node_counts.items())
        },
        "structural_fallback_cases": structural,
        "structural_fallback_rate": structural / n,
        "cases_with_bounded_commit": departures,
        "bounded_commit_case_rate": departures / n,
        "evaluated_token_steps": evaluated_steps,
        "decision_reasons": dict(sorted(reasons.items())),
        "single_fault_probes": probes,
        "single_fault_same_as_clean": retained,
        "single_fault_fallback_to_generalist": fallback,
        "single_fault_clean_retention_rate": (
            retained / probes if probes else None
        ),
        "single_fault_generalist_fallback_rate": (
            fallback / probes if probes else None
        ),
        "references_read": False,
        "medical_correctness_evaluated": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    protocol = json.loads((args.run / "protocol.json").read_text(encoding="utf-8"))
    if protocol.get("experiment_protocol") != "bard":
        raise ValueError("run is not a completed BARD protocol")
    if protocol.get("shards_complete") is not True:
        raise ValueError("BARD run is not fully merged")
    outputs = json.loads((args.run / "bard.json").read_text(encoding="utf-8"))
    report = summarize_bard(outputs)
    text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
