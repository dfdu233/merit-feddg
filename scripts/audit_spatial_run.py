#!/usr/bin/env python3
"""Label-free transport/coverage/cost audit; changed text is not medical gain."""

import argparse
import json
from collections import Counter
from pathlib import Path


def audit_run(root):
    root = Path(root)
    protocol = json.loads((root / "protocol.json").read_text())
    names = protocol["methods"]
    outputs = {name: json.loads((root / f"{name}.json").read_text()) for name in names}
    baseline = outputs["generalist"]
    if not baseline or any(set(rows) != set(baseline) for rows in outputs.values()):
        raise ValueError("all arms must cover the identical complete manifest")
    result = {"n": len(baseline), "identity": protocol["identity"],
              "excluded_experts": protocol.get("excluded", {}), "arms": {},
              "clinical_gain_measured": False, "labels_loaded": False}
    for name, rows in outputs.items():
        calls, adopted, presented, reasons, modalities = (Counter() for _ in range(5))
        nonzero, gate_calls, accepted, gate_seconds, verifier_forwards = 0, 0, 0, 0., 0
        for row in rows.values():
            modalities[row.get("input_modality", "unknown")] += 1
            seen, changed = set(), False
            for event in row.get("trace", []):
                if event.get("event") == "tool":
                    calls[event["expert"]] += 1
                    adopted[event["expert"]] += bool(event.get("adopted"))
                    reasons[event.get("reason", "unknown")] += 1
                    gate = event.get("vector_gate")
                    if gate:
                        gate_calls += 1
                        accepted += bool(gate.get("accepted"))
                        gate_seconds += gate.get("seconds", 0.)
                        verifier_forwards += gate.get("verifier_queries", 0)
                if event.get("event") == "decode":
                    transport = event.get("evidence_transport", {})
                    seen.update((v["expert_id"], v["evidence_id"]) for v in transport.get("presented", []))
                    changed |= transport.get("spatial_fusion", {}).get("max_abs_token_delta", 0) > 0
            presented.update(expert for expert, _ in seen)
            nonzero += changed
        result["arms"][name] = {
            "calls": dict(calls), "adopted": dict(adopted), "presented": dict(presented),
            "reasons": dict(reasons), "modality_rows": dict(modalities),
            "cases_with_nonzero_token_intervention": nonzero,
            "text_changed_vs_generalist": sum(row["text"] != baseline[i]["text"] for i, row in rows.items()),
            "gate_calls": gate_calls, "gate_accepted": accepted,
            "gate_seconds": gate_seconds, "verifier_forwards": verifier_forwards,
            "mean_engine_seconds": sum(row.get("seconds", 0.) for row in rows.values()) / len(rows),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    result = audit_run(args.run)
    path = args.run / "spatial-audit.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    main()
