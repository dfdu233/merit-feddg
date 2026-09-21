"""Replay frozen historical source prompts without reading answer labels."""

import json
import sys
from pathlib import Path

from merit_feddg.generalist_factory import load_generalist
from merit_feddg.io import load_experiment_yaml

c = load_experiment_yaml(sys.argv[1])
p = load_generalist(c["generalist"])
results = []
for r in json.loads(Path(sys.argv[2]).read_text()):
    o = p.generate_with_usage(r["image"], r["prompt"], r["max_new_tokens"])
    result = {
        "id": r["id"],
        "exact_token_parity": o["token_ids"] == r["token_ids"],
        "expected_tokens": len(r["token_ids"]),
        "observed_tokens": len(o["token_ids"]),
        "observed_token_ids": o["token_ids"],
    }
    results.append(result)
    print(result, flush=True)
Path(sys.argv[3]).write_text(json.dumps(results, indent=2))
assert all(r["exact_token_parity"] for r in results), (
    "Historical parity failed: stop efficacy interpretation"
)
