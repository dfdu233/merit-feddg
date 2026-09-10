# Full VQA-RAD training-free spatial result bundle

This directory is the Git-tracked analysis bundle for the complete 451-question
official VQA-RAD test run. The protocol identity is
`f42e591e4c92265c1c3d1827ac0378167285a353b143079d2657e8714d794b4b`.
All four arms were generated on every row: `generalist`, `spatial_equal`,
`spatial_weighted`, and `spatial_gate`. No target partition was made and target
references were not loaded during generation.

## Result

The result does **not** support scaling. Strict mixed score is 0.2945 for the
generalist, 0.2885 for both ungated spatial arms, and 0.2923 for the spatial
gate. The gate reduced but did not remove harm, accepted 12/403 usable-evidence
events, and increased mean recorded engine time from 0.528 s to 1.661 s per
case. The apparent content-diagnostic improvement contains a medically false
pregnancy/X-ray statement and is not counted as clinical gain.

## Files

- `evaluation-summary.json`: aggregate fixed-reference scores, paired deltas,
  channel diagnostics, transport counts, gate totals, examples, and limitations.
- `per-case-results.jsonl`: 451 records in official order. Each row contains the
  question, references, answer type, image hash/repository-relative image path,
  image-only routing output, and the complete compacted output of all four arms.
  Arm outputs retain answers, token IDs, generation settings, evidence objects,
  controller/tool/decode traces, expert adoption, tensor-packet metadata, gate
  decisions/gains/query counts, and recorded timings.
- `spatial-audit.json`: proof that real mask structures were loaded and which
  experts were included or excluded from the spatial protocol.
- `routing.json`: all 451 image-only modality decisions and routing timings.
- `protocol.json`: frozen experiment and arm configuration, with machine paths
  normalized to repository-relative paths or `$ANCHOR/...`.
- `SHA256SUMS`: hashes of the machine-readable result files.

## Deliberate payload omission

The raw run directory is
`runs/matched-spatial/f42e591e4c92265c1c3d1827ac0378167285a353b143079d2657e8714d794b4b`
and is approximately 18 GB. Its soft-mask arrays are embedded as base64 strings,
so they are not suitable for GitHub. In `per-case-results.jsonl`, every omitted
`data` string is replaced in place by `data_omitted_from_git: true` and its
`data_character_count`. Shape, encoding, mask statistics, labels, boxes,
selected groups, provenance, confidence, adoption, and gate metadata remain.
Model weights, images, expert caches, and background logs are also not copied.

This compaction changes only storage. The tracked aggregate JSON files are
copied from the completed run, and the per-case bundle is derived from its four
final arm files plus the frozen manifest, references, and routing output.

## Minimal analysis example

```python
import json

with open("docs/results/vqarad_spatial_full_2026-09-10/per-case-results.jsonl") as f:
    rows = [json.loads(line) for line in f]

changed = [
    row for row in rows
    if row["arms"]["spatial_gate"]["text"]
    != row["arms"]["generalist"]["text"]
]
print(len(rows), len(changed))  # 451, 12
```
