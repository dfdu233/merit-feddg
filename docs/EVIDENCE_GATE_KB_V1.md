# Evidence purpose gate, text seeds and specialist preparation

## Version and scope

Independent worktree: `/home/dbw/merit-feddg-evidence-gate-kb`; branch
`implementation/evidence-gate-kb-v1`; base
`e6f88744900a11b516279980124dc4775a5ee07b`. The original dirty worktree and all
semantic_all/compact_rows/compact_all configurations/results are untouched.
`medcave-risk-agent` was audited at `6a64fab`; its calibrated risk policy is NOT
merged into this branch. No usable Gate/KB/small-specialist implementation was
found in server source or the inspected remote branches. The visualization ZIP
was not treated as code; it was not available in the inspected attachment paths.

## Implementation

- `evidence_use.py`: restores existing request-contract, modality, task,
  capability and scope checks. Applies to inherited and new observations. An
  optional frozen generalist labels use as ANSWER/AUXILIARY/IRRELEVANT/UNKNOWN,
  not clinical truth. No confidence is manufactured; disagreement alone does
  not trigger rejection. Excluded native artifacts remain in audit, not in the
  final answer prompt. Dependencies cannot be erased to smuggle derived content
  past a rejected parent. Missing contracts remain unknown, not wrong.
- `plug_run --gate-comparison`: incumbent, no_new_gate, scope_restored,
  purpose_gate; fixed acquisition order, same layout/prompt/model/token/action
  budgets. Restored scope may prevent an otherwise inapplicable tool call.
  Purpose judging is additional measured model work, not a zero-cost oracle.
  Original invocation without the flag preserves old arms and policies.
- `text_kb.py`: SQLite FTS5 build/query and the existing native factory interface.
  Six original factual paraphrases with source URLs, declared CC0 text license,
  scope and source family. No patients, reference answers or embeddings.
  Text is returned, not just pointers. Lexical scores are not correctness.
  Queries never turn general knowledge into patient findings. The KB was NOT
  injected into the Gate canary; this keeps the four-arm comparison unconfounded.
- `native_small_medical.py`: strict local checkpoint loading for BreastMNIST
  ResNet18-28 and binary BUSI U-KAN, with source/checkpoint fingerprints and
  original-coordinate mask export. No training, downloads inside inference, or
  permissive partial parameter loading. New model resources are independent of
  the earlier canary identity.
- `evidence_html.py`: static offline HTML with no external resources. Default
  output excludes images, questions, native payloads, paths and answer text;
  source/scope/case identifiers are hashed. Gate destination, actual delivery,
  dependency references, predicted boxes, answer hashes/token counts and costs
  remain visible. `--include-text` is an explicit sensitive LOCAL export only.
  Admission is never displayed as medical correctness.

## Real canary — STOPPED, not a medical success

The original complete 1,793-row VQA-RAD TRAIN manifest and complete compact_rows
base were verified. Exactly the first two rows were scheduled, without answer
selection. No test data, temperatures or thresholds were fitted. References were
opened only for the following offline diagnostics. Both rows are CLOSED.

Run: `runs/gate-kb-canary/ced22041fb76dfea3298a15f243ccabcbf948b830113ff9084655e6cc5d66c65/`.
`shards/0/results.json` has `complete=false`; no full protocol or full score is
claimed. Generalist initialization took 12.498 s.

| Arm | Real candidates | Diagnostic score | Improve/harm vs incumbent | New tool invocations | Answer-model calls | Mean incremental time |
|---|---:|---:|---:|---:|---:|---:|
| incumbent | original answer | 0/2 | — | inherited | 2 | 1.017 s baseline generation |
| no_new_gate | 2/2 | 0/2 | 0/0 | 4 | 2 | 4.067 s |
| scope_restored | 0/2 | 0/2 | 0/0 | 0 | 0 | 0.028 s |
| purpose_gate | 0/2 | 0/2 | 0/0 | 0 | 0 | 0.038 s |

Both no_new_gate answers changed. Both gated arms return the exact incumbent;
zero conditional harmful acceptance is NOT claimed. The diagnostic is the
existing `agent_evaluate.diagnostic_score`, not a clinical adjudication or a
full frozen ANCHOR score. Two zero-score outputs need not be medically equivalent.
Inherited expert cost is kept separately in local outputs and is not recharged
as new tool inference.

All five base registry entries lack request_contract. The first case acquired
no scope-eligible observation; the second retained one historical observation
in audit but rejected answer admission as `missing_request_contract`. Therefore
the purpose model made **zero calls** in this canary. Its CPU routing tests are
not real clinical Gate validation. There were no predicted-region operations in
these two rows. Per the stop condition, the remaining 1,639 stopped-pilot rows
were NOT resumed. No blanket free_query was added to force the gate open.

## Verified environment and commands

Existing Python `/home/dbw/merit-feddg/.venv/bin/python`, Python 3.14.6,
torch 2.14.0+cu130, transformers 4.57.6. Existing OpenCV, timm, Albumentations,
Pillow and SQLite suffice; no shared packages upgraded. The MedMNIST adapter
uses pinned official architecture source and does not require installing the
medmnist package. Both original U-KAN and MedMNIST architecture imports passed.
Container GPU 0 is the authorized physical GPU 1, UUID
`GPU-3846413a-4238-d307-b1f3-10c2dfbe002c`; timed CUDA allocation succeeded.

```bash
cd /home/dbw/merit-feddg-evidence-gate-kb
scripts/with_gate_kb_env.sh -m pytest -ra
scripts/with_gate_kb_env.sh -m ruff check .
scripts/with_gate_kb_env.sh -m merit_feddg.plug_run --help
scripts/with_gate_kb_env.sh -m merit_feddg.text_kb build \
  --seed assets/knowledge/imaging_seed.json --output runs/new-imaging.sqlite
scripts/with_gate_kb_env.sh -m merit_feddg.text_kb query \
  --index runs/new-imaging.sqlite --text 'MRI magnetic fields'
```

The existing index is `runs/gate-kb-checks/imaging.sqlite`; reuse it for queries.
Build and HTML commands refuse overwrite. The wrapper configures the existing
environment and offline inference; it does not install packages.

Actual canary command (do not remove its scheduling stop):

```bash
scripts/with_gate_kb_env.sh -m merit_feddg.plug_run --gate-comparison \
  --base-run /home/dbw/merit-feddg/runs/vqarad-official-train-v3/transport-fixed/0379506d53573b3a4ba8eb1c1e3648604b22fdf781949f780b6b1091d3f8dbd6 \
  --incumbent compact_rows \
  --manifest /home/dbw/merit-feddg/runs/vqarad-official-protocol-v3/data/train/manifest.jsonl \
  --artifacts /home/dbw/merit-feddg/artifacts \
  --expert-registry configs/small_medical_optional.yaml \
  --config configs/evidence_gate_kb_canary.yaml \
  --output runs/gate-kb-canary --canary-cases 2
```

Use `--check-only` instead of `--canary-cases 2` for preflight. After new weights
or code are added, identity changes: this cannot resume or overwrite the old run.
The actual run used tmux session `evidence-gate-kb-canary` and has finished.

```bash
scripts/with_gate_kb_env.sh -m merit_feddg.evidence_html \
  --results runs/gate-kb-canary/ced22041fb76dfea3298a15f243ccabcbf948b830113ff9084655e6cc5d66c65/shards/0/results.json \
  --output runs/new-canary-redacted.html
```

Existing visualization: `runs/gate-kb-checks/canary-redacted.html`.
Committed [redacted HTML](results/gate_kb_canary_2026-09-14/redacted.html) and
[sanitized verification log](results/gate_kb_canary_2026-09-14/verification.txt)
contain no original patient text or images.
Local logs: `runs/gate-kb-checks/{full.log,lint.log,preflight.log,canary.log,kb-build.log,kb-query.log,html.log}`.

## Verification and limits

Old focused tests: 124 passed. New plus old focused suite: initial 1 failure,
136 passed; the new SQL-like query fixture wrongly expected digit `1` not to
match literal text. Corrected to assert a DROP TABLE string is data and the
index remains unchanged. Initial full suite: 1 failed, 858 passed; corrected
full suite: 859 passed in 14.87 s; latest adapter build: 859 passed in 14.19 s.
Inherited 38 lint findings were repaired in
separate commit `7c895c0`, preserving the public mask validation exception and
binding loop closures explicitly. No test assertions were removed for a pass.
Ruff, shell syntax and diff checks passed before GPU execution.

The source contracts still need authoritative, model-specific reconstruction,
not outcome-selected permissiveness. The usefulness labels are uncalibrated
frozen model decisions. Two source-lineage-related specialists are not independent
votes. No ICLR novelty or medical benefit follows from these engineering checks.

See [expert/resource audit and research boundary](EXPERT_LIBRARY_RESEARCH.md).
