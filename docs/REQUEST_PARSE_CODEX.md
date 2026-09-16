# Codex: next request-interpretation audit (no clinical generation)

Read `skills/cs197-research/SKILL.md`, then `docs/REQUEST_PARSE_VECTOR.md`.
The dated de056a8 snapshot supersedes the skill's old 12-case project-status tail;
do not restart candidate-support work in this Vector.

Use a clean independent worktree of this branch. Preserve all old runs and dirty
worktrees. Reuse the existing authorized environment; do not upgrade packages,
download weights or use a second GPU. No training and no test-label access.

1. Run focused tests and the full CPU regression. Distinguish preexisting failures.
   Verify text-only LLaVA generation on one prompt through the official
   `images=None` path. Unexpected return layout or missing GPU UUID support must
   stop and be reported, not silently select another device or use blank images.
2. Prepare 12 real CXR TRAIN questions with the fixed selection. Review the blank
   worksheet independently before reading parser outputs. Do not fill the gold
   worksheet using the same parser, reference answers or disease predictions.
3. Execute the three actual parse calls per question: one shared question-only
   parse and one catalog-conditioned parse for each of the two XRV experts.
   The old regex decision is computed without modification. Each method gets the
   same frozen instructions and budget, apart from catalog exposure.
4. Report syntax validity, unknowns, every missing/extra entity-attribute/relation
   and cost, including truncated/failed calls. No retry loops, score thresholds,
   new synonyms, hand-coded disease blacklists, new experts or altered decoder.
5. Optional actual cached packets may be supplied only to the offline evaluator.
   Bind to exact question and source image hash. Do not substitute synthetic masks
   or interpret missing packets as successful rejection. Report actual-payload
   coverage separately and keep full payload/answers on the server.
6. On successful execution of the frozen interface (not merely a high score), a
   separate 100-case diagnostic may be prepared with unchanged prompt/algorithm.
   Do not combine identities or advertise it as a new held-out test split. If you
   change the method, report a new development iteration instead.
7. Commit only source changes needed for verified runtime compatibility plus a
   deidentified report, test logs/summary and frozen protocol hashes. No patient
   questions/answers/images/masks, gold worksheet or weights. Verify remote SHA.

Example commands (from the independent worktree; use the established environment):

```bash
OPENBLAS_NUM_THREADS=1 bash scripts/with_gate_kb_env.sh -m pytest -o addopts='' -q tests/test_request_parse_study.py tests/test_request_parse_metadata.py
OPENBLAS_NUM_THREADS=1 bash scripts/with_gate_kb_env.sh -m pytest -o addopts='' -q

PYTHONPATH=. bash scripts/with_gate_kb_env.sh scripts/run_request_parse_study.py prepare \
  --manifest /home/dbw/merit-feddg/runs/vqarad-official-protocol-v3/data/train/manifest.jsonl \
  --incumbent-json /home/dbw/merit-feddg/runs/vqarad-official-train-v3/transport-fixed/0379506d53573b3a4ba8eb1c1e3648604b22fdf781949f780b6b1091d3f8dbd6/compact_rows.json \
  --output runs/request-parse-pilot-v1 --limit 12

# No model loaded by this preflight:
PYTHONPATH=. bash scripts/with_gate_kb_env.sh scripts/run_request_parse_study.py run \
  --output runs/request-parse-pilot-v1

# Only after confirming authorized device mapping and independent semantic review:
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. bash scripts/with_gate_kb_env.sh scripts/run_request_parse_study.py run \
  --output runs/request-parse-pilot-v1 --execute \
  --gpu-uuid GPU-3846413a-4238-d307-b1f3-10c2dfbe002c

PYTHONPATH=. bash scripts/with_gate_kb_env.sh scripts/evaluate_request_parse_study.py \
  --run runs/request-parse-pilot-v1 --gold /private/reviewed-requests.jsonl \
  --output runs/request-parse-pilot-v1/semantic-report.json
```

The explicit UUID above is the one in the prior authorized canary, not permission
to use any other device if it cannot be found. Gold and packet paths are local
inputs; this repository does not pretend those annotations already exist.

Final result table: question count; unique parser calls; schema-valid fraction;
parsed/unknown counts; exact semantic agreement; atom/relation precision/recall;
missing unsupported entities; optional unexpected/missed delivery vs the reviewed
request under the same policy; wall time/tokens; and one new belief update.
No medical-accuracy, ICLR novelty, calibration or semantic-safety claim yet.
