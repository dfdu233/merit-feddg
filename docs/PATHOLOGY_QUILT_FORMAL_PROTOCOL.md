# PathVQA formal Quilt experiment (2026-09-17)

## V2: explicit alignment with the requested formal evaluation

User subsequently requested inspecting thread
`01a05d29-9d0b-7121-8a53-488b8cd1a125` and starting the formal evaluation.
The local transcript and benchmark core agree: whole packets that do not fit
are logged and omitted, not forcibly delivered. The 64-token reserve and separate
input/output limits were already correctly applied in our strict v1 adapter.

The separate `pathology-quilt-formal-budgeted-v2` run uses an explicit frozen
`formal-budgeted` transport policy. It DOES NOT claim strict delivery acceptance:
v1 stopped because both fixed cases omitted Quilt with only 58/63 tokens left.
V2 retains that observation, all old evidence, every budget and packet content.
If a new packet does not fit, exact unchanged prompt/evidence hashes are required
before reusing incumbent; rejection reasons and zero new actor calls are recorded.
If it fits, full expert text must be present and a real answer is generated.
Displaced old evidence, unknown rejection, truncation and baseline parity drift
still stop. Coverage is reported separately for matched and wrong-image arms.
An all-reuse result is a transport limitation, never a successful expert method.

Both host GPUs are now explicitly authorized. Full workers use scheduling
`--shard-count 2 --shard-index 0/1`; both controls for one target stay on its lane.
HostGPU1 runs through container CUDA0; hostGPU0 through the existing SSH account.
Quilt and actor remain separate sequential processes on each lane. Actual GPU
UUID stays in provenance, while equivalent model/cache identity excludes only
the physical execution location. Exact disjoint prediction sets, all case IDs,
and completion records are checked by `--stage merge --shard-count 2` before the
unchanged main scorer runs. No new dataset partition or test-driven optimization.

The original strict protocol below and its failed artifacts remain preserved.

User explicitly selected the current formal evaluation protocol over the upstream
64-token TRAIN pilot. This is a separate full TEST protocol, not relabelled TRAIN.
No inference on this test is used to select prompts, thresholds, weights or scope.

## Frozen comparison

- Upstream `implementation/pathology-quilt-v1`, `54be077`; independent branch
  `experiments/pathology-quilt-gpu1-20260917`. Original pilot entry remains intact.
- Full canonical PathVQA manifest: 6719 rows, no new split. Existing formal
  compact_rows baseline/cache identity `a095b918...`, benchmark core `22cb384`.
- Preserve main benchmark prompts, seed42, greedy generation and 1024 output
  cap, original 64-token evidence packing reserve, prior native evidence and
  source image. Do not increase the existing 30000-character evidence budget.
- Use the already-frozen router decisions: 3821 microscopy-routed rows; 2898
  outside scope explicitly reuse incumbent across arms. This router can be
  wrong; dataset name alone does not establish microscopy. No routing retraining.
- Five arms: compact incumbent; remove CONCH; Quilt alone in eligible scope;
  incumbent + matched Quilt; incumbent + same-question/different-image Quilt.
  Outside scope, the quilt-alone arm is a hybrid incumbent reuse, NOT a genuine
  all-image standalone Quilt baseline.
- Deterministic cyclic next-distinct-image donors after hash ordering; retain
  negative-control source identity privately, not in the answering prompt.
- Two fixed hash-selected engineering cases, same full manifest and settings,
  then full continuation. No answers or scores used for canary selection.
- Stop on prompt/token baseline parity drift, omitted/displaced evidence,
  truncated specialist text, empty/capped output, cache mismatch or runtime/OOM.
  Do not select a better-performing configuration in response to test outcomes.

## Runtime and lineage

Only host GPU1/container CUDA0, UUID `GPU-3846413a-4238-d307-b1f3-10c2dfbe002c`.
Quilt and LLaVA-Med run sequentially in different interpreters. Reuse CONCH,
CLIP tower, generalist weights and cached native experts. No shared dependency
upgrade. Full fp16 Quilt chosen before model results because the authorized
card has 48 GB; this is not the upstream 24GB/4-bit diagnostic.

Quilt official source `aldraus/quilt-llava` at
`7e70fc39f792ac55de010eb37bff0a6d6f491c13`, generic model
`wisdomik/Quilt-Llava-v1.5-7b` at
`1bdf5f8b75fb26acc80b08aba7f1979e8e9b12bd`. The checkpoint declares
CLIP-L/14-336, not QuiltNet. Model terms are noncommercial; training overlap
with pathology books/PathVQA is not independently ruled out. No finetuning.

Official model download was missing on server. Hugging Face transfer is retained
with partial files; mirror probe redirected to the same service. Do not download
another duplicate copy to claim speed. Verify both pinned LFS SHA-256 hashes
before model loading. The worker records hashes for checkpoint/source/vision.

Actor adapter explicitly imports the frozen existing formal benchmark core;
the new branch's historical runtime cannot be silently substituted. The new
dependency-light Quilt packet retains the same upstream text/provenance schema.
New prompt hashes, transport records, costs and candidates remain private.

## Checks and artifacts

Initial 37 focused tests and 1006 full CPU tests passed, including both formerly
skipped upstream integration checks. Repeated full suite after formal adapter:
1006 passed in 12.63 s. Compilation and CLI checks pass. Full cached baseline
and image file identities validated; compact caches include JSON and gzip rows.
Real GPU CONCH direct-vs-adapter audit passed on the first fixed formal canary
image. This is an engineering identity test, not a TRAIN efficacy experiment.

Frozen root `runs/pathology-quilt-formal-v1`, identity
`5acebe5400506a0e26cce67d9103af6e744fd3539363070e0162a19a426f17b2`.
No efficacy score exists at preparation time. Existing baseline files unchanged.

Pipeline entry points:

1. `run_pathology_quilt_formal.py --stage prepare --output ROOT` (already done).
2. Quilt environment: `run_quilt_worker.py --formal --canary --precision fp16 ...`.
3. Formal actor environment: `run_pathology_quilt_formal.py --stage actor --canary ...`.
4. Same worker and actor without `--canary`, resume identity-checked caches.
5. `evaluate_pathology_quilt_formal.py --run ROOT` only after 6719 complete IDs.

Final scoring calls the current main ANCHOR mixed-VQA function, with separate
CLOSED accuracy/OPEN recall and image-cluster paired intervals. References are
only opened by the offline evaluator. Frozen ANCHOR source drift stops scoring.
Retain control/new calls, failed attempts, loads, memory and inherited costs
separately; no clinical accuracy/SOTA or new algorithm claim from setup/tests.
