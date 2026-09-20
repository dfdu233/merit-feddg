# Decision-path algorithm chain — actual server attempt

2026-09-20 UTC. **BLOCKED_TECHNICAL at A before model loading. No scientific A/B/C result exists, and READY_FOR_SCALE was not reached.** This is a resource-preflight failure, not a failed scientific hypothesis or a successful algorithm run.

## Frozen execution

- Repository branch: `experiments/huatuo-decision-path-algorithm-chain-v1`.
- Requested implementation: `50668281a0ceadb70098660568655ce87b9f9cff`, verified against fetched origin; base `0be2c3cdc3a89f6119ac73d163c6436e3070798e` is its ancestor.
- Actual generation revision: `5d51ef2` (full SHA in `provenance.json`). Source frozen before launching; no running source was edited.
- Immutable plan identity: `24b0c1e910978243905e6b750e176cd514b7cc39ca6fcdb549ab2528aea762a3`.
- Plan policy exactly equals the complete checked-in default JSON: prefix 16, seed 0, bootstrap 4000, min delta .005, retention .75, damage .10, mean decode budget 10 seconds/case, at most 2 attempts/node, 200000 total forwards.
- Python 3.10.20, Torch 2.0.1+cu117, CUDA runtime 11.7, Transformers 4.37.2. Existing environment/checkpoints only; no installation or downloads.

Actual path: **freeze → detached controller → A-attempt-1 worker → device-check exception → BLOCKED_TECHNICAL**. The original two-argument wrapper submitted host controller PID 1721217. The worker started and its exit record has return code 1. The controller persisted `A-attempt-1/decision.json` and terminal `state.json` and exited. No retry is queued.

## Resource blocker

Authorized host GPU0 UUID `GPU-809e1541-5fe0-e1a6-d360-d0ea647e9023`, PCI `00000000:01:00.0`, has 49140 MiB memory. It is not visible inside this container, so the chain was submitted through the already authorized host SSH account.

Although sampled utilization was 0%, the driver reports another user's `/usr/bin/nautilus` process as **C+G**, with 34 MiB in the compute-process query. Total device use was 428 MiB including display contexts. The existing `native.check_device` rejects any other compute context, with the exact exception:

> Another compute process owns the authorized GPU; no job is killed

The other authorized GPU, UUID `GPU-3846413a-4238-d307-b1f3-10c2dfbe002c`, PCI `00000000:05:00.0`, was running an unrelated Huatuo model process at about 25 GiB and 98–99% utilization. Container GPU0 corresponds to this host GPU1, freshly verified rather than assumed.

Neither process was stopped. No desktop exception, GPU-occupancy threshold, policy relaxation or opportunistic device switch was added to defeat the frozen guard. One A attempt is recorded, zero forwards were consumed, and one explicit same-plan technical retry remains available after the blocking context is gone.

## Coverage and unavailable results

| Node | Planned / complete | Actual disposition |
|---|---:|---|
| A | 128 / 0 | Blocked before model load and before the first native canary |
| B / C | Not reached | No candidate or matched-control predictions |
| D / E | No certified cohorts declared | `confirmation=[]`, `extension=[]` frozen in advance |

All 128 planned development identities are retained as anonymized `not_attempted_resource_block` records in `per-case-results.jsonl`, with null predictions, scores and diagnostics. These are **not** 128 failed model predictions. No cached scores were substituted for real A activations or logits. There are no directional margins, selected layer, candidate gain, harm repairs, gain retention or medical-correction claims to report.

The nominal relative boundaries would be layers 8 and 17 for the pinned 28-layer model, but neither intervention was executed. No official TEST inference or evaluation was run. CPU tests and a successfully created output directory do not constitute algorithm success.

## Source data and independent-confirmation audit

The complete existing native SLAKE128 source run was verified using its protocol, completion identity, case content hashes, original image bytes and decoded pixels. Cached generalist/compact tokens are present. This is the original 128-case queue, not the earlier 65-case mass-complete subset. Historical prompt reconstruction/native token parity were **not executed in this attempt**, because device preflight stopped before loading.

A real local audit read 316 experiment metadata files across existing worktrees and ANCHOR runs, recorded file hashes and decoded image dimensions/content hashes, and produced raw-RGB SHA256 arrays. The conservative union contains 2167 distinct referenced images. Official SLAKE and VQA-RAD TEST membership was separately enumerated as 383 distinct pixel images, exclusively for exclusion checks.

The registry conservatively includes images declared in historical experiment manifests, even when exact observation history is not recoverable; bare dataset-storage train/test manifests were excluded from the exposure union. Under that rule, none of the 586 SLAKE TRAIN or 313 VQA-RAD TRAIN images are outside both the known registry and official TEST set. This does **not** assert that every such TRAIN image was actually generated or inspected. No already exposed cache was relabeled as independent confirmation.

Six unresolved historical image references come from a separate study-view dataset; they remain explicitly unresolved in local audit records. Native SLAKE/VQA queues provide no usable patient/study mapping to certify patient independence. Filename, language and encoding differences were not treated as independent patients. The registry cannot establish completeness of external historical exposures. Given these limitations, no independent compatible D/E cohort was certified before freeze. A/B/C were still submitted; a later transition to D would stop as BLOCKED_DATA without changing this plan.

Original registry sources, full image inventory, unresolved mappings and executable plan remain local under `runs/preparation/` and `runs/chain-v1/`. Public fingerprints and aggregate limitations are supplied; patient/study mappings, raw images, references and expert payloads are not uploaded.

## Compatibility fix and validation

A concrete pre-freeze incompatibility was reproduced: the new storage code compared raw RGB SHA256 to the older native pipeline's `SHA256(str(rgb.size).encode() + rgb.tobytes())`, incorrectly raising “Changed decoded image pixels”. Revision `5d51ef2` validates the old size-prefixed source digest independently while retaining the chain's raw-RGB registry algorithm, dimensions and byte-file hash. A corrupted source digest is still rejected. No source cache was rewritten.

The same revision adds measured per-phase forward/time accounting and actual MERIT import provenance. Residual, coordinate-roll, projection formulas, layer-selection policy, thresholds and scorer remain unchanged. In particular, the existing strict GPU guard is unchanged.

Actual server checks: original chain tests **81 passed**; fixed chain plus pathway regression **136 passed**, 0 failed, 0 skipped. Compilation, CLI help, shell syntax and diff checks passed. The host resolved MERIT to this worktree and the adapter to the intended ANCHOR path. The native model class was pinned but never instantiated, so no runtime model-import parity claim is made.

All local checkpoint weight files, model/tokenizer configuration, adapter/native code and relevant Transformers generation files have real SHA256 pins. Scorer functions and reference files are pinned separately. An unused root-only Hugging Face tree metadata file initially prevented freeze; it was excluded from runtime pins before successful freeze. It is not a weight/configuration/code dependency. Shared cache paths were explicitly configured for the host.

## Costs and artifacts

- Data-audit measurement: 38.9880 seconds.
- Initial runtime-file hashing: 8.7509 seconds; later repeated integrity checks add unaggregated overhead.
- Failed worker wall time: 9.4120 seconds, including imports and integrity checks before device rejection.
- Model loads, model forwards, generated tokens, canary/A/candidate/control calls: **0**.
- Offline scorer was not invoked. Model peak allocated memory is **unavailable**, not the other process's memory and not an inferred zero.
- No new Baseline or expert cache inference. Whole-task preparation wall was not instrumented, and cached Baseline deployment latency is unknown.

See `summary.json`, `node-decisions.json`, `validation.json`, `provenance.json`, `plan.redacted.json`, `data-audit.json`, and `commands.md`. Artifact fingerprints include the real local plan/state/input job/decision/exit and data registries. The public plan is descriptive and intentionally not a replacement executable plan.

The existing main worktree, untracked weekly reports, other experiments and prior results were preserved. No main/PR #13/PR #14 merge was performed. The next legitimate operation is the documented same-plan retry only after the device condition changes; no unattended wait/retry is running.
