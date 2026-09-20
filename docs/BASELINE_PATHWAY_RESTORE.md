# Baseline-pathway restoration: implementation pilot

## Status and scope

Implemented on top of `608d239bfeafdd55b5e6c4f676d92311563b50e3`
(`experiments/observation-blind-train-v1`). This is an **opt-in experimental
inference path**, not a replacement for main, and not a validated medical gate.
The existing router, specialist pool, packet compiler, prompts, baseline outputs,
and scoring protocol are unchanged. No learned projection, new judge, threshold
fit, domain-invariant projection, or correctness estimator is added.

Local validation used CPU PyTorch and synthetic random-weight attention models.
No Huatuo medical checkpoint, patient image, GPU evaluation, net accuracy gain,
clinical non-inferiority, or SOTA result was obtained in this implementation
session. See `reports/pathway_restore_cpu_validation.json` for the exact scope.

## Hypothesis and operation

A specialist may contribute useful information while also changing the receiving
VLM's original visual computation. Separate these effects by restoring a
specified visual-attention contribution, not by selecting between final answers.

For the last query of selected layer/head:

```
visual = sum(attention[visual positions] * native_value[visual positions])
visual = visual_mass * normalized_visual_direction

restore_vector: output += W_O (visual_reference - visual_receiver)
restore_mass:   output += W_O ((mass_reference / mass_receiver - 1)
                              * visual_receiver)
```

The second mode keeps the receiver's within-image direction. A zero receiver
mass with positive reference mass is undefined and raises an error, rather than
inventing a direction using an epsilon. These are activation interventions;
they do not claim that restored attention weights sum to one. `W_O` is applied
to the difference **without its bias**, which is already present in native
output. Value-projection bias is retained. GQA/MQA repeat native KV heads in
query-head order.

The default selected layer is **the final decoder layer only** (`--layers -1`),
fixed in advance, not selected using labeled validation/test outcomes. Explicit
other layers can be studied in a separate frozen experiment. At prefill only the
last prompt query is modified; expert/image prompt token states are not patched.
At subsequent steps the newly consumed committed token's last query is modified.
The mechanism does not repair every attention path in every layer.

## Branch and cache semantics

`paired_generate` runs one frozen model sequentially with two private KV caches:

1. No-expert reference consumes the currently committed receiver prefix.
2. Expert-conditioned receiver consumes **the identical token IDs**, with its own
   prompt length, positions and KV cache; selected visual contributions are
   restored using the same-step reference.
3. Receiver logits select one token; only that token is committed to both streams.

After a divergence this reference is **not** the separately generated baseline
trajectory. Stored historical baseline answers remain separate evaluation
controls. Cached baseline answers cannot supply the shadow's new-prefix
activations; the additional reference forwards are necessary, not redundant
baseline answer regeneration.

Modes: `off` installs no attention hooks; `audit` reads but does not modify
activations; `restore_mass` and `restore_vector` are separate experimental arms.
Identical prepared inputs take the off path with no shadow forward. All hooks
are removed on normal exit and exceptions. Model weights stay frozen.

## Native integration and restrictions

- `huatuo_pathway.prepare_native` calls the existing `HuatuoOEAdapter._inputs`
  serializer and Huatuo's `prepare_inputs_labels_for_multimodal_new`.
- Unique integer **position markers**, not medical labels, identify the actual
  expanded image span and detect missing/reordered prompt tokens. They never
  enter the language-model loss. No fixed `576` visual-token assumption.
- Single original image, batch one, no padding, contiguous position IDs, ordinary
  separate linear Q/K/V/O projections, full non-evicting KV cache, deterministic
  greedy decoding only. Expanded image embeddings must match between branches.
- Selected native attention must explicitly support `output_attentions` and
  return usable weights. FlashAttention, incompatible newer interfaces,
  projection tensor parallelism, quantized V/O modules, and sliding-cache
  eviction are rejected, not silently replaced or counted as baseline fallbacks.
- SDPA implementations may use their native eager fallback when weights are
  requested. The canary requires native/manual/historical token parity and
  audit/no-hook parity. A failed parity is a technical stop, not an algorithmic
  rejection. Kernel-level floating-point differences can still be a limitation
  outside the canary; parity on two cases is not a universal equivalence proof.
- Only selected layers request weights, and only visual projected values are
  retained. Native prefill attention may still allocate quadratic matrices;
  there is no claim that peak memory or latency has been reduced.

## Existing-run replay

`scripts/run_huatuo_pathway.py` reuses a completed native Huatuo TRAIN run with
`protocol.json`, `complete.json`, and `cases/<id>.json`, as produced by the existing
`run_huatuo_admission_probe.py`. It reconstructs the SAME evidence prompt via the
existing `NativeSession.context`, not a new evidence compiler. Specialist outputs
and historical baseline/compact answers are reused. All source rows are included
in `run`; only `canary` uses the fixed leading cases, without outcome selection.

Required source evidence: complete identity/count, label-free rows, unchanged
image and case hashes, recorded prior test-image exclusion audit, and unchanged
pinned source/backend files when supplied. The source's train/test image exclusion
is inherited from that audit, **not re-proved** by this runner. `answer_type`, when
present as metadata, is not used; `benchmark_prompt` is kept verbatim.

Before the canary, freeze an explicit JSON containing the exact native baseline
generation kwargs. The five required keys are `do_sample` (false), `num_beams`
(1), `max_new_tokens` (identical to source), `min_new_tokens`, and
`repetition_penalty`. Optional keys: `eos_token_id`, `pad_token_id`, `use_cache`.
Read these from the actual pinned native backend. Do not assume that the budget
is 64: the recent Huatuo source protocol uses 1024. Do not copy the upstream
interactive chatbot's sampled defaults. Unsupported generation processors stop
execution rather than being silently removed.

Use the existing compatible server Python, ANCHOR checkout and local weights:

```bash
# Run in an isolated worktree of the new branch. Supply actual existing paths.
PYTHON=/home/dbw/.runtime/miniconda3/envs/huatuo/bin/python
SOURCE=/home/dbw/merit-feddg-huatuo-critic/runs/native-slake128-confirm-v2
GENERATION=/absolute/path/to/verified-native-greedy-kwargs.json
OUTPUT=runs/pathway-restore-train-v1

$PYTHON scripts/run_huatuo_pathway.py \
  --source-run "$SOURCE" --generation-json "$GENERATION" --output "$OUTPUT" \
  --import-root /home/dbw/ANCHOR --stage check

# Set CUDA_VISIBLE_DEVICES to an authorized idle physical device using the
# server's existing device mapping. This program never selects/kills jobs.
$PYTHON scripts/run_huatuo_pathway.py \
  --source-run "$SOURCE" --generation-json "$GENERATION" --output "$OUTPUT" \
  --import-root /home/dbw/ANCHOR --stage canary

# Same frozen inputs/settings; canary case outputs are resumed, not overwritten.
$PYTHON scripts/run_huatuo_pathway.py \
  --source-run "$SOURCE" --generation-json "$GENERATION" --output "$OUTPUT" \
  --import-root /home/dbw/ANCHOR --stage run
```

The first two cases require exact historical/native/custom-greedy token replay
for both baseline and compact, plus audit parity. At least one must exercise an
expert-exposed attention path; all-identical inputs do not pass that canary.
The adapter factory is configurable, defaulting to the existing
`anchor.corrected_sgta.models_oe:HuatuoOEAdapter`. Imports do not install packages.
Weights are offline-only. No remote GPU job was launched from this implementation
session. Start any long run through the user's established server scheduler.

Outputs are exclusive, atomic, locked and resumable. Identity includes source
and code hashes, generation settings, modes/layers, adapter source, runtime
versions and model configuration. Checkpoint file stats are explicitly recorded
as **not full weight-content hashes**. No scores, reference answers, patient
images, weights or credentials are submitted with this code.

## Acceptance criteria for the next real experiment

Use the existing frozen evaluator OFFLINE; do not change CE/OE parsing or prompt
contracts. Report both recovered baseline-correct cases and retained
baseline-wrong/expert-correct cases, plus new harm, no-op coverage, generation
failures, token budget hits, reference/receiver forward counts, latency and peak
memory. A zero-intervention/all-baseline result is not a successful gate.

Compare mass restoration with vector restoration and matched audit/off controls.
For a mechanism claim, add a predeclared equal-budget nonvisual/random-path
control before attributing improvements specifically to the visual pathway.
This control is not implemented in the current pilot. Correct experts may need
to redirect visual processing: restoration can suppress that benefit. Stable
misleading context can survive restoration. There is no non-inferiority theorem.

## Source/code provenance

Existing replay/serialization contracts inspected at base commit `608d239...`:
`merit_feddg/capability_runtime.py` (`NativeSession.context`),
`merit_feddg/evidence_transport.py` (`pack_records`),
`scripts/run_huatuo_admission_probe.py`, and `scripts/observation_blind_probe.py`.

Native Huatuo upstream interfaces inspected:
- https://github.com/FreedomIntelligence/HuatuoGPT-Vision/blob/main/llava/model/language_model/llava_qwen2.py
  (`LlavaQwen2ForCausalLM.forward/generate`; inspected blob
  `cc190dd7f27316b93718fd695a00b9aad7f6b82d`).
- https://github.com/FreedomIntelligence/HuatuoGPT-Vision/blob/main/cli.py
  (native image/ conversation serialization; inspected blob
  `ea168c95b30a45d4ca4a85bc194556e1b2f0bd97`).

Those are implementation references, not evidence that the new intervention
improves medical accuracy. This patch implements the discussion's pathway
hypothesis; it does not reproduce DeCoRe/PAI or claim a new theorem.
