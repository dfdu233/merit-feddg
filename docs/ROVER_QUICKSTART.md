# ROVER: source-only regional evidence pilot

Purpose: test whether a spatially specific, crop-stable visual branch improves
frozen LLaVA-Med answers. This is a new experimental entry point, not a positive
result, clinical certificate, or demonstrated domain generalization method.
No training, target inference, automatic downloads, or dependency upgrades.

## Instructions for the server Codex

1. Preserve local changes. Pull with `git pull --ff-only`; if blocked, inspect the
   diff and stash/commit intentionally, never reset. Use the existing working
   huatuo Python and LLaVA source directory. Do not install research extras.
2. Find the actual v0.12 source JSONL and reference JSON under `runs/` or
   `artifacts/`. Use existing strict label-free source manifests (same schema as
   `open_data.read_manifest`). Do not repurpose target cases. Choose source CXR
   images by image/task compatibility, not by baseline correctness or answers.
   Manually inspect modality only: dataset-level `cxr` metadata is not proof.
3. Prepare predicted regions using the existing XRV anatomical PSPNet checkpoint.
   Inspect `PSPNet.targets` to select an exact individual lung target. Run one
   left or right lung pilot; a union/full-image region may have no usable control.
   Never call anatomy masks lesions. A source image may have abnormality in the
   control region too; the control is NOT assumed healthy.
4. Run `--check-only`, then two cases / 16 tokens. Inspect saved PNG crops and
   production parity. If engineering passes, run four cases / 24 tokens. Reuse
   the same settings and cache. Do not change weights after looking at results.
5. Report every case/arm, region failures, changed tokens, effective weights,
   runtime, and medical assessment. If lexical gains lack medical benefit, say so.
   Never claim DG or ICLR readiness from this pilot. Do not expand to target.
6. Commit code fixes and a concise STATUS.md summary only; do not commit weights,
   raw images, credentials, or large run artifacts. Push only scoped changes.

## Commands (fill the data/checkpoint variables from actual server files)

```bash
cd /home/dbw/merit-feddg
git pull --ff-only
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0
PY=/home/dbw/.runtime/miniconda3/envs/huatuo/bin/python
# If this path does not exist, use the existing /opt/miniconda3/envs/huatuo/bin/python.
SOURCE=/absolute/path/to/existing/source.jsonl
REFS=/absolute/path/to/existing/references.json
XRV=/absolute/path/to/existing/anatomical_pspnet_checkpoint
LLAVA_SRC=/actual/working/llava-med-1.5

"$PY" -c 'from torchxrayvision.baseline_models.chestx_det import PSPNet; print(PSPNet.targets)'
# Set TARGET to the exact individual lung name printed above, not an invented label.
TARGET='REPLACE_WITH_EXACT_TARGET'
"$PY" -m merit_feddg.rover_run --source-manifest "$SOURCE" \
  --regions runs/rover-source/regions.json --prepare-xrv "$XRV" --xrv-target "$TARGET"

"$PY" -m merit_feddg.rover_run --source-manifest "$SOURCE" \
  --regions runs/rover-source/regions.json --check-only --limit 2

"$PY" -m merit_feddg.rover_run --source-manifest "$SOURCE" \
  --references "$REFS" --regions runs/rover-source/regions.json \
  --llava-source "$LLAVA_SRC" --limit 2 --max-tokens 16 \
  --output runs/rover-source/canary
```

The YAML defaults already point to the existing remote LLaVA model and CLIP tower.
Use `--model-path` and `--vision-tower-path` if the server paths differ. This module
runs directly from the repo; no editable reinstall required. `--prepare-xrv` exits
before loading LLaVA, avoiding concurrent specialist/generalist VRAM use.
It processes the supplied source manifest, not just the inference `--limit`.

## Predicted regions from another detector/segmenter

The same inference entry accepts MedSAM/BiomedParse/other predictor results:

```json
{
  "source-case-id": {
    "box": [0.1, 0.2, 0.4, 0.6],
    "producer": "actual-model-and-prompt-protocol",
    "checkpoint_sha256": "64-character-sha256-of-actual-checkpoint",
    "image_sha256": "same-RGB-pixel-hash-as-source-manifest",
    "kind": "predicted_lesion"
  }
}
```

Coordinates are normalized x1,y1,x2,y2 in the original image. For masks, take the
predicted mask bounding box after reversing preprocessing; preserve prompt/model
provenance in the producer record and a separate audit. No target labels, manual
answer-based boxes, or boxes derived from ground-truth masks. The JSON declaration
cannot itself prove provenance: the server operator must audit its origin.
SAM must receive automatically produced prompts in this automatic experiment;
mask production does not independently confirm the requested disease exists.

## Method and comparisons

R is the predicted box, C a same-size corner crop with <=10% overlap, and E the
15%-expanded R. If no valid C exists, the case is excluded and reported. Each
branch uses raw RGB pixels, the identical question/template, and no tool diagnosis.
The original remains its own branch, not pasted into a multi-image mosaic.

At each exact committed prefix calculate distributions b,r,c,e from the SAME VLM.
Let s=JS(r,c), u=JS(r,e), g=max(0,(s-u)/(s+u+epsilon)). JS is Jensen-Shannon
divergence, a symmetric distribution difference measured in nats. Generate from
`(1-weight*g)*b + weight*g*r`. Default weight=0.5 is fixed, not a learned optimum.
This heuristic applies to all generated positions; it is not a medical token
detector. It can still assign high weight to a confidently wrong region.

Arms: base, crop only, control only, fixed fusion, ROVER checked fusion. Fixed
fusion deliberately computes the same four branches to match the full method's
forward-call budget. Region generation and saved crops are shared across arms.
The pilot does not yet include text-expert, overlays, oracle regions, or full
multi-expert scheduling; do not present it as that larger study.

Production greedy token IDs must match base replay. Every branch receives the
method's identical committed prefix at that step. Existing production forced-prefix
replay avoids silent FP16 parity differences, but repeats prefill/vision work:
this pilot is NOT an optimized decoder or a speed claim. Keep the token budget
small. Cache complete case results, not hidden-state tensors. Interrupted cases
are rerun; completed cases are reused under matching code/model/image/settings
fingerprints. References are read only after all generations; rescoring needs no
model load when every selected case is cached.

## Decision after the pilot

- Crops incorrect: repair modality/region proposal; do not tune fusion to hide it.
- Correct region beats base but full method fails: inspect weights and control.
- Correct/control crops behave alike: specificity hypothesis lacks support.
- F1 rises without better finding/location: no clinical improvement established.
- Positive medical signal: add a task-aligned lesion predictor and a second
  modality, then real source-held-out evaluation with frozen settings.

Anatomy-only CXR crops are an immediately runnable diagnostic, not the final
lesion-grounded method. True hospital DG remains a subsequent evaluation.
Related inspiration: [DeepScan, CVPR 2026](https://github.com/YChenL/DeepScan)
for region inspection; [HALC, ICML 2024](https://arxiv.org/abs/2403.00425) for
local decoding. This code is an independent small pilot, not copied upstream
code or a claimed reproduction. Novelty and medical gains remain unproven.
