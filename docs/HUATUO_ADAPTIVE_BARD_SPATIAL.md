# HuatuoGPT-Vision native spatial backend for Adaptive BARD

## Scope

The formal MERIT spatial receiver path was originally implemented only for
`backend: llava_med`.  The factory explicitly rejected
`training_free_spatial` for every other backend.

This branch adds the same **parameter-free native spatial path** to the legacy
HuatuoGPT-Vision-7B receiver used in the existing Huatuo experiments.

It targets the released **LLaVA-Qwen2** checkpoint/runtime:

    model_type = llava_qwen2
    llava/model/language_model/llava_qwen2.py

It intentionally does **not** silently adapt the newer upstream Qwen2.5-VL
rewrite.  A checkpoint with another `model_type` fails closed.

## Same algorithm, different receiver backend

The spatial algorithm is not reimplemented for Huatuo.

For both LLaVA-Med and Huatuo:

    heterogeneous segmentation / CAM geometry
        -> spatial_packet(...)
        -> SpatialEvidenceBridge (zero parameters)
        -> hook on native mm_projector output
        -> receiver next-token distribution
        -> BARD receiver residual

The only receiver-specific part is image/prompt serialization and the frozen VLM
forward path.

Huatuo uses its released prompt contract:

    <|user|>
    <image>
    {question/evidence prompt}
    <|assistant|>

and deterministic square padding with the checkpoint image processor.

## Why the same spatial operator is valid structurally

The legacy Huatuo model is also LLaVA-style:

- a CLIP vision tower produces one square patch grid;
- `model.get_model().mm_projector` maps patch features into the Qwen2 hidden
  dimension;
- `prepare_inputs_labels_for_multimodal_new` inserts those projected visual
  tokens into the language-model input.

MERIT's training-free operator acts at the mm-projector output, so it does not
depend on Mistral versus Qwen2 language layers.

Runtime validation still requires:

- deterministic square padding;
- patch-only vision-tower features;
- a square `num_patches` grid;
- matched crop/resize geometry;
- flat patch merging;
- `mm_projector` existence;
- frozen model weights.

If any condition fails, spatial evidence is rejected rather than approximated.

## Configuration

Use:

    configs/matched_bard_huatuo.yaml

Set the actual existing server assets:

    export HUATUO_VISION_CHECKPOINT=/absolute/path/to/HuatuoGPT-Vision-7B
    export HUATUO_VISION_SOURCE=/absolute/path/to/legacy/HuatuoGPT-Vision/source

The source directory must contain:

    llava/model/language_model/llava_qwen2.py
    llava/model/llava_arch.py
    llava/constants.py

Use the existing Huatuo Python environment.  Do not upgrade its Transformers or
Torch merely for this branch.

## First technical canary

Before running any benchmark:

    PY=/home/dbw/.runtime/miniconda3/envs/huatuo/bin/python

    "$PY" scripts/check_huatuo_spatial_backend.py       --config configs/matched_bard_huatuo.yaml       --image /absolute/path/to/one/source-image.png       --output runs/huatuo-spatial-canary.json

This canary uses a synthetic full-image geometry packet only to verify mechanism
plumbing.  It must report:

- one spatial record;
- an exercised `mm_projector` hook;
- nonzero finite visual-token delta;
- nonzero finite receiver-score delta;
- frozen model weights;
- no references read.

It is **not** a medical efficacy test.

## Adaptive BARD source/development test

After the technical canary passes, use the same source/development manifest used
for the corresponding LLaVA comparison:

    "$PY" -m merit_feddg.matched_evaluation       --protocol bard       --config configs/matched_bard_huatuo.yaml       --manifest /absolute/path/to/source-manifest.jsonl       --output runs/adaptive-bard-huatuo       --reuse-expert-run /absolute/path/to/compatible-expert-cache

The expert cache can be shared across receivers only when the existing strict
manifest/route/expert provenance checks pass.  Generalist outputs cannot be
shared between LLaVA-Med and Huatuo.

Report at minimum:

1. Generalist score;
2. joint-all;
3. isolated mean;
4. isolated geometric median;
5. Adaptive BARD;
6. n=1 / n=2 / n>=3 commit coverage;
7. semantic-only versus semantic+native-spatial branch counts;
8. original harms/rescues;
9. synthetic one-node fault robustness;
10. wall time and receiver forward cost.

## Efficiency boundary

The Huatuo backend currently uses the correctness-first production replay
`next_scores(prefix)` path.

The LLaVA-specific vision-cache / persistent-KV acceleration is **not assumed to
transfer** to Huatuo.  Do not enable a Huatuo fast path until a separate
real-checkpoint parity canary proves token-level equivalence.

Result quality has priority over speed for this cross-receiver test.

## Scientific interpretation

A successful Huatuo result would support **receiver transferability** of the
spatial/BARD mechanism.  It would not prove that the same absolute gain should
hold across backbones.

A failure is also informative:

- if the spatial technical canary fails, the backend adaptation is invalid;
- if the canary passes but BARD fails, the mechanism does not transfer to the
  stronger/different receiver under the frozen protocol;
- if semantic branches work but spatial branches hurt, the spatial operator may
  be receiver-specific despite a structurally compatible projector.
