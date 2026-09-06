# v0.8: LLaVA-Med and a small native-tool verification suite

## 1. Run on the existing Linux server

```bash
cd /home/dbw/merit-feddg
git pull --ff-only
bash run_llava_med.sh --install-deps --mirror global
```

For the complete method matrix on host GPU 0, after all assets have already been
prepared, use the offline resumable launcher:

```bash
cd /home/dbw/merit-feddg
bash run_llava_med_full_gpu0.sh
```

It defaults to 16 source cases per group, 16 target cases per dataset and explicitly
enables the local CheXagent. It requires at least 28 GiB free GPU memory before starting.
With 20--28 GiB free, run the same core suite without the optional 3B generator via
`MERIT_CHEXAGENT=off bash run_llava_med_full_gpu0.sh`. Use `MERIT_CHECK_ONLY=1` for
an offline preflight, and set `MERIT_OUTPUT` to choose a persistent resumable run root.
The memory check is a launch guard, not a guarantee against another process growing.

The configured paths are **remote server paths**, not paths available to this development workstation:

- Host Python: `/home/dbw/.runtime/miniconda3/envs/huatuo/bin/python`
- Container-compatible fallback: `/opt/miniconda3/envs/huatuo/bin/python`
- Model: `/home/dbw/ANCHOR/hf_cache/llava-med-v1.5-mistral-7b`
- Official source: `/home/dbw/ANCHOR/data/medheval/code/baselines/Med-LVLMs/llava-med-1.5`
- Vision tower: `/home/dbw/ANCHOR/hf_cache/hub/models--openai--clip-vit-large-patch14-336/snapshots/ce19dc912ca5cd21c8a653c79e251e808ccabcd1`

The similarly named `baselines/Mitigation/llava-med-1.5` tree is patched for
ANCHOR's `corrected_sgta` experiments and is therefore not used as the clean
LLaVA-Med runtime source for this benchmark.

The launcher discovers the host environment first, then the container fallback.
Explicit `MERIT_LLAVA_PYTHON` or `--python` always takes precedence. Conda activation
is optional because the launcher invokes the selected interpreter by absolute path.

No full `bootstrap.sh --profile research-2d` is run. Existing torch, torchvision,
transformers and LLaVA code are preserved. `--install-deps` installs only missing
optional packages using `--no-deps`; it does not certify every existing ABI/version.
If an incompatible installed package is detected, resolve that specific package
instead of upgrading the whole medical-model environment. CONCH installs from the
same pinned official commit used by the previous bootstrap when absent.

The official Mistral v1.5 code must import successfully in that interpreter.
A different legacy LLaVA or HF-converted checkpoint is rejected. The separate
CLIP vision tower must also exist locally/in the HF cache; missing CLIP is not
silently downloaded. Supply a local override if its original identifier is not cached:

```bash
bash run_llava_med.sh --check-only --vision-tower-path /actual/local/clip-directory
```

`--check-only` reads environment/model/weight status without loading 7B weights or
downloading. It exits unsuccessfully for missing required assets. Full loading and
GPU compatibility are only checked by the actual run. `--prepare-only` downloads
small assets and creates the real data/config but does not run medical inference.

`--mirror cn` selects Tsinghua PyPI and `hf-mirror.com` explicitly. Global is the
default because the reported server downloaded from official HF efficiently.
XRV pickle-format weights remain on official GitHub releases by default. Optional
`MERIT_GITHUB_PROXY` is an explicit third-party transport choice, not an official mirror.
No TLS verification is disabled. Keep tokens in the environment, never in commands
submitted to Git or in saved YAML. CONCH access must already be approved by its owner.

Weight preparation resumes partial files and skips verified completed assets.
The XRV release has no published authenticated SHA256 in our registry: expected
bytes plus a **locally computed** SHA detects later corruption, not malicious
upstream replacement. Only load trusted pickle weights. A damaged completed file
is reported, not overwritten. Cache keys bind code, model identity, external LLaVA
source, CLIP checkpoint, settings and inference input. Changing the algorithm may
rerun inference but does not imply redownloading medical weights.
BiomedCLIP's tiny external BiomedBERT architecture config is separately pinned and
hashed; full BiomedCLIP weights supply its text encoder, so a second BERT model is
not downloaded. The auxiliary config is stored outside the snapshot's asset
fingerprint under `.cache`, with its SHA explicitly bound to the run configuration.

## 2. What is actually integrated

| Tool | Native purpose | Configured image types | Download policy |
| --- | --- | --- | --- |
| CONCH | Fixed tissue-appearance classification | Microscopic histopathology only | Reuse prior pinned snapshot; gated download if missing |
| BiomedCLIP anatomy | Non-exhaustive major-organ similarity catalog | CXR, other X-ray, CT, MRI, gross specimens | Reuse BiomedCLIP; not a specialized CT/MRI diagnostician |
| BiomedCLIP source retrieval | Image AND question matching to another source domain | Same-image-type source cases across supported 2D types | Reuse encoder; only source cases enter index |
| XRV DenseNet | Chest-radiograph finding scores | CXR only | Official 28,382,008-byte checkpoint |
| XRV PSPNet | Lung/heart anatomical masks and crop-aware geometry | CXR only | Official 272,988,989-byte checkpoint |
| CheXagent-2-3B | Short specialist chest-image description | CXR only | Optional, existing local checkpoint only; no default download |

The last tool is enabled only if `artifacts/models/StanfordAIMI--CheXagent-2-3b`
exists (or `checkpoint_path` is changed). Missing optional tools are listed in
`excluded_tools`, not described as tested. Required tools do not silently disappear.
Its remote visual code additionally needs the pinned `albumentations==1.3.1` and
`qudida==0.0.4` preprocessing packages. The explicit preflight rejects a CheXagent
run when either is absent. BF16 loading normalizes custom FP32 positional parameters
and casts image inputs to the SigLIP convolution dtype before inference.
Existing MedSAM remains an optional explicit-ROI adapter. No pretrained lesion
detector is included in this default suite. Interface extensibility is not evidence
that every modality/task, 3D volume or every small model has been validated.

Segmentation supplies native masks in artifacts; the language model currently
receives compact geometry, **not mask pixels as visual tokens**. These are anatomy
masks, not tumors. No claim of segmentation accuracy is made without mask labels.
Image type routing is a model prediction, not verified modality/OOD metadata.
Inspect actual modalities and calls; a tiny random pilot may lack usable CXR cases.

## 3. Algorithm remains simple

Image + question → image-type applicability → finite tool selection → native
evidence → medical VLM continuation. Repeat only within the call/answer budget.

1. Infer image type using the image alone with finite natural modality labels;
   arbitrary numeric labels are avoided because the real LLaVA-Med showed severe
   option-position bias in canary testing. Never mark every PathVQA picture as a
   tissue slide. A transparent question-type cue filters manifestly mismatched
   capabilities. These are applicability heuristics, not clinical truth or OOD scores.
2. The controller chooses `0` (continue) or one prebound tool ID. It no longer
   writes interchangeable expert/capability/scope JSON strings. This constrains
   action syntax, not the free-text medical answer. ROI arguments are generated
   only for tools that require them and remain strictly validated.
3. Execute the actual specialist, with duplicate-request reuse. The evidence
   keeps its native meaning: relative similarities, source-only analogies, anatomy
   masks, unverified specialist descriptions. Unknown is not absence.
4. Insert compact evidence, reiterate the original question and concise answer
   format, and continue with exact committed token IDs. No candidate-answer
   reranking, final-answer replacement, or copying an expert diagnosis into logits.
5. DG variant admits only source-qualified task/modality/capability scopes.
   Source calibration uses **forced single-tool real interventions**, so invalid
   controller syntax cannot falsely make retrieval appear unsupported. Empty
   executions count in utility; runtime failures and nonempty-support counts are
   separately reported. No target labels fit the gate.

The prior entropy trigger is not reintroduced. There is no new binary error
estimator or complex two-stage OOD network. The gate uses real source full-answer
continuous lexical gain and a conservative heuristic margin. It is a source-only
robustness constraint, not a proven clinical safety bound or a full FedDG training
algorithm. A singleton qualification card cannot certify adaptive selection or
multi-tool composition; that remains an experiment, not an assumption.

The controller now generates at most eight action tokens rather than up to 160
JSON tokens; LLaVA-Med context is limited to 1,600 evidence characters after a
3,000-character canary exceeded its 2,048-token expanded image context. Its VLM prefill still
costs time. This implementation rebuilds context after evidence changes, with no
cross-block KV reuse guarantee. Measure latency; do not infer speedup from budget.
One ROI decision includes a second VLM forward; `controller_calls` counts decisions,
while its token usage includes both calls.

Retrieval defaults to **one** image/question-matched source QA, clearly attached to
its different source image. Answers never participate in ranking. This knowledge
variant can still be harmful; use `--retrieval-answers off` as a no-answer ablation.
Unlike the prior three image-only cases, it is not injected indiscriminately into
all modalities. Source calibration determines whether it actually helps.

## 4. Data and comparisons

Default `--dataset both --source-per-group 16 --target-limit 16` requests at most
64 source + 32 target examples, not 32 + 16 total. Every answer is from an actual
dataset, using non-yes/no questions, one question per RGB image.

- PathVQA: retain official train/test roles, with exact-pixel/group leakage audit.
- VQA-RAD: pool official QA splits, deduplicate RGB images, assign fixed hash image
  groups, select one question per image. This **custom image-disjoint resplit** is
  explicitly not an official VQA-RAD test score. Original QA split membership is saved.
- Cross-dataset same-image copies are removed and logged. Groups remain proxies,
  not hospitals/patients. No independent-center DG conclusion follows.

Sampling does not select target diagnoses or favorable outputs. Source sample
requirements remain eight genuine nonempty interventions per scope/domain and two
domains. Splitting a small sample across modalities may leave all DG tools closed.
Report that honestly; first examine task alignment and source support, not thresholds.

Per generalist, compare the same cases, prompt constraints and answer budget:

- Generalist, without tools/controller.
- Each compatible single tool forced before answering (the author remains the VLM).
- `all_evidence`: compatible unprompted tools in configured order **within the same
  call/memory budget**, not an unlimited dense oracle.
- Adaptive no-DG and adaptive source-qualified DG.

Retain an OpenMed comparison in its own already compatible environment:

```bash
bash run_llava_med.sh --python /home/dbw/merit-feddg/.venv/bin/python \
  --generalist openmed --skip-download
```

The 3B OpenMed and 7B LLaVA architectures/sizes differ; compare tool gains within
each backbone first. This command requires OpenMed weights and the optional tools
in that environment. Do not upgrade huatuo's transformers to run Qwen.
Each backbone also supplies its own image router; between-backbone results are
full-system comparisons, not a pure backbone swap with a fixed routing policy.

Outputs: `runs/native-v08/{llava,openmed}/latest.json`, result/qualification JSON,
complete source interventions, image-routing traces, native evidence, per-tool
baselines, timings, and a blinded annotation template. `--dataset pathvqa` runs just
the historical dataset subset. The same seed/limits retain deterministic case selection.

## 5. What would count as progress toward the paper

This code addresses demonstrated engineering/semantic confounds. It does **not**
establish an ICLR contribution or improved medical performance by itself.

First verify LLaVA baseline correctness and supported tool execution, then compare
actual medical factuality (blinded review), rescue/harm, evidence dependence,
call rate and latency. Lexical `kidney → renal` gains are not clinical improvements.
Check `predicted_target_modalities`, per-capability calls and sequence traces before
claiming multi-capability or mid-generation collaboration. Short answers may only
exercise calls at token 0. Use real long-answer/localization tasks later without
inflating answer lengths merely to manufacture collaboration.

For an academic claim, add predefined real source/target centers, held-out tools
and capabilities, controlled evidence corruption/replay, competitive tool-use
baselines and independent clinical factuality assessment. Frozen single-tool
source qualification alone does not prove domain-generalizing composition.

## Primary implementation sources

- [Microsoft LLaVA-Med source and v1.5 usage](https://github.com/microsoft/LLaVA-Med)
- [TorchXRayVision official models](https://github.com/mlmed/torchxrayvision)
- [Official CONCH package](https://github.com/mahmoodlab/CONCH)
- [BiomedCLIP model card](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)
- [VQA-RAD dataset](https://huggingface.co/datasets/flaviagiammarino/vqa-rad)
- [PathVQA dataset](https://huggingface.co/datasets/flaviagiammarino/path-vqa)

Upstream model/data licenses and research-use restrictions continue to apply.
