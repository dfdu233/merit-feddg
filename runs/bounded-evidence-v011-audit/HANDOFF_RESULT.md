# v0.11 bounded-evidence source validation

Date: 2026-09-07 UTC

This report records a source-only engineering and diagnostic run. Token-F1 is a
lexical overlap metric, not clinical factuality, hallucination rate or patient safety.
All recorded domains are dataset/hash proxies, not independent hospitals.

## Frozen runtime and assets

- Repository base revision: `268cb4c13b2823f5eed6891e81285f508e4797f7`
- Python: `/home/dbw/.runtime/miniconda3/envs/huatuo/bin/python`
- Generalist: `/home/dbw/ANCHOR/hf_cache/llava-med-v1.5-mistral-7b`
- LLaVA-Med source: `/home/dbw/ANCHOR/data/medheval/code/baselines/Med-LVLMs/llava-med-1.5`
- CheXagent: `/home/dbw/merit-feddg/artifacts/models/StanfordAIMI--CheXagent-2-3b`
- Full source: `runs/native-v09-value-full-gpu1/data/both-source16-target16-seed17/source.jsonl`
- Sealed target manifest: `runs/native-v09-value-full-gpu1/data/both-source16-target16-seed17/target.jsonl`
- References: `runs/native-v09-value-full-gpu1/data/both-source16-target16-seed17/references.json`
- GPU: visible device 0, 48 GiB class; models/data were reused offline.

The inference environment reported Torch 2.0.1 and Transformers 4.37.2. The project
test environment completed 486 tests with two environment skips; Ruff, Bash syntax
and `git diff --check` passed.

## Failures found and repaired before accepting results

1. With `device_map: auto`, CheXagent loaded on CPU beside the resident LLaVA model.
   A single generation remained CPU-bound for more than ten minutes. The configured
   CheXagent profile now requests an explicit whole-model CUDA mapping, with a unit
   test covering the mapping.
2. The original `next_scores` appended a committed prefix and performed a one-step
   full prefill. On real FP16 LLaVA this is not numerically identical to the original
   autoregressive KV-cache path. A real source case diverged at token 7: production
   selected `right` at 11.7734375 while re-prefill selected `superior` at 11.78125.
   The adapter now forces the committed tokens through production generation and
   returns the following score vector. The same case then matched 16/16 tokens.
3. Trace and handoff text were updated from the obsolete `single_token_reprefill`
   description to `production_kv_replay`.

No baseline assertion, sample, KL budget or domain threshold was relaxed.

## Commands

Final canary:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
MERIT_LLAVA_PYTHON=/home/dbw/.runtime/miniconda3/envs/huatuo/bin/python \
bash run_llava_med.sh --study evidence --evidence-stage source \
  --source-per-group 2 --target-limit 1 --chexagent on \
  --output runs/bounded-evidence-canary-v011-final
```

Frozen full source diagnosis:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
MERIT_LLAVA_PYTHON=/home/dbw/.runtime/miniconda3/envs/huatuo/bin/python \
bash run_llava_med.sh --study evidence --evidence-stage source \
  --source-manifest runs/native-v09-value-full-gpu1/data/both-source16-target16-seed17/source.jsonl \
  --target-manifest runs/native-v09-value-full-gpu1/data/both-source16-target16-seed17/target.jsonl \
  --references runs/native-v09-value-full-gpu1/data/both-source16-target16-seed17/references.json \
  --source-per-group 16 --target-limit 16 --chexagent on \
  --output runs/bounded-evidence-source-v011-kvreplay-full
```

The full result root is
`runs/bounded-evidence-source-v011-kvreplay-full/llava/e634250537642f06`.

## Engineering acceptance

| Check | Full-source result |
| --- | ---: |
| Source cases | 64/64 |
| Target generations | 0 |
| Strength-specific records | 82 |
| Unique real expert executions | 41 |
| Cases with / without a compatible expert | 29 / 35 |
| Tool executions adopted / failed | 41 / 0 |
| Production baseline tokens matched | 992/992 |
| Real-guidance active token steps | 1,246 |
| Format-control active token steps | 1,244 |
| Maximum token KL | 0.02000 (budget 0.02) |
| Maximum cumulative case KL | 0.25635 (budget 0.32) |
| Maximum active evidence position | 15 (16-token lifetime) |
| Single-original-image branches | all |
| Peak PyTorch allocated / reserved | 23.91 / 46.84 GiB |
| Sum of per-case source diagnostic time | 1,982.83 s |

Unique tool coverage was: CONCH 17 cases, XRV findings 10, CheXagent 11,
BiomedCLIP anatomy 2 and XRV anatomy segmentation 1. Some cases had multiple tools,
so these sum to 41 executions over 29 cases.

## Source lexical results

The following means are over applicable expert/case interventions; direct context is
identical for the two strengths and is shown once.

| Expert/scope | Cases | Direct context delta | Guided 0.25 | Format 0.25 | Guided 0.50 | Format 0.50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CONCH tissue | 17 | +0.00254 | -0.00017 | -0.00017 | -0.00017 | -0.00017 |
| XRV CXR findings | 10 | +0.00646 | -0.02667 | +0.00000 | -0.02667 | -0.02667 |
| CheXagent description | 11 | +0.01771 | +0.00000 | +0.00000 | -0.02424 | -0.02424 |
| BiomedCLIP anatomy (CT+CXR) | 2 | +0.00000 | +0.00000 | +0.00000 | +0.00000 | +0.00000 |
| XRV anatomy segmentation | 1 | +0.00000 | +0.00000 | +0.00000 | +0.00000 | +0.00000 |
| All applicable interventions | 41 | +0.00738 | -0.00658 | -0.00007 | -0.01308 | -0.01308 |

Direct context improved 8/41, harmed 3/41 and was unchanged on 30/41. At strength
0.25, guided minus format averaged -0.00650: one case was harmed and none improved.
At strength 0.5, guided minus format averaged exactly zero in Token-F1. Guided and
format text differed in only 3/41 interventions at 0.25 and 4/41 at 0.5. Thus most
bounded changes were absent or attributable to the evidence-envelope format rather
than specialist content.

The clearest harmful real-content case asked whether a lung was hyperinflated on one
or both sides (reference: `both sides`). Baseline and format control answered both
sides; XRV-guided strength 0.25 changed this to `right side`, reducing Token-F1 by
0.26667. Direct XRV context happened to retain both sides and improved brevity/F1,
showing that the present residual decoder does not inherit direct-context benefit.

A positive direct-context example asked for chest-radiograph plane (reference: `pa`).
CheXagent direct context changed AP to PA and raised Token-F1 by 0.14286, while both
bounded strengths left the incorrect AP baseline unchanged. This supports the claim
that a specialist can occasionally supply useful content, but not that the current
bounded bridge reliably transfers it.

## Calibration and decision

All six cards returned `qualified: false`, status `proxy_or_unverified_domains`, and
strength 0.0. The calibration requires at least two real source domains and eight
independent groups per selection/confirmation partition in each domain. These four
domains are proxy splits; several scopes also have only one case or one proxy domain.

Therefore no deployable/source-calibrated strategy was fitted or unlocked. No target
evaluation or target hyperparameter search was run. The next defensible experiment is
to acquire declared real source-domain coverage and complete blinded medical review
of tool evidence and answers, then rerun source calibration unchanged.
