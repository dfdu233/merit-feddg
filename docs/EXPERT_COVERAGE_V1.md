# Capability-selected expert pool v1

This opt-in branch preserves historical methods and results. It evaluates one
expanded-pool arm on the unchanged 6,719-case official PathVQA manifest; cached
baselines are not regenerated. No training or test-score-based selection occurs.

## Selection and limits

The existing question parser is unchanged. Published capability cards constrain
input modality, requested attribute, and (where necessary) explicitly established
site/entity. Greedy marginal attribute coverage and frozen priority select at most
three tools. Eligibility is **not** correctness, utility confidence, or a medical
diagnosis. Unknown requests retain an explicit coverage gap. This is not a claim
that every medical modality, organ, disease, or acquisition protocol is covered.

UniMed-CLIP, FLAIR, and MONET augment the existing pool. Eight fixed catalogs share
three frozen image/text models; they are not eight independent specialists.
Catalog similarities and native logits retain their meaning and are not converted
into calibrated diagnosis probabilities. UniMed catalogs use the first published
template, not the paper's multi-template ensemble. MONET's small bullae/blister
catalog does not provide unrestricted dermatology diagnosis.

The `source_family` fields are provenance grouping hints, not verified independent
training sets. In particular, `UniMed-LC25000` identifies an upstream evaluation
catalog, **not** proof that LC25000 is the pretraining source. UniMed pretraining
includes Quilt-1M and MIMIC-CXR: UniMed and Quilt evidence must not be counted as
independent. Retinal assembly overlap has not been ruled out. BUSI-related and
BreastMNIST resources must not be treated as independent sources.

3D volume and video coverage, unsupported sites, unavailable checkpoints, and
gated-license resources remain explicit gaps. MedSigLIP is not enabled; its model
card also lists SLAKE among training resources. No gated license is auto-accepted.

## Resources and environment

Weights and upstream sources remain outside Git in the existing artifacts tree.
`scripts/prepare_coverage_experts.py` downloads only pinned inference resources,
checks publisher LFS hashes, and writes local resource manifests. No shared
environment upgrade or dependency installation was needed.

| Model | Public repository | Revision |
|---|---|---|
| UniMed-CLIP | UzairK/unimed-clip-vit-b16 | 84a67cb0331b511b630136e4878beaa5a3fdc657 |
| FLAIR | jusiro2/FLAIR | 5f6bdd0a068353dc41a896ba3abdd7c0f6d35938 |
| MONET | chanwkim/monet | 1d1efd0b8d61bde82cde22d2e93c28d73441f29e |
| ClinicalBERT | emilyalsentzer/Bio_ClinicalBERT | d5892b39a4adaed74b92212a44081509db72f87b |
| BiomedBERT | microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract | d673b8835373c6fa116d6d8006b33d48734e305d |

Official inference code: [UniMed-CLIP](https://github.com/mbzuai-oryx/UniMed-CLIP),
[FLAIR](https://github.com/jusiro/FLAIR), [MONET](https://github.com/suinleelab/MONET).
The adapter preserves their preprocessing and frozen checkpoint parameters.
The only legacy state-dict compatibility exception removes deterministic
position-ID buffers after exact value comparison with the runtime buffers;
learned parameter loading remains strict.

## Acceptance and experiment

- CPU regression: 1,026 passed (15.09 s), local log
  `runs/coverage-preparation/cpu-tests-final.log`.
- Actual CUDA smoke: four calls spanning UniMed CT/pathology, FLAIR fundus, MONET
  dermatology returned finite native scores with frozen weights. Local summary:
  `runs/coverage-native-smoke-v3/summary.json`. This is execution evidence, not
  clinical accuracy or external validation.
- Full runner: `scripts/run_coverage_pathvqa.py`; configuration:
  `configs/expert_coverage_v1.yaml`. Preparation freezes full manifest, original
  caches, configuration, model resources, source files and scorer identity.
- Reuse requires matching request and image identities. Each selected evidence
  item must actually appear in the final transport; budget omission stops the run.
- Engineering canary selects the first newly routed case in each deterministic
  scheduling lane, without looking at answers. Original compact token parity is
  required before accepting any new candidate. All final answers use the original
  image and frozen formal generalist, input limit 8192 and output limit 1024.
- No compatible evidence reuses the native MERIT+Quilt incumbent and is reported
  separately, never as a successful new expert call.

All raw case outputs, images, weights, and restricted data stay local. Full metrics
are not available until actual inference, exact completeness checks, and the
unchanged formal scorer finish. Runtime failures are not medical negative evidence.
