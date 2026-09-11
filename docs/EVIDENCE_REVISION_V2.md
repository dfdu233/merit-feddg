# Training-free evidence revision experiment

This additive protocol preserves the published `verified_packets` implementation
and configurations. Base: fc93dcad9965d5a979960c7fd94f7e4bd8966101.
No GPU efficacy results have been obtained for this successor.

## Motivation and implemented changes

The published SLAKE case 0250 changes spleen to liver using BiomedCLIP, then
reverts after arbitration. The configured verifier has the same model identity
as that expert; the existing guard abstains rather than scores this evidence.
An unavailable verification is not a negative finding. The new candidate-default
policy is an explicit experimental alternative, NOT a correctness guarantee.

Eight arms run on the same complete manifest. VQA experiments use the frozen
`anchor-ce-v1` CE/OE prompt contract so the Generalist and final scores remain
comparable with the existing VQA-RAD/SLAKE evaluation; a legacy generic prompt
remains available only for non-paper exploratory inputs.

| Arm | Change |
| --- | --- |
| generalist | No expert evidence |
| semantic_all | Original semantic packet serialization |
| compact_rows | Original row representation |
| compact_all | Original compact representation |
| compact_verified | Original baseline-default arbitration |
| compact_guarded | Same candidate AND verifier audit; retain candidate on abstention, veto only on explicit rejection |
| compact_geometry | compact_all with measured native-grid mask moments |
| compact_spatial | compact_geometry with the existing parameter-free spatial relation operator |

The two arbitration arms share the exact candidate and score trace; the new arm
does not spend extra verifier calls. Same-checkpoint checks in the new protocol
are resolved before loading a verifier. Scores are still cosine preferences,
NOT expert correctness probabilities. An abstention remains an abstention even
when the candidate is retained. Candidate-default can retain wrong evidence;
compare it with both old arbitration and no gate.

Geometry carries native-grid coordinates, weighted centroid and mask moments.
Parent crop/coordinate metadata is preserved; no millimetres, normality claims,
lesion existence or calibrated confidence are invented. Full arrays stay in the
audit. Spatial mode really consumes supported arrays through the existing
frozen bridge; it is not a new ProxyCLIP implementation. Geometry and spatial
arms share the same text configuration, but budget-admitted packet identities
must still be audited. Geometry versus compact_all can differ in packet admission
because geometry costs tokens; it is not automatically a pure geometry ablation.

## Run

From repository root, with the existing model environment and artifacts:

```bash
python -m merit_feddg.matched_evaluation \
  --manifest /path/to/full_answer_blind_manifest.jsonl \
  --config configs/matched_evidence_revision.yaml \
  --output runs/matched-evidence-revision \
  --artifacts artifacts --protocol evidence_revision
```

The checked-in VQA configuration uses `anchor-ce-v1`. Reuse is permitted only
when the sanitized Generalist package and expert run pass the runner's existing
full manifest, model, route and cache-identity checks; otherwise regenerate the
matched baseline. No medical reference answers are allowed in generation
manifests. Existing sharding only
distributes work; the merger must cover the full manifest before scoring.

## Diagnose completed results without running models

```bash
python -m merit_feddg.revision_audit \
  --baseline RUN/generalist.json --candidate RUN/compact_all.json \
  --verified RUN/compact_guarded.json \
  --scores /path/to/frozen_scores.json \
  --groups /path/to/existing_hospital_metadata.json \
  --output /path/to/revision-audit.json
```

`--scores` and `--groups` are optional. Scores have schema
`{"evaluator_id":"fixed-version", "cases":{"case-id":{"baseline":0,"candidate":1}}}`.
Groups map every case ID to an existing hospital/domain label. No domain labels
are inferred, no hospital split is created, and labels never enter inference.
Missing IDs and mismatched candidate/final outputs fail rather than silently
intersecting samples. Audit outputs contain expert scalar packets and array
references, not dense mask blobs. Report beneficial/harmful/tied decisions,
fallback reasons, per-domain candidate/final gains, and cost. These scores are
diagnostic, not permission to tune thresholds on the test set. Use a medically
appropriate fixed evaluator; CLOSED does not imply binary on SLAKE.

## Research lineage and limits

- ProxyCLIP (ECCV 2024): external spatial correspondence can organize CLIP
  features without learning cross-model projection. Code:
  https://github.com/mc-lan/ProxyCLIP/blob/main/open_clip/transformer.py
  (`custom_attn`). Our existing spatial bridge uses region pooling and return,
  not the paper's dense feature affinity; no reproduction claim.
- ARO (ICLR 2023): evaluate attribute/relation sensitivity before treating
  contrastive similarity as a verifier. Code:
  https://github.com/mertyg/vision-language-models-are-bows/tree/main/dataset_zoo
  Medical relation qualification is still outstanding.
- Semantic Entropy (ICLR 2023; Nature 2024 extension): multiple answers, semantic clustering and
  entropy; https://github.com/jlko/semantic_uncertainty/blob/main/semantic_uncertainty/uncertainty/uncertainty_measures/semantic_entropy.py
  Optional implementation described below; two verifier renderings are not semantic entropy.
- WILDS (ICML 2021): real distribution shifts and group-aware evaluation;
  https://github.com/p-lambda/wilds . The new group audit is evaluation only,
  not an implemented domain-generalization learning algorithm.

No new specialist weights, domain calibration, entropy-based fitting, or trained
router are introduced. Broader expert coverage, independently qualified medical
verification, semantic uncertainty and robustness under real hospital shifts
remain subjects for validation. No SOTA or publication-level novelty claim follows from
these implementation changes. Preserve old runs and compare on full manifests.

## Published uncertainty estimator comparison (optional)

Use `configs/matched_uncertainty_comparison.yaml` with the same
`--protocol evidence_revision` command. Prepare the NLI checkpoint explicitly at
the configured path; no implicit downloads or training occur. Current stochastic
backend support is LLaVA-Med v1.5 only. Spatial sampling is rejected rather than
silently dropping the spatial operator. The default 5 samples at T=1 are an
experiment budget choice, not a claim to reproduce paper hyperparameters.

Implemented in `semantic_uncertainty.py`, with official code provenance pinned
to `jlko/semantic_uncertainty@a8d9aa8cecd5f3bec09b19ae38ab13552e0846f4`:

| Estimator | Reference functions | Our implementation |
| --- | --- | --- |
| Length-normalized predictive entropy | predictive_entropy | Negative mean of sampled mean-token log probabilities |
| Semantic entropy | get_semantic_ids, logsumexp_by_id, predictive_entropy_rao | Strict bidirectional NLI clusters; normalized length-normalized likelihood mass; entropy over clusters |
| Discrete semantic entropy | get_semantic_ids, cluster_assignment_entropy | Entropy of empirical semantic-cluster frequencies |

Reference papers: https://arxiv.org/abs/2302.09664 (ICLR 2023 Spotlight),
https://www.nature.com/articles/s41586-024-07421-0 (Nature 2024).
Official implementation:
https://github.com/jlko/semantic_uncertainty/blob/a8d9aa8cecd5f3bec09b19ae38ab13552e0846f4/semantic_uncertainty/uncertainty/uncertainty_measures/semantic_entropy.py

Adaptations are explicit: prepend the whole question to every answer for NLI;
use strict bidirectional entailment; deterministic representative clustering
never reassigns an already assigned sample; log-sum-exp is numerically stable.
NLI is frozen and local, has no CE/OE branch and refuses silent truncation.
Empty or truncated generations make estimates unavailable. The method currently
does not score semantic entropy if NLI fails; failures must be reported.

Each representation's exact semantic prompt is reconstructed from its native
evidence and original packing config. Five independent ancestral generations
use a deterministic per-case seed inside a restored RNG context, preserving
other arms' generation state. The three estimators share these same samples.
The extra gate compares baseline-context versus evidence-context uncertainty
and selects one of the original deterministic answers; ties/unavailable values
retain the ungated candidate as a declared fallback ablation. This comparative
gate is a project-level composition, NOT an algorithm or improvement claimed by
the cited papers. Low entropy can reflect a stable error or evidence anchoring.

The default grid adds 3 estimators x 3 evidence representations, not a learned
weighted blend. Use existing independent development data (do not split the
current target test set) for choice of combinations; freeze before external
evaluation. Since VQA-RAD/SLAKE test errors have already informed development,
report that limitation and use an untouched external evaluation for confirmatory
claims. No combination is selected automatically from test scores.

Candidate-default fallback and geometry serialization are engineering controls;
they are not attributed to a paper or promoted as novel published algorithms.
New expert architectures and domain adaptation methods require separate verified
paper/code entries before implementation. DG group summaries remain evaluation,
not a trained/adapted domain-generalization module.

## Validation of this implementation

The complete CPU regression suite passed: `703 passed, 17 skipped`
(`python -m pytest -q -o addopts=''`). Ruff on the changed Python modules/tests
and `git diff --check` also passed. Tests cover analytical entropy values,
bidirectional clustering, seeded sampling/RNG restoration, exact candidate
selection, evidence transport, and the matched runner with mocked backends.
No new GPU benchmark has been run. The optional local DeBERTa checkpoint must
be prepared before running uncertainty comparisons; these tests do not establish
medical accuracy, uncertainty calibration, or improved domain generalization.
