# MERIT-Tx v3: Source-Qualified Expert Portfolio

## Motivation

A medical expert pool is not a monotone resource: adding another specialist can reduce the performance of a strong Generalist. MERIT has already observed cases where disabling an expert improves the aggregate output. This is expected when specialists are correlated, answer a different clinical attribute, or exercise an unsafe veto over another useful specialist.

MERIT-Tx v3 separates four questions:

1. Should a model enter the candidate library?
2. What evidence role is its native interface allowed to play?
3. Does its action signal qualify on source/development data?
4. Does the expert still have positive marginal utility when combined with the other qualified experts?

Only the fourth question determines the frozen runtime verifier portfolio. The target/test set answers none of these questions.

## 1. Literature-grounded candidate library

### Capability-oriented medical agents

MedRAX (ICML 2025) explicitly separates CXR disease classification, segmentation, phrase grounding, report generation and visual QA tools. Its official implementation exposes selective tool initialization rather than requiring every available tool:

- Official code: https://github.com/bowang-lab/medrax
- The repository lists ChestXRayClassifierTool, ChestXRaySegmentationTool, XRayPhraseGroundingTool, ChestXRayReportGeneratorTool and XRayVQATool, and initialize_agent(..., tools_to_use=selected_tools).

VILA-M3 (CVPR 2025) uses medical expert model cards and distinct 3D/2D segmentation and CXR classification experts. Expert triggering is therefore a capability/model-card problem rather than a flat ensemble.

Draft and Refine (DnR, CVPR 2026 Highlight) demonstrates that external visual experts should be consulted conditional on the current query and visual information need. MERIT does not equate DnR visual utilization with medical correctness, but adopts the principle that expert availability is not sufficient reason to inject it.

### Candidate model admission criteria

A model is added to configs/expert_pool_literature.yaml only when all of the following are recorded:

- peer-reviewed/top-venue or top-journal evidence for the claimed capability;
- exact modality and input dimensionality;
- native task interface exposed by the released code/weights;
- whether a new trained head is required;
- whether the output is patient-specific verification, localization, retrieval, or proposal generation;
- expected failure group;
- public/auditable weights or an explicit gated-access path.

This prevents a generic embedding foundation model from silently becoming a diagnostic expert.

## 2. Evidence roles are not interchangeable

| Role | Native output | May commit alone? |
|---|---|---:|
| direct_visual_verifier | candidate-specific image score / native class evidence | source-qualified only |
| spatial_localizer | mask / box / anatomy geometry | no, unless the exact transaction has source-qualified spatial verification |
| proposal_generator | free specialist text / report | no |
| knowledge_retriever | literature passages | no |
| broad_embedding_fallback | broad image-text/anatomy similarity | source-qualified only in exact cells |

A model's publication venue never bypasses these rules. CheXagent may propose CXR observations but generated text is not independent proof; MedCPT retrieves knowledge but cannot prove the current patient has a finding; MedSAM/VISTA3D localize structures but localization is not a diagnosis.

## 3. Roadmap driven by uncovered capability cells

### Pathology

- CONCH, Nature Medicine 2024: integrated candidate-specific pathology image-text verifier.
- PLIP/OpenPath, Nature Medicine 2023: independent pathology verifier.
- MUSK, Nature 2025: high-priority gated candidate. The official code uses musk_large_patch16_384 and documents with_head=True for zero-shot image-text retrieval and zero-shot image classification. Official code: https://github.com/lilab-stanford/MUSK
- UNI/Virchow remain source-image retrieval candidates until a real retrieval index is provided.

### Dermatology

- MONET, Nature Medicine 2024: integrated optional candidate-specific
  dermatology image-text verifier. The official release exposes Hugging Face
  zero-shot image classification/concept annotation, so no new diagnosis head
  is trained by MERIT.
- PanDerm remains a foundation-encoder candidate when its downstream use
  requires a new trained task head; that head is not silently invented.

### Ophthalmology

- FLAIR, Medical Image Analysis 2025: open-weight fundus language-image model
  with zero-shot generalization; next preferred open retinal verifier adapter.
- EyeCLIP, npj Digital Medicine 2025: specialized candidate when fundus/OCT/
  ophthalmic data are present. Its official repository contains zero_shot.py
  and retrieval.py: https://github.com/Michi-3000/EyeCLIP
- RETFound remains an image-encoder/retrieval candidate rather than an invented
  text verifier.

### Echocardiography / ultrasound

- EchoCLIP, Nature Medicine 2024: native zero-shot and semantic-search interface for echocardiography: https://github.com/echonet/echo_CLIP
- Generic ultrasound foundation encoders such as USFM can support source-image retrieval but do not receive diagnosis authority without a native training-free claim interface.
- EchoCLIP is not counted as coverage for a still-image ultrasound benchmark until the frame/video contract is implemented.

### CT / MRI

- VISTA3D, CVPR 2025: 3D CT segmentation foundation model; code and weights: https://github.com/Project-MONAI/VISTA
- The current MONAI release also exposes NV-Segment-CTMR for CT+MRI automatic segmentation. These are spatial localizers and require true volume inputs.
- Merlin, Nature 2026: released 3D CT vision-language model: https://github.com/StanfordMIMI/Merlin. It is a direct CT verifier/retriever candidate only when volumetric CT is actually available.

### Broad 2D medical image verification

- BiomedCLIP is already integrated but treated as a broad fallback.
- MedSigLIP is a gated broad frozen image-text candidate across CXR, CT/MRI
  slices, pathology, dermatology and ophthalmology; it gains no authority until
  the exact source cell qualifies.
- MedImageInsight is a broad Microsoft image-text embedding candidate spanning
  X-ray, CT, MRI, dermatology, OCT/fundus, ultrasound, histopathology and
  mammography. It is not counted as active coverage until an official local
  runtime is pinned and source-qualified.

## 4. Source qualification v3: prevalence is not precision

MERIT-Tx v2 used one specificity lower bound. Source diagnosis exposed a structural problem: transactions whose task score did not change were counted as directional failures. When useful/harmful proposals are sparse, even a perfect discriminator on consequential cases can therefore be forced below the threshold.

v3 separates:

- action_rate_lcb: frequency of a nonzero differential expert action signal;
- utility_lcb: conservative utility among all positive support actions, including neutral actions;
- harm_ucb: conservative harmful-action rate among support actions;
- support_precision_lcb: precision among consequential positive support actions only;
- veto_precision_lcb: precision among consequential negative/veto actions only;
- action-specific domain coverage.

Neutral proposals still matter to utility and computation, but they no longer pretend to be evidence that the direction classifier was wrong. Positive support and negative veto remain separately qualified.

## 5. Why per-expert qualification is insufficient

Two specialists can be independently source-qualified yet harmful together. For example, expert A can safely support one region of the source distribution while expert B has a separately-qualified veto that suppresses those corrections. That is a set interaction, not a single-expert calibration problem.

This design is motivated by broader subset/ensemble work. NeurIPS 2024 Most Influential Subset Selection shows that collective influence is non-additive and additive/greedy individual scores can fail. Selective prediction motivates keeping an abstain/incumbent option rather than forcing coverage.

## 6. Interaction-aware source portfolio

For a fixed modality x task x claim_type cell let Q be the individually qualified patient-specific verifiers. For every independent-fault-group subset S of Q up to the frozen maximum size, MERIT replays the exact source transaction rule:

1. collect qualified support signals from S;
2. exclude proposer fault groups from independent support;
3. apply separately qualified vetoes;
4. commit only when the frozen transaction rule would commit;
5. score that action with the already-frozen source task metric.

For reports, multiple claim edits from one patient/study are not treated as independent observations. If a portfolio commits several edits, the worst committed transaction is retained for that source group.

The portfolio records population expected utility and LCB, action utility, coverage LCB, harmful-intervention UCB, beneficial-vs-harmful precision LCB, action-domain coverage and estimated call cost.

A non-empty subset is feasible only if all frozen source safety constraints hold. Ranking is lexicographic rather than a target-tuned weighted sum: utility LCB -> action utility LCB -> coverage -> lower harm/cost -> fewer experts.

If no non-empty subset qualifies, the selected portfolio is empty and MERIT returns the immutable incumbent.

## 7. Expert-removal and interaction diagnostics

For a selected portfolio S, the policy stores leave-one-out marginal utility:

    Delta_remove(e) = U(S) - U(S without e)

For every individually-qualified expert excluded from S it also stores the
direct add-back effect:

    Delta_add(e) = U(S union {e}) - U(S)

Thus a negative Delta_add is an explicit source-only statement that restoring
that expert would reduce frozen portfolio utility. This directly operationalizes
the observed "masking an expert improves performance" phenomenon without using
target ablations.

It also records pair interaction:

    I(e_i,e_j) = U({e_i,e_j}) - U({e_i}) - U({e_j})

If removing an expert improves source utility, that fact is recorded explicitly instead of assuming that more specialists must help. Target/test results never remove or restore experts.

## 8. Frozen workflow

1. Freeze the Generalist incumbent and one proposal distribution **before**
   reading source references.
2. Partition source/development groups into three patient/study/image-disjoint
   sets: qualification, portfolio-selection, and fresh-canary. Never split by QA
   row when several questions share one image/patient. The answer-blind splitter
   additionally unions identical image bytes even when group IDs differ:

    python scripts/split_merit_tx_source.py --manifest /path/to/source.jsonl --baseline /path/to/generalist.json --candidate /path/to/proposal.json --references /path/to/references.json --output runs/merit-tx-source-splits

3. Build qualification observations on source-Q:

    python scripts/build_merit_tx_source_observations.py --manifest /path/to/source-q.jsonl --baseline /path/to/q-generalist.json --candidate proposal=/path/to/q-proposal.json --references /path/to/q-references.json --config configs/merit_tx.yaml --output runs/source-q-observations.jsonl

4. Fit v3 per-expert qualification on source-Q:

    python scripts/fit_expert_qualification.py --input runs/source-q-observations.jsonl --output artifacts/qualification/merit-expert-qualification-v3.json

5. Independently build transaction observations on source-P using the **same
   frozen proposal mechanism**, then fit the interaction-aware portfolio:

    python scripts/build_merit_tx_source_observations.py --manifest /path/to/source-p.jsonl --baseline /path/to/p-generalist.json --candidate proposal=/path/to/p-proposal.json --references /path/to/p-references.json --config configs/merit_tx.yaml --output runs/source-p-observations.jsonl

    python scripts/fit_expert_portfolio.py --input runs/source-p-observations.jsonl.transactions.jsonl --qualification-cards artifacts/qualification/merit-expert-qualification-v3.json --config configs/merit_tx.yaml --output artifacts/qualification/merit-expert-portfolio-v1.json

   The fitter hard-fails if any source-P group appeared in source-Q.
6. Audit effective coverage after source qualification and portfolio selection:

    python scripts/audit_transactional_expert_pool.py --manifest /path/to/source.jsonl --config configs/merit_tx.yaml --qualification-cards artifacts/qualification/merit-expert-qualification-v3.json --portfolio-policy artifacts/qualification/merit-expert-portfolio-v1.json --output runs/merit-tx-expert-pool-audit.json

7. Run source-C as a fresh canary. The canary executable hard-fails if any
   source-C group overlaps source-Q or source-P. Only if source-C shows nonzero
   useful intervention with bounded harm may the exact frozen policy run once
   on untouched target/test data.

## 9. Asset policy

Run python scripts/prepare_expert_pool_assets.py --print-commands. Open checkpoints already supported by runtime adapters may be downloaded with --download-open. Gated or research-only models are printed but not silently downloaded. Downloading a checkpoint never grants runtime authority.

## References

- Fallahpour et al., MedRAX, ICML 2025; https://github.com/bowang-lab/medrax
- Nath et al., VILA-M3, CVPR 2025.
- Jeong et al., Draft and Refine with Visual Experts, CVPR 2026 Highlight.
- Hu et al., Most Influential Subset Selection, NeurIPS 2024.
- Geifman & El-Yaniv, SelectiveNet, ICML 2019.
- Xiang et al., MUSK, Nature 2025; https://github.com/lilab-stanford/MUSK
- Shi et al., EyeCLIP, npj Digital Medicine 2025; https://github.com/Michi-3000/EyeCLIP
- Christensen et al., EchoCLIP, Nature Medicine 2024; https://github.com/echonet/echo_CLIP
- He et al., VISTA3D, CVPR 2025; https://github.com/Project-MONAI/VISTA
- Blankemeier et al., Merlin, Nature 2026; https://github.com/StanfordMIMI/Merlin
- CONCH, Nature Medicine 2024; PLIP/OpenPath, Nature Medicine 2023.