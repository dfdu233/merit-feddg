# MERIT-Tx: Literature-Grounded Expert Pool and Proof-Carrying Transactions

## Why this revision exists

Full PathVQA runs showed that adding more specialists does not monotonically improve a strong medical VLM. Some specialists changed the receiver frequently without adding question-specific medical information, and disabling an expert could improve the aggregate result.

MERIT-Tx therefore treats expert availability and expert authority as different concepts. An expert may propose an alternative claim, localize a region, retrieve background knowledge, or directly verify a patient-specific claim. Only patient-specific verification roles can acquire permission to modify the immutable Generalist output, and only after source-only qualification. Target/test labels never select experts, thresholds, or qualification cards.

## Literature-derived design principles

### Capability roles, not model count

MedRAX (ICML 2025) organizes a medical agent around explicit specialist capabilities such as chest-X-ray classification, segmentation, phrase grounding, VQA, and reporting. VILA-M3 (CVPR 2025) similarly integrates medical expert knowledge according to modality/task contracts. DnR (CVPR 2026 Highlight) adds a complementary principle: external visual expertise should be consulted according to the information need of the current question, not injected unconditionally.

MERIT-Tx operationalizes these principles as ExpertRoleCard roles:

- direct_visual_verifier
- spatial_localizer
- proposal_generator
- knowledge_retriever
- broad_embedding_fallback

The runtime covers orthogonal role/capability/failure-group cells before adding correlated redundancy.

### Atomic claims, not task-specific output formats

Following the atomic-fact and decontextualized-claim literature, MERIT-Tx reasons over the smallest self-contained clinical proposition that can be independently verified. VQA normally has one atomic claim. A radiology report contains multiple claims.

For radiology, optional RadGraph-XL grounding converts report observations into observation-centered clinical subgraphs. For pathology and other domains, the self-contained proposition remains the common interface; no radiology-specific ontology is forced onto them.

### Immutable incumbent and minimal revision

The complete Generalist output is generated and frozen before collaboration. Specialists can only propose ADD, DELETE, or REPLACE transactions over atomic claims. Rejected transactions return the exact incumbent output. Report rendering is a minimal patch rather than a second unrestricted report generation.

## Expert authority

### Role authority

commit_authority = never is used for experts whose output is useful context or proposal but is not patient-specific proof. Current examples are CheXagent generated descriptions and MedCPT/PubMed retrieval. Neither can change a patient-specific claim by itself.

commit_authority = source_qualified means that an expert is structurally eligible to verify a patient-specific claim, but it still cannot commit until a source-only qualification card is present.

### SourceQualificationCard

Qualification is indexed by expert, capability, scope, modality, task, and claim_type. A card contains conservative source/development statistics:

- utility_lcb
- harm_ucb
- specificity_lcb
- domains
- n

The deployed permission rule requires positive lower-bound utility, bounded upper-confidence harm, sufficient current-patient-vs-knockoff specificity, and source-domain coverage. The card is a fixed statistical permission, not a trained router or gate.

Build cards from source/development observations:

    python scripts/fit_expert_qualification.py \
      --input runs/source-tx-observations.jsonl \
      --output artifacts/qualification/merit-expert-qualification.json

The fitter rejects rows marked target or test.

## Differential evidence effect

MERIT-Tx prefers expert-native verification signals over a shared VLM receiver residual whenever the expert exposes a meaningful native score.

For candidate claim c and immutable incumbent claim c0:

    m_real  = score_e(I, c)       - score_e(I, c0)
    m_knock = score_e(I_knock, c) - score_e(I_knock, c0)
    D_e     = m_real - m_knock

The knockoff is a matched wrong-patient image/evidence observation from the same modality and expert protocol. This subtracts nonspecific expert/context preference.

Examples:

- CONCH/PLIP: candidate-vs-incumbent image-text similarity margin.
- XRV: matched finding-score margin when the concept is in the native vocabulary.
- segmentation/localization: candidate-specific spatial support relative to the same operator on a matched control.

A nonpositive differential effect is not commit proof.

## Proposer is not the sole validator

If an expert proposes the transaction, its own fault group cannot be the only validator when require_independent_validator is true.

Example for CXR:

    CheXagent -> proposes small left pleural effusion
    XRV / spatial verifier -> patient-specific verification
    MedCPT -> optional contextual corroboration

CheXagent validating itself is forbidden.

For pathology:

    Generalist/other proposer -> alternative diagnosis
    CONCH -> claim-specific image-text verification
    PLIP -> independent pathology image-text verification

CONCH and PLIP are separate fault groups.

## Why the old fixed CONCH catalog is disabled

The legacy conch_tissue tool asked CONCH to choose among ten broad tissue appearance labels. PathVQA frequently asks about a specific disease or lesion, so this adapter measured a different task from the claim needing verification.

The model itself is not retired. The new conch_claim_verifier uses CONCH's published image-text space directly on candidate propositions. This change is structural and answer-blind; it is not selected because a target-test ablation happened to score better.

## Literature-grounded pool

Integrated or transaction-ready:

| Expert | Role | Scope | Authority |
|---|---|---|---|
| TorchXRayVision | direct visual verifier | CXR findings | source-qualified |
| XRV PSPNet | spatial localizer | CXR anatomy | source-qualified |
| CheXagent | proposal generator | CXR observations | never |
| BiomedParse | spatial localizer | multi-modality 2D objects | source-qualified |
| MedSAM | prompted spatial localizer | broad 2D medical images | source-qualified after a real ROI exists |
| CONCH | pathology claim verifier | pathology image-text | source-qualified |
| PLIP | independent pathology claim verifier | pathology image-text | source-qualified |
| MedCPT | knowledge retriever | PubMed | never |
| BiomedCLIP | broad embedding fallback | anatomy / broad retrieval | source-qualified only if the exact role qualifies |

Candidate models intentionally not faked as active coverage:

- MAIRA-2: high-value CXR phrase grounding/report proposal, but gated and needs a dedicated frozen-runtime adapter and parity canary.
- VISTA3D (CVPR 2025): strong public 3D CT segmentation, but requires real 3D volume support; a 2D slice benchmark must not count it as available coverage.
- UNI / Virchow: strong pathology vision foundation models; should enter through a real source-image retrieval index rather than an invented text-verification head.
- RETFound: useful fundus/OCT encoder with public weights; likewise needs a source-image retrieval protocol.
- PanDerm: strong dermatology foundation model, but the released pretrained model generally requires a downstream probe/fine-tune for diagnostic tasks; that conflicts with the current training-free commit rule.

The roadmap is stored in configs/expert_pool_literature.yaml.

## Asset preparation

Audit the literature-selected pool:

    python scripts/prepare_expert_pool_assets.py --print-commands

Download open checkpoints currently supported by the new adapters:

    python scripts/prepare_expert_pool_assets.py --download-open

This downloads vinid/plip and wanglab/medsam-vit-base. CONCH and MAIRA-2 are gated. The helper only prints commands for them; it never bypasses upstream access terms.

## Coverage audit before GPU inference

Do not infer coverage from the number of configured model names.

Run:

    python scripts/audit_transactional_expert_pool.py \
      --manifest /path/to/source-manifest.jsonl \
      --config configs/merit_tx.yaml \
      --output runs/merit-tx-expert-pool-audit.json

The audit performs no expert inference and reads no reference answers. It reports independent fault-group count, patient-specific verifier count, role distribution, missing optional checkpoints, and cases with source-qualified commit authority.

Prompted tools such as MedSAM do not count as available unless a real ROI is present. If no qualification card exists, commit-authorized coverage is zero by design.

## One algorithm for VQA and reports

The shared algorithm is:

    immutable Generalist output
      -> atomic claimization
      -> expert proposal
      -> candidate-specific native verification
      -> matched knockoff specificity
      -> source-qualified proof
      -> atomic transaction
      -> exact fallback or minimal patch

VQA uses one or a few claims and the whole short answer span can be the patch target. MIMIC reports contain multiple observation-centered claims and only approved clinical claims may change; unrelated report content remains incumbent text.

The algorithm has no separate VQA method and report method. Only claimization and domain grounding differ.

## Current implementation boundary

This branch implements:

- immutable transaction representation and patching;
- VQA decontextualization;
- optional RadGraph-XL report claimization;
- literature-grounded role cards;
- source-only expert qualification;
- native real-vs-knockoff differential margins;
- proposer/validator separation;
- PLIP claim verification;
- CONCH claim-specific transaction configuration;
- MedSAM prerequisite-aware registration;
- answer-blind expert-pool audit;
- asset download/audit helpers.

It deliberately does not claim a new benchmark improvement yet.

Before a new full PathVQA or MIMIC run:

1. produce source/development qualification observations;
2. freeze qualification cards;
3. audit effective commit-authorized coverage;
4. run candidate-oracle and first-divergence diagnostics;
5. run a small MERIT-Tx source canary;
6. only then evaluate the untouched target/test benchmark.

## References

- Savage et al., MedRAX, ICML 2025.
- Nath et al., VILA-M3, CVPR 2025.
- DnR: Draft and Refine with Visual Experts, CVPR 2026 Highlight.
- Ma et al., Segment Anything in Medical Images (MedSAM), Nature Communications 2024.
- Lu et al., CONCH, Nature Medicine 2024.
- Huang et al., PLIP/OpenPath, Nature Medicine 2023.
- Jain et al., RadGraph, NeurIPS Datasets & Benchmarks 2021.
- Delbrouck et al., RadGraph-XL, ACL Findings 2024.
