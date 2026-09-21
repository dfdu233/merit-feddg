# MERIT-Tx: Literature-Grounded Expert Pool and Proof-Carrying Transactions

The detailed v3 subset-selection rationale and literature matrix are in
[EXPERT_PORTFOLIO_V3.md](EXPERT_PORTFOLIO_V3.md).

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

### SourceQualificationCard v3

Qualification is indexed by expert, capability, scope, modality, task, and
claim_type.  The previous v2 statistic conflated two different questions:
"does this expert act often?" and "when it acts on a consequential proposal,
does it point in the correct direction?"  This was especially damaging when
most frozen candidate transactions had zero task-utility change.

The v3 schema (`merit-expert-qualification-v3`) separates:

- `action_rate_lcb`: how often the expert produces a nonzero differential;
- `utility_lcb`: conservative task utility over positive-support actions;
- `harm_ucb`: upper bound on harmful positive-support actions;
- `support_precision_lcb`: beneficial fraction **conditional on consequential**
  positive-support actions;
- `veto_precision_lcb`: harmful fraction **conditional on consequential**
  negative/veto actions;
- support/veto action counts and their domain coverage;
- neutral-action counts, retained for coverage/cost but not mislabeled as
  directional failures.

This keeps useful-action prevalence, conditional discrimination, and harm risk
as separate quantities.  A stable but mostly irrelevant expert therefore
cannot look reliable merely by rarely changing anything, while a sparse expert
is not automatically failed because most frozen proposals were score-neutral.

Build cards only from source/development observations:

    python scripts/fit_expert_qualification.py \
      --input runs/source-tx-observations.jsonl \
      --output artifacts/qualification/merit-expert-qualification-v3.json

The fitter rejects target/test rows.

### Interaction-aware Source Expert Portfolio

Per-expert qualification is necessary but not sufficient. Two experts can each
look acceptable alone while their joint use is redundant or harmful, exactly
matching the observed "disabling one expert improves the system" failure mode.

MERIT-Tx v3 therefore replays every feasible, independent verifier subset over
the **same frozen source transactions**.  For each modality/task/claim cell it
records:

- conservative population utility and action-conditional utility;
- harmful-commit upper bound and beneficial-action precision lower bound;
- intervention coverage;
- expert cost units;
- leave-one-out marginal utility/harm;
- pairwise interaction utility.

The empty portfolio is a first-class option. If no non-empty source portfolio
has positive conservative utility with bounded harm, the frozen policy selects
no verifier and preserves the immutable Generalist.

    python scripts/fit_expert_portfolio.py \
      --input runs/source-tx-observations.jsonl.transactions.jsonl \
      --qualification-cards artifacts/qualification/merit-expert-qualification-v3.json \
      --config configs/merit_tx.yaml \
      --output artifacts/qualification/merit-expert-portfolio-v1.json

This is a source-only subset-selection policy, not a target-trained router. It
makes expert removal auditable rather than treating ablations as a post-hoc
paper result.

## Differential evidence effect

MERIT-Tx prefers expert-native verification signals over a shared VLM receiver residual whenever the expert exposes a meaningful native score.

For candidate claim c and immutable incumbent claim c0:

    m_real  = score_e(I, c)       - score_e(I, c0)
    m_knock = score_e(I_knock, c) - score_e(I_knock, c0)
    D_e     = m_real - median_k(m_knock,k)

The implementation uses multiple deterministic matched wrong-patient controls (four by default), not a single arbitrary control. Controls are selected without labels from the same modality/task and, when available, the same answer-blind question type but a different patient/study group. The median control margin reduces sensitivity to one atypical wrong-patient case while subtracting nonspecific expert/context preference.

Examples:

- CONCH/PLIP: candidate-vs-incumbent image-text similarity margin.
- XRV: strict native-label claim scoring only when the proposition uniquely names an XRV finding. Raw sigmoid outputs are converted to symmetric log-odds comparison scores; unsupported diagnoses fail closed rather than being guessed.
- segmentation/localization: candidate-specific spatial support relative to the same operator on a matched control.

A positive differential effect is patient-specific support for the candidate and is usable only with support/commit authority. A negative differential effect is patient-specific support for retaining the incumbent and is usable only with separately qualified veto authority. Positive support reliability is deliberately **not** reused as proof that negative vetoes are safe. Zero differential effect carries no proof in either direction.

## Adaptive BARD is a proposal mechanism, not commit proof

Adaptive BARD remains useful because it can move the frozen Generalist toward
specialist-informed candidate answers without training. In MERIT-Tx v3 that
movement is deliberately treated as **proposal generation**:

    b = frozen Generalist(I, q)
    c = Adaptive-BARD(I, q, expert evidence)
    T = atomic transactions compiled from c versus b
    decision(T) = independent native verification + source qualification

BARD residual/logit movement is not reused as transaction proof. All experts
that may have influenced the frozen BARD candidate must be declared through
`--proposer-expert-ids`; their fault groups are excluded from the independent
support set for the resulting transactions. This prevents an expert from
changing the candidate and then validating the same change through a second
interface.

For example, a source canary whose Adaptive BARD candidate may use CheXagent,
XRV and BiomedCLIP should declare the full frozen provenance set:

    python scripts/run_merit_tx_source_canary.py \
      --manifest /path/to/source-manifest.jsonl \
      --baseline /path/to/source-generalist.json \
      --candidate /path/to/source-adaptive-bard.json \
      --candidate-name adaptive-bard-v1 \
      --proposer-expert-ids chexagent_description cxr_findings biomedclip_claim_verifier \
      --qualification-cards artifacts/qualification/merit-expert-qualification-v3.json \
      --portfolio-policy artifacts/qualification/merit-expert-portfolio-v1.json \
      --config configs/merit_tx.yaml \
      --output runs/merit-tx-source-canary

The provenance declaration is conservative: if routing differs per sample, the
union of all experts that could influence the frozen candidate may be supplied.
A future per-sample provenance manifest can tighten this without changing the
transaction rule.

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
| BiomedCLIP dynamic claim verifier | direct visual verifier | CXR/X-ray/CT/MRI/pathology | source-qualified |
| MedSigLIP (gated optional) | direct visual verifier | CXR/CT/MRI/pathology/dermatology/fundus | source-qualified |
| MedImageInsight (candidate) | direct visual verifier | X-ray/CT/MRI/dermatology/OCT/fundus/ultrasound/pathology | source-qualified after adapter + source qualification |

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

This downloads vinid/plip and wanglab/medsam-vit-base. CONCH, MAIRA-2, and MedSigLIP are gated. The helper only prints commands for gated assets; it never bypasses upstream access terms. MedSigLIP is included because its official frozen image-text interface directly supports zero-shot classification and semantic retrieval across several modalities that were previously uncovered, but it still has zero commit authority until its exact source cell qualifies.

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

This branch now implements:

- immutable transaction representation and exact fallback;
- VQA decontextualization and optional RadGraph-XL report claimization;
- conservative candidate-to-transaction compilation shared by VQA and reports;
- report patch atomicity: all clinical claims carried by the same sentence-level patch must be approved, otherwise the incumbent sentence is preserved;
- patch-safe report compilation: mixed ADD+REPLACE sentences, implicit sibling deletion, and cross-incumbent-sentence merges fail closed; pure ADD sentences are decomposed into atomic proposition patches;
- literature-grounded role cards and capability/fault-group diversity selection;
- source-only **action-conditional** expert qualification with utility LCB, harm UCB, and directional-specificity LCB;
- signed multi-control real-vs-knockoff differential margins, including qualified contradiction vetoes;
- patient/study-level qualification aggregation for reports, preventing pseudo-replication from many claims in one report;
- proposer/validator separation;
- claim-specific CONCH and PLIP pathology verification;
- strict native XRV CXR finding verification;
- MedSAM prerequisite-aware spatial registration;
- answer-blind expert-pool and coverage audits;
- source qualification observation generation;
- report ADD verification through an explicit presence/absence counterfactual; uncertain ADD claims fail closed and omission never implies DELETE;
- verifier-specific call budgeting so non-verifying roles cannot consume independent-verifier slots;
- an end-to-end source MERIT-Tx canary runner;
- asset download/audit helpers.

It deliberately does not claim a new benchmark improvement yet.

### Frozen execution order

Do not use target/test results to decide which experts survive. The source workflow is:

1. **Freeze one proposal mechanism** before reading source references.
2. Partition source/development patients/studies into three disjoint sets:
   source-Q for per-expert qualification, source-P for portfolio selection, and
   source-C for the fresh canary. The code rejects group overlap.
3. Build native source-Q observations:

       python scripts/build_merit_tx_source_observations.py \
         --manifest /path/to/source-manifest.jsonl \
         --baseline /path/to/source-generalist.json \
         --candidate proposal=/path/to/frozen-source-candidate.json \
         --references /path/to/source-references.json \
         --config configs/merit_tx.yaml \
         --output runs/source-tx-observations.jsonl

   References are joined only after the candidate outputs have been frozen. For reports, one patient/study contributes at most one conservative observation per expert qualification cell.

4. Freeze v3 qualification cards from source-Q:

       python scripts/fit_expert_qualification.py \
         --input runs/source-p-tx-observations.jsonl.transactions.jsonl \
         --output artifacts/qualification/merit-expert-qualification-v3.json

5. Build a separate transaction-level observation file on source-P with the
   same frozen proposal mechanism, then freeze the interaction-aware expert
   portfolio. This is where harmful redundancy and "remove expert B" effects
   are measured before target evaluation. The fitter rejects any source-P
   patient/study also seen in source-Q:

       python scripts/fit_expert_portfolio.py \
         --input runs/source-tx-observations.jsonl.transactions.jsonl \
         --qualification-cards artifacts/qualification/merit-expert-qualification-v3.json \
         --config configs/merit_tx.yaml \
         --output artifacts/qualification/merit-expert-portfolio-v1.json

6. Audit effective commit-authorized coverage with both frozen policies:

       python scripts/audit_transactional_expert_pool.py \
         --manifest /path/to/source-manifest.jsonl \
         --config configs/merit_tx.yaml \
         --qualification-cards artifacts/qualification/merit-expert-qualification-v3.json \
         --portfolio-policy artifacts/qualification/merit-expert-portfolio-v1.json \
         --output runs/merit-tx-expert-pool-audit.json

7. Run the fresh source-C canary. The executable verifies source-C has no
   patient/study overlap with source-Q or source-P. The transaction decisions
   do not open references;
   optional source scoring occurs only after outputs are frozen:

       python scripts/run_merit_tx_source_canary.py \
         --manifest /path/to/source-manifest.jsonl \
         --baseline /path/to/source-generalist.json \
         --candidate /path/to/frozen-source-candidate.json \
         --candidate-name proposal-v1 \
         --qualification-cards artifacts/qualification/merit-expert-qualification-v3.json \
         --portfolio-policy artifacts/qualification/merit-expert-portfolio-v1.json \
         --config configs/merit_tx.yaml \
         --references /path/to/source-references.json \
         --output runs/merit-tx-source-canary

8. Only if the fresh canary shows useful nonzero commit coverage with bounded
   harm should the exact frozen policy be evaluated once on untouched
   target/test data.

Candidate-oracle and first-divergence analysis of already-completed target experiments remains diagnostic only; it cannot choose experts, qualification thresholds, or transaction rules.

## References

- Savage et al., MedRAX, ICML 2025.
- Nath et al., VILA-M3, CVPR 2025.
- DnR: Draft and Refine with Visual Experts, CVPR 2026 Highlight.
- Ma et al., Segment Anything in Medical Images (MedSAM), Nature Communications 2024.
- Lu et al., CONCH, Nature Medicine 2024.
- Huang et al., PLIP/OpenPath, Nature Medicine 2023.
- Jain et al., RadGraph, NeurIPS Datasets & Benchmarks 2021.
- Delbrouck et al., RadGraph-XL, ACL Findings 2024.
- Min et al., FActScore, EMNLP 2023.
- Gao et al., RARR, ACL 2023.
- Ostmeier et al., GREEN, EMNLP Findings 2024.
- CLEAR, EMNLP Findings 2025.
- Xiang et al., MUSK, Nature 2025.
- EyeCLIP, npj Digital Medicine 2025.
- EchoCLIP, Nature Medicine 2024.
- Jiao et al., USFM, Medical Image Analysis 2024.
- Blankemeier et al., Merlin, Nature 2026.
- He et al., VISTA3D, CVPR 2025.
- Codella et al., MedImageInsight, ML4H 2024 / Microsoft Research.
