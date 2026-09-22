# MERIT-Tx v4: Absolute-Specific Evidence Certificates

## Why v4 exists

MERIT-Tx v3 fixed several structural failures of the earlier token-level
collaboration path: the Generalist is immutable, edits are atomic transactions,
verifiers are source-qualified, and expert subsets are selected on a disjoint
source portfolio split.

The latest experiments show that the next bottleneck is **not** the portfolio
optimizer. It appears earlier, in the semantics of a verifier action.

### Source-Q: no pathology verifier qualified

On the frozen v3 source-Q study, neither receiver produced a
commit-authorized pathology verifier.

| Receiver | Verifier | Utility LCB | Support precision LCB |
|---|---|---:|---:|
| LLaVA-Med | BiomedCLIP | -0.01937 | 0.19824 |
| LLaVA-Med | CONCH | -0.04060 | 0.23098 |
| LLaVA-Med | PLIP | -0.01167 | 0.31951 |
| HuatuoGPT | BiomedCLIP | -0.32263 | 0.08179 |
| HuatuoGPT | CONCH | -0.11504 | 0.25544 |
| HuatuoGPT | PLIP | -0.07433 | 0.23072 |

Therefore no source-P portfolio should be interpreted as the main failure:
there was no qualified verifier authority to compose.

### Full PathVQA exploratory diagnostic

On 6,719 PathVQA TEST examples, used only as post-hoc diagnosis and not for
policy fitting:

| Arm | Mixed score | Corrected | Harmed |
|---|---:|---:|---:|
| Generalist | 38.6567 | - | - |
| raw BARD candidate | 35.8285 | 465 | 668 |
| BiomedCLIP filter | 38.5612 | 163 | 164 |
| CONCH filter | 38.1974 | 166 | 191 |
| PLIP filter | 38.8949 | 79 | 55 |
| joint filter | 38.8118 | 91 | 77 |

PLIP and the joint rule recovered most of the BARD damage, but their bootstrap
gain intervals included zero. CONCH remained harmful. More importantly, the
same filtering rule was not uniformly desirable on other datasets.

### Cross-dataset asymmetry

- OmniMedVQA 17,673: Generalist 68.1039, BARD 68.2736, joint filter 68.2057.
- SLAKE 2,094: Generalist 54.5504, BARD 57.1073, while the current verifier
  filter reverted much of the useful candidate movement.
- VQA-RAD 451: Generalist 63.8910, BARD 63.0648.

Thus the proposal mechanism itself is cell-dependent: sometimes harmful,
sometimes useful. A universal "always trust BARD" or "always reject BARD unless
an expert approves" rule is not justified by these exploratory target results.
Target data must not choose that mode.

## The v3 evidence-definition bug

For expert e, candidate claim c, and immutable incumbent b, v3 used

    m_real = score_e(I, c) - score_e(I, b)
    m_k    = score_e(I_k, c) - score_e(I_k, b)
    D_e    = m_real - median_k(m_k)

and treated D_e > 0 as candidate support.

This is only a **relative uplift** statement. It does not imply that the
current image supports the candidate.

The full OmniMedVQA exploratory trace contains a concrete counterexample:

    real_margin      = -0.88933
    control_median   = -1.04280
    D                = +0.15347

The current image still preferred the incumbent, but because wrong-patient
controls preferred it even more strongly, v3 called the candidate "supported."

This is a definition error, not a threshold-tuning problem.

## v4 evidence certificate

v4 separates two obligations.

### 1. Absolute support

The current patient image itself must prefer the proposed claim:

    m_real > 0

for support, or

    m_real < 0

for a veto.

### 2. Counterfactual specificity

The real patient margin must be more extreme than every matched wrong-patient
control:

    support: m_real > max_k m_k
    veto:    m_real < min_k m_k

Otherwise the expert abstains.

The combined training-free certificate is therefore

    support(c > b)
      iff m_real > 0
      and m_real > max_k m_k

    veto(c > b)
      iff m_real < 0
      and m_real < min_k m_k

    otherwise abstain

With K exchangeable matched controls, strict extremeness also has an auditable
rank bound of 1/(K+1). MERIT records this rank quantity but does not tune an
acceptance threshold on target data.

The implementation is in:

- merit_feddg/merit_tx.py::differential_margin_controls
- merit_feddg/merit_tx.py::TransactionEvidence
- merit_feddg/transactional_runtime.py::verify_transaction

The exact OmniMedVQA failure is frozen as a regression test in
tests/test_merit_tx_closed_loop.py.

## Diagnostic replay on OmniMedVQA

The 17,673-case OmniMedVQA exploratory result stores all native margins for 520
candidate-changed cases. Replaying the already-frozen v4 rule without model
inference gives:

| Quantity | v3 | v4 |
|---|---:|---:|
| verifier effects | 191 | 191 |
| support actions | 95 | 32 |
| veto actions | 96 | 38 |
| joint accepted candidate changes | 42 | 24 |

Among the v3 actions:

- 32 support actions had real_margin <= 0;
- 27 veto actions had real_margin >= 0.

These are exactly the relative-only action class that v4 forbids.

The replay is mechanism diagnosis only. No target correctness labels are used
to set the rule or a threshold.

A reusable zero-inference source replay is implemented in:

    python scripts/audit_evidence_certificate_upgrade.py \
      --input runs/source-Q.jsonl.transactions.jsonl \
      --output runs/source-Q-v4-certificate-replay.json

It reports removed support/veto actions and their already-frozen source utility,
including the explicit relative_only_false_support_shape count.

## Source utility: stop using token-F1 as the default VQA qualification label

The v3 source run also exposed a separate confound: token-F1 penalizes a correct
answer merely for adding explanatory words. This can make a verifier appear
harmful because of output length rather than task correctness.

v4 therefore adds mixed_vqa_score:

- binary/closed VQA: leading Yes/No correctness;
- open VQA: reference answer-token recall;
- report generation: the existing atomic report-claim utility.

The default source observation metric is now mixed_vqa.

Implementation:

- merit_feddg/contribution.py::mixed_vqa_score
- scripts/build_merit_tx_source_observations.py --metric mixed_vqa

Generic token-F1 remains available only as an explicit diagnostic.

## Matched controls

The native verifier already scores the same candidate and incumbent propositions
on the real image and controls. The VQA question type is therefore not an input
nuisance for a generic image-text verifier.

v3 unnecessarily required same-question-type controls by default, causing
avoidable "need 4, found 3" failures. v4 defaults to:

    same modality
    same task
    different patient/study group

An expert may explicitly declare extra answer-blind knockoff_match_fields when
its native interface truly requires them.

This changes control availability, not the certificate definition.

## What v4 deliberately does not do

v4 does **not** use the exploratory target results to decide whether BARD or the
Generalist should be the default output.

The results show that this is a real next question:

- PathVQA / VQA-RAD: proposal movement is net harmful;
- OmniMedVQA / SLAKE: proposal movement can be net beneficial.

The next source-only experiment should therefore estimate a
**proposal-utility card** on source-Q and validate it on source-C before
introducing a bidirectional policy:

    baseline-default cell:
        candidate needs a support certificate

    source-qualified proposal-default cell:
        candidate is provisional and a qualified veto can roll it back

That policy is not enabled in v4. Enabling it now would use target observations
to justify a new default and would contaminate the evaluation.

## Required next experiment

1. Freeze the existing proposal mechanism before reading source references.
2. Rebuild source-Q observations with --metric mixed_vqa.
3. Run the zero-inference v3-to-v4 certificate replay first.
4. Refit merit-expert-qualification-v4.
5. Fit the interaction-aware portfolio on group/image-disjoint source-P.
6. Run a fresh source-C canary.
7. Stop unless source-C shows nonzero useful coverage with bounded harm.
8. Only then run the exact frozen v4 policy on untouched target/test.

The v4 change is a mechanism correction. No new benchmark improvement is
claimed until this source protocol succeeds.
