# Adaptive BARD + MedCPT Knowledge Expert

## Scope

This branch upgrades BARD without training a new gate, router, bridge, critic, or
retriever. Frozen specialist models remain plug-and-play. The changes address
four concrete problems observed in the first BARD prototype:

1. a hard four-expert requirement caused most cases to reproduce the Generalist;
2. segmentation experts lost most of their dense spatial information when forced
   through a text-only receiver interface;
3. non-CXR modalities have sparse specialist coverage;
4. exact same-prefix replay is expensive, especially when matched BARD ablations
   independently repeat the same receiver computation.

No target answer is consumed by generation, expert selection, retrieval, or
Byzantine aggregation.

## 1. Adaptive centralized robust aggregation

The first BARD prototype reused the distributed Byzantine-agreement condition
3f+1. MERIT has a trusted centralized aggregator, so the new threat model uses
the robust-aggregation requirement f < n/2.

For the default declared fault budget f=1:

- n=0: Generalist;
- n=1: Generalist. One fallible expert cannot be identified as right or wrong
  without another independent observation;
- n=2: unanimous non-forcing commit. Both isolated receiver branches must move
  the same candidate over the incumbent;
- n>=3: geometric-median receiver residual plus n-f supporting nodes and a
  positive conservative support margin.

The rule is a robustness property, not a medical correctness theorem.

### Fault groups

A BARD node is a declared failure-correlation group, not necessarily a tool
name. The optional expert field `fault_group` merges tools that should not get
independent votes. The current configuration puts XRV DenseNet findings and XRV
PSPNet anatomy in the same `torchxrayvision` group. MedCPT is a separate
knowledge-source group.

This is deliberately conservative: correlated errors across nominally distinct
models can still violate the bounded-fault assumption.

## 2. Native heterogeneous receiver branches

The common space is the receiver distribution, not the expert representation.

Each fault group gets its own branch at the exact committed prefix:

    native expert output
       -> expert-specific receiver interface
       -> p_e(next token | same image, question, prefix)
       -> centered receiver residual r_e

Text/generation/retrieval evidence uses the frozen semantic token interface.
Classification/segmentation/detection evidence additionally uses the existing
parameter-free spatial packet when it can be encoded. In particular, masks and
CAMs no longer need to be reduced to centroid/area text before the receiver sees
their spatial structure.

If a packet has no supported spatial record, the branch falls back to the
semantic interface and records that decision. There is no learned cross-model
projection.

## 3. PubMed knowledge expert

The optional `medcpt_pubmed` expert uses the released MedCPT architecture:

- Query Encoder for first-stage dense retrieval;
- Article Encoder for offline PubMed article embeddings;
- Cross Encoder for top-candidate reranking.

The returned item contains PubMed document IDs, titles, abstracts, years and
retrieval scores. Scores retain relevance semantics and are never interpreted
as probabilities that the patient has a disease.

A literature branch still receives the query medical image through the
Generalist. Retrieved literature is therefore external knowledge conditioned by
the receiver, not a replacement image or a patient-specific diagnosis.

### Why MedCPT

MedCPT was trained for biomedical information retrieval from large-scale PubMed
search logs and explicitly couples a first-stage bi-encoder retriever with a
second-stage cross-encoder reranker:

- https://github.com/ncbi/MedCPT
- https://pubmed.ncbi.nlm.nih.gov/37930897/

MedRAG (Findings ACL 2024) evaluates PubMed, StatPearls, textbooks and Wikipedia
with multiple retrievers and reports benefits from combining medical corpora and
retrievers:
https://aclanthology.org/2024.findings-acl.372/

MMed-RAG (ICLR 2025) motivates the opposite failure mode as well: retrieved
contexts can misalign a Med-VLM and can be wrong. We do not copy its trained
preference-alignment module; retrieval is instead treated as one fallible BARD
expert:
https://openreview.net/pdf?id=s5epFPdIW6

## 4. Real multi-source knowledge base

The default KB is now **MedCorp**, not a patient-case cache. It contains
answer-free biomedical literature with explicit source provenance. The built-in
sources are PubMed abstracts and StatPearls clinical chapters; additional
licensed/approved PMC, CPG, textbook, or institutional corpora can be supplied
through the same answer-free JSONL contract.

This follows the multi-corpus lesson from MedRAG (PubMed + StatPearls +
textbooks + Wikipedia) and RAG2 (PubMed + PMC + CPG + textbooks), while keeping
MERIT's first reproducible build limited to sources with explicit local
preparation paths.

As of 2026, NLM states that the 2026 PubMed production baseline contains
`pubmed26n0001` through `pubmed26n1334`.

Preparation is explicit; inference never silently downloads models or corpora.

### Audit/download models

    python scripts/prepare_medcpt_assets.py --print-commands

To download the three released MedCPT checkpoints through huggingface_hub:

    python scripts/prepare_medcpt_assets.py --download-models --print-commands

Expected local snapshots:

    artifacts/models/ncbi--MedCPT-Query-Encoder
    artifacts/models/ncbi--MedCPT-Article-Encoder
    artifacts/models/ncbi--MedCPT-Cross-Encoder

### Download a source-only pilot

For example, download the first real 2026 PubMed baseline file:

    mkdir -p artifacts/knowledge/pubmed-2026-baseline
    wget -c https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/pubmed26n0001.xml.gz       -P artifacts/knowledge/pubmed-2026-baseline

Do not call a one-file pilot the full PubMed corpus.

### Build a 100k-document pilot KB

    python scripts/build_medcpt_kb.py       --pubmed-dir artifacts/knowledge/pubmed-2026-baseline       --article-encoder artifacts/models/ncbi--MedCPT-Article-Encoder       --output artifacts/knowledge/medcpt-medcorp       --limit 100000

The output contains:

- `documents.sqlite3`: PMID/title/abstract/source/year/provenance;
- sharded 768-D article embeddings;
- row-ID mappings;
- `manifest.json` with hashes and a KB fingerprint.

The generic JSONL builder explicitly rejects benchmark answer/reference/label
fields.

### Scale after the pilot

Exact memory-mapped search is intentionally dependency-light and appropriate for
a small pilot. A large PubMed KB should use the optional FAISS index:

    python scripts/build_medcpt_kb.py       --pubmed-dir artifacts/knowledge/pubmed-2026-baseline       --article-encoder artifacts/models/ncbi--MedCPT-Article-Encoder       --output artifacts/knowledge/medcpt-medcorp-full       --index-backend hnsw       --hnsw-m 32

FAISS is optional and must be installed explicitly in the KB/retrieval
environment. Do not change the existing LLaVA-Med environment merely to build
the index.

For a true full 2026 baseline, download/process all 1334 baseline files. NLM
daily updates are a separate data-maintenance step and are not silently mixed
into a frozen experiment.

The official MedCPT repository also publishes precomputed PubMed embeddings.
Those are a good future full-scale shortcut, but this branch's builder keeps the
first experiment auditable by binding article text and embeddings generated by
the pinned local Article Encoder.

## 5. Efficiency without changing the algorithm

### Implemented now: source-aware retrieval budget

The large PubMed corpus must not drown out smaller clinical sources. The KB
stores a source ID beside every embedding. Exact scan and optional FAISS/HNSW
build source-specific candidate pools, split the fixed `candidate_k` budget
across sources, then apply the same MedCPT cross-encoder reranker. Final top-k
prefers one document from each source before filling remaining slots by
relevance. This is deterministic diversity, not a learned source prior.

### Implemented now: matched-arm score sharing

`isolated_mean`, `isolated_geomedian` and `bard` now run in a bundle.
Receiver scores are reused only while two methods have exactly the same
committed token prefix. Once their prefixes differ they are evaluated
independently.

This removes duplicate experimental computation without changing any arm.

### Implemented now: expert-major cache preparation

For heterogeneous experts with incompatible memory footprints, the recommended
execution is two-phase rather than keeping every model resident beside LLaVA-Med.

Phase A runs one specialist model across the frozen queue, writes the exact raw
native `CapabilityResult` cache, releases that model, then proceeds to the next
specialist:

    python scripts/prepare_bard_expert_cache.py \
      --manifest /absolute/path/to/manifest.jsonl \
      --config configs/matched_bard.yaml \
      --output runs/bard-expert-cache

If a compatible routing JSON already exists, pass `--routing-json` to avoid
loading the Generalist during this preparation phase.

Phase B runs BARD with the resulting donor directory:

    python -m merit_feddg.matched_evaluation \
      --protocol bard \
      --config configs/matched_bard.yaml \
      --manifest /absolute/path/to/manifest.jsonl \
      --output runs/adaptive-bard \
      --reuse-expert-run /absolute/path/to/completed-cache-run

The existing donor checks bind the full manifest, routes, expert configuration,
model provenance and exact request keys. A cache-only run does not load
references or benchmark answers.

This avoids requiring LLaVA-Med, CheXagent, BiomedParse and MedCPT to remain
simultaneously resident. Compatible Generalist baselines can still be reused
separately through `--reuse-generalist`.

### Implemented experimentally: parity-guarded persistent KV

The reference `next_scores(prefix)` remains the scientific reference because
previous FP16 tests showed that alternative execution paths can flip near-tied
tokens. `receiver_mode: auto` now creates a persistent-KV cursor for every
receiver branch and compares its normalized next-token log probabilities and
the resulting BARD commit decision against the replay implementation for a
fixed canary prefix.

If any branch exceeds the declared numerical tolerance, changes the selected
token, changes commit/abstain, or cannot construct a native cursor, the run
automatically falls back to replay and records the reason. Only a passing
canary enables the O(K*T) cursor path. Real-checkpoint parity is therefore an
experiment precondition, not assumed from CPU unit tests.

## 6. First experiment before scale

Use a development/source queue and report:

1. delivered fault-group count distribution (0/1/2/3/4+);
2. receiver-channel distribution: semantic vs semantic+native-spatial;
3. Generalist, joint-all, isolated mean, isolated geometric median, Adaptive BARD;
4. BARD n=1 fallback, n=2 unanimous and n>=3 robust-commit coverage;
5. original Generalist harms/rescues and newly introduced harms;
6. one-node synthetic corruption stress test;
7. MedCPT retrieval documents/PMIDs and whether the retrieved context actually
   changes the receiver;
8. wall time, receiver score calls, native tool calls, and peak VRAM.

Do not tune the fault budget, retrieval top-k or consensus rule on the same
target benchmark used for the final claim.

A successful pilot requires useful nonzero collaboration coverage. Returning the
Generalist on almost every case is not a positive result.
