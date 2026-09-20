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

## 4. Real knowledge base

The repository now builds an auditable KB from the real PubMed annual baseline.
As of 2026, NLM states that the 2026 production baseline contains
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
    wget -c https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/pubmed26n0001.xml.gz \
      -P artifacts/knowledge/pubmed-2026-baseline

Do not call a one-file pilot the full PubMed corpus.

### Build a 100k-document pilot KB

    python scripts/build_medcpt_kb.py \
      --pubmed-dir artifacts/knowledge/pubmed-2026-baseline \
      --article-encoder artifacts/models/ncbi--MedCPT-Article-Encoder \
      --output artifacts/knowledge/medcpt-pubmed \
      --limit 100000

The output contains:

- `documents.sqlite3`: PMID/title/abstract/source/year/provenance;
- sharded 768-D article embeddings;
- row-ID mappings;
- `manifest.json` with hashes and a KB fingerprint.

The generic JSONL builder explicitly rejects benchmark answer/reference/label
fields.

### Faster official MedCPT PubMed import

The MedCPT project also publishes precomputed PubMed Article-Encoder embeddings,
PMID lists and compact article content. This is the preferred way to create a
larger retrieval canary without recomputing millions of Article Encoder
forwards.

For the official example chunk 36:

    python scripts/prepare_medcpt_assets.py --print-commands

or download the three files under
`https://ftp.ncbi.nlm.nih.gov/pub/lu/MedCPT/pubmed_embeddings/` and import:

    python scripts/import_medcpt_pubmed.py \
      --source artifacts/knowledge/medcpt-precomputed-pubmed \
      --chunks 36 \
      --output artifacts/knowledge/medcpt-pubmed

The importer preserves PMID/title/date/abstract metadata, hashes all source
assets, skips records without abstracts, and never accepts benchmark answers.

### Scale after the pilot

Exact memory-mapped search is intentionally dependency-light and appropriate for
a small pilot. A large PubMed KB should use the optional FAISS index:

    python scripts/build_medcpt_kb.py \
      --pubmed-dir artifacts/knowledge/pubmed-2026-baseline \
      --article-encoder artifacts/models/ncbi--MedCPT-Article-Encoder \
      --output artifacts/knowledge/medcpt-pubmed-full \
      --index-backend hnsw \
      --hnsw-m 32

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

## 5. Efficiency and execution order

The reference replay decoder remains the scientific baseline. Efficiency changes
must preserve the same expert outputs, branch contexts and committed-token
semantics.

### 5.1 Shared frozen vision features

All isolated BARD branches use the same original image. The frozen CLIP vision
tower is executed once and its raw patch features are shared across branches.
Each branch still executes its own `mm_projector`; branch-local parameter-free
spatial hooks therefore remain active. Cache sharing fails closed unless the
processed image tensor, dtype/device and image sizes are identical.

### 5.2 Matched-arm score sharing

`isolated_mean`, `isolated_geomedian` and `bard` are evaluated together.
Receiver scores are reused only while two arms have exactly the same committed
token prefix. Once their prefixes differ, their receiver computation separates.

### 5.3 Expert-major cache preparation

The recommended memory-safe execution is two-phase. Specialists are run one
model at a time, cached with full provenance, released, and only then is
LLaVA-Med loaded for BARD decoding.

Before any expert inference, inspect scheduled coverage only:

    python scripts/prepare_bard_expert_cache.py \
      --manifest /absolute/path/to/source-manifest.jsonl \
      --config configs/matched_bard.yaml \
      --output runs/bard-expert-cache \
      --coverage-only

This writes `coverage.json` with per-case fault groups and the n=0/1/2/3+
distribution without loading references, answers, or expert outputs. If most
cases are still n<=1, stop before spending GPU time.

If coverage is usable, build the native expert cache:

    python scripts/prepare_bard_expert_cache.py \
      --manifest /absolute/path/to/source-manifest.jsonl \
      --config configs/matched_bard.yaml \
      --output runs/bard-expert-cache

If a compatible routing JSON already exists, pass `--routing-json` so this
stage does not need to load the Generalist for routing.

Then run the reference Adaptive BARD experiment:

    python -m merit_feddg.matched_evaluation \
      --protocol bard \
      --config configs/matched_bard.yaml \
      --manifest /absolute/path/to/source-manifest.jsonl \
      --output runs/adaptive-bard \
      --reuse-expert-run /absolute/path/to/completed-cache-run

Compatible Generalist baselines may additionally be supplied through
`--reuse-generalist`. Donor reuse requires the exact manifest/routes, request
keys, current expert provenance, and the provenance recorded by the donor run;
an in-place model or KB replacement therefore invalidates the donor.

### 5.4 Persistent-KV fast path is implemented but opt-in

A single committed BARD trajectory has an O(K*T)-style persistent-KV decoder.
It is not silently used by `matched_evaluation`.

Before activation, the Generalist and every actual fault-group branch must be
compared against production replay on a fixed prefix. Every tested step must
preserve the greedy argmax. Any branch mismatch disables the fast path and the
reference replay decoder remains authoritative.

### 5.5 Large-KB acceleration

Small PubMed pilots use exact memory-mapped inner-product search. Large corpora
can use optional FAISS Flat or HNSW indexes. This changes retrieval engineering,
not the frozen MedCPT weights or BARD decision rule.

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
