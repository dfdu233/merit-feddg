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
    wget -c https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/pubmed26n0001.xml.gz       -P artifacts/knowledge/pubmed-2026-baseline

Do not call a one-file pilot the full PubMed corpus.

### Build a 100k-document pilot KB

    python scripts/build_medcpt_kb.py       --pubmed-dir artifacts/knowledge/pubmed-2026-baseline       --article-encoder artifacts/models/ncbi--MedCPT-Article-Encoder       --output artifacts/knowledge/medcpt-pubmed       --limit 100000

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

    python scripts/build_medcpt_kb.py       --pubmed-dir artifacts/knowledge/pubmed-2026-baseline       --article-encoder artifacts/models/ncbi--MedCPT-Article-Encoder       --output artifacts/knowledge/medcpt-pubmed-full       --index-backend hnsw       --hnsw-m 32

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

### Implemented now: matched-arm score sharing

`isolated_mean`, `isolated_geomedian` and `bard` now run in a bundle.
Receiver scores are reused only while two methods have exactly the same
committed token prefix. Once their prefixes differ they are evaluated
independently.

This removes duplicate experimental computation without changing any arm.

### Implemented already: native expert cache

Compatible specialist outputs and Generalist baselines can be reused via the
existing matched-evaluation cache/provenance checks. The MedCPT KB manifest and
reranker assets are part of model provenance.

### Deferred until efficacy signal: persistent KV

The reference `next_scores(prefix)` deliberately replays production generation
because previous FP16 tests showed that naive full-prefill alternatives can
change near-tied tokens. Persistent per-branch KV state could reduce decoding
from approximately O(K*T^2) to O(K*T), but it must first pass exact/token-level
parity canaries.

Do not make that engineering change before Adaptive BARD shows useful clean and
fault-injection behavior.

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
