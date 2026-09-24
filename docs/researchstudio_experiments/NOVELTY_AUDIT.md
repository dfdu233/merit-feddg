# Preliminary novelty audit

This is a scoped collision check following the ResearchStudio/Scoop-Check decomposition. It is **not** an exhaustive formal novelty certification.

## Proposed work axes

- **Problem framing:** training-free collaboration between a frozen medical VLM and heterogeneous specialist models for medical VQA.
- **Core mechanism:** task-native expert artifacts condition isolated branches of the same receiver; branch effects become comparable in the receiver vocabulary under a shared prefix; a source-aware decoder aggregates/commits interventions.
- **Key insight:** the common space need not be an expert representation or cross-model vocabulary; the receiver itself can serve as the measurement operator.
- **Application domain:** multimodal medical question answering across multiple specialist types.

## Closest works

1. **Visual Evidence Prompting (ACL 2025):** matches the use of specialist visual models and prompting, but not isolated receiver-side intervention measurement.
2. **CCD (Findings ACL 2026):** matches training-free medical expert-guided token decoding, but is centered on structured radiology signals rather than a heterogeneous task-native interface across specialist output types.
3. **RobustRAG (SaTML 2026):** matches isolate-then-aggregate, but its atoms are retrieved text contexts rather than heterogeneous specialist artifacts.
4. **Inference-Time Consensus (arXiv 2026):** matches source-wise next-token consensus and base-relative/quorum rules, but assumes source-specific reference models already live in one model output space. This is the most important recent collision for Adaptive BARD and must be included in related work and experiments.

## Defensible delta

The most defensible paper-level novelty is the **receiver-mediated heterogeneous interface**, not geometric-median aggregation or quorum logic alone.

The paper should avoid presenting isolation, geometric medians, base-relative consensus, or quorum logic as standalone inventions.
