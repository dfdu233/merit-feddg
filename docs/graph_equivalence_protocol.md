# Graph equivalence mechanism pilot — frozen before execution

Experiment: `graph-equivalence-dev-20260916-v1`.
Base: `adeda00589ba67d97adc634bf8f18deab20b9ac2`.
Branch: `experiments/graph-equivalence-20260916`.

Vector: Does lossless relation reification change source-fact retrieval beyond ordinary canonicalization and representation-dependent budgets?

This is a synthetic DEVELOPMENT mechanism pilot, not a benchmark, MedRAX reproduction, or evaluation of HippoRAG/GeAR implementations. No LLM or GPU. All parameters below are fixed before examining outputs.

- Generate 64 independent seeded directed graph fixtures. Each contains a two-relation query with a unique answer, distractor relations, and a directed entity ring (no dangling vertices). Store fact ID, source ID, timestamp, and polarity. No clinical semantics or real patient data.
- Compare direct edges, all edges reified, and a deterministic 50% subset reified. A reified fact is two role edges through a fresh event node. Fact metadata must survive exactly. All vertices/edges are accessible; no truncation during construction.
- Independently check exact fact round-trip and two-relation query execution on each encoded graph. A wrong-tail intervention must change the query answer; a lost qualifier must fail the round-trip check. Query answers are offline checks only and not passed to generic retrieval.
- Generic retrieval operators: (1) directed personalized PageRank, continuation 0.85, residual tolerance 1e-12; rank source facts by summed endpoint mass conditional on entity vertices; (2) outgoing reachability under budget two. These are transparent operator probes, not official system reproductions.
- PageRank returns six original source facts in every schema. Report Jaccard distance, support recall, query-support coverage, and conditional entity distribution L1. Source fact IDs provide deterministic schema-independent tie breaking.
- Reachability has both raw edge budget two (a deliberately confounded control) and original-fact budget two (each traversal of a represented relation costs one). Compare output source-fact sets; report their sizes rather than imply an equal-size retrieval comparison.
- Strong simple baseline: decode schema to the canonical relation table and run the identical operator. Also include an exact two-relation table query: this baseline receives query relation structure, so its recall is an upper reference for this narrow structured setting, not a fair natural-language retriever comparison.
- Negative controls: vertex renaming plus edge-order permutation, with seed mapping; identical canonical fact rendering for generation inputs. There is no generation experiment.
- H1a: raw graph operators can change under lossless reification. H1b: the measured failure survives cheap exact canonicalization. H0 for scientific escalation: any change is eliminated by canonicalization or source-fact budget accounting.
- Stop immediately if semantic checks fail. If canonicalization eliminates differences, do not escalate this transformation family to GPU, dense retrieval, real corpora, or claim a new method. Report construction-specific limits: known invertible schemas make canonicalization available by design.
- Report paired descriptive counts/means only. Fixtures are not population samples; no biomedical or corpus generalization, significance claim, or test-set inference.

Next step is conditional on this stop rule, not GPU availability. Prior literature audit: `/home/dbw/research-notes/medrax-graphs-20260916/RESEARCH.md`.
