# Graph equivalence pilot: stop at the simple-baseline boundary

Experiment `graph-equivalence-dev-20260916-v1`, 2026-09-16.
Protocol commit `1653769`; executed pilot code commit `f0b7c8e`.
Independent branch `experiments/graph-equivalence-20260916`.

## Evidence update

64 synthetic DEVELOPMENT graph fixtures, each encoded as direct relations, full reification, and partial reification: 192 schema cases. All fact round-trips, source/qualifier preservation checks, and independent two-relation query checks passed. Wrong-tail positive controls changed the answer; qualifier-loss controls were detected. No real corpus, no LLM generation, no MedRAX benchmark, and no GPU work was performed.

The queries concern relation composition only, not interpretation of clinical polarity. Qualifiers are retained and checked as data; this is not an evaluation of medical reasoning.

| Metric | Direct | All relations reified | Approximately half reified |
|---|---:|---:|---:|
| Fixtures | 64 | 64 | 64 |
| PPR Top-6 fact set changed vs direct | 0/64 | 27/64 | 20/64 |
| Mean Top-6 Jaccard distance | 0 | 0.156622 | 0.105283 |
| Mean conditional entity PPR L1 distance | 0 | 0.305274 | 0.179634 |
| Mean support-fact recall | 0.656250 | 0.625000 | 0.640625 |
| Both query-support facts retrieved | 22/64 | 16/64 | 18/64 |
| PPR Top-6 changes after canonicalization | 0/64 | 0/64 | 0/64 |
| Raw two-edge reachability mean Jaccard distance | 0 | 0.684602 | 0.445141 |
| Original-fact-budget reachability distance | 0 | 0 | 0 |

Renaming and edge-order controls preserved selected sets; maximum numerical error was below 1e-10. Canonical evidence rendering was identical. The independent structured query obtained the expected answer in all schema cases. This exact-query reference receives relation structure, whereas generic retrieval receives only the seed entity: it is not a fair natural-language retrieval accuracy comparison.

Raw pilot runtime was about 0.71 seconds in this environment. This is not a latency benchmark. The unchanged canonical results are expected by construction: a known invertible mapping recovers exactly the same facts and graph. They establish a stopping boundary for this particular proposal, not a statistical refutation of every possible representation problem.

## Affinity-map location and Bit Flip status

This operator probe addresses source-fact access, the objective shared by associative graph retrieval methods. It is not a reproduction of HippoRAG or GeAR and does not establish their failure rates.

H1a, raw operator sensitivity, is supported on these fixtures. H1b, an effect beyond cheap exact canonicalization, is not supported here. The proposed Bit Flip is weakened; ordinary graph subdivision and budget effects explain the observed phenomenon. Do not advertise the Top-6 change counts as a new community finding.

## Mechanism: full subdivision changes the effective continuation probability

Let P be the original row-stochastic entity transition matrix, s the restart distribution over entities, and alpha the PageRank continuation probability. There are no dangling vertices in this pilot. In the fully reified graph, every original entity-to-entity transition now requires two steps. If e is the unnormalized stationary mass on entity vertices, eliminating event vertices gives:

`e = (1 - alpha) s + alpha^2 P^T e`.

The total entity mass is `1 / (1 + alpha)`. Conditioning on entities therefore gives:

`e_normalized = (1 - alpha^2) s + alpha^2 P^T e_normalized`.

Thus the fully reified graph at alpha=0.85 equals the original entity graph at continuation probability 0.7225, after conditioning on entities. This is an elementary consequence of the two-step construction, not a new theorem claim. A separate numerical check across all 64 fixtures found maximum absolute discrepancy `1.0347278589506459e-13`, below the fixed 1e-10 tolerance. This analytical check was added after observing the pilot and is explicitly post-hoc mechanism explanation, not a pre-registered result.

With partial reification, different transitions consume different numbers of graph steps. The raw two-edge budget has the same obvious problem: a two-fact query may require four representation edges. Charging original-fact steps restores reachability exactly. The raw-edge arm was deliberately included as a confounded control and must not be described as an equal-semantic-budget failure.

## Current Vector and Core / Periphery

The tested Vector was whether this lossless transformation creates source-fact retrieval changes beyond canonicalization and budget accounting. The answer for this controlled construction is no.

Core completed: invertible encoding, independent query check, PPR and reachability probes, simple normalization baseline, positive/negative controls, source-fact metrics. Periphery stayed frozen: experts, gate, actor training, medical datasets, GPU inference, and new agent components.

## Minimum experiment decision

The pre-registered stop rule has fired. No dense baseline or real-corpus/GPU escalation is justified for this transformation family after the exact simple baseline resolves the phenomenon. This is an early stopping result, not completion of the larger conditional real-data design in the literature report.

A stronger future hypothesis would first need a real evidence setting where canonicalization is unavailable or expensive for a principled reason. We cannot assume such a setting merely to rescue this direction. More complex encoders or more experts would not change the present result's scientific interpretation.

## Novelty collision and velocity

The original review already identified database representation invariance, graph serialization sensitivity, and schema-sensitive GraphRAG as close prior work. The present pilot narrows the gap further: the simplest demonstration is fully explained by basic operator mechanics. There is no basis for an ICLR novelty claim from this run.

What was learned: graph representation can alter a transparent retrieval operator, but the tested change does not survive the strongest simple baseline. This avoids spending GPU time establishing a downstream consequence of an already explained effect.

## Re-vector rules

- Current negative branch: stop building a new method around relation reification.
- Return to the affinity map before selecting another hypothesis. The existing reserve question is whether graph dependencies support evidence reuse across unseen same-case query combinations beyond exact caching and structured tables. Its task necessity and closest methods remain unverified; it is not yet a selected method or a new Bit Flip.
- Reopen representation research only if an independently motivated real setting defeats the simple canonicalization explanation while preserving a checkable information-equivalence contract.

## Artifacts and reproduction

- Frozen protocol: [graph_equivalence_protocol.md](graph_equivalence_protocol.md).
- Executed script: [graph_equivalence_pilot.py](../scripts/graph_equivalence_pilot.py).
- Local raw outputs: `runs/graph-equivalence-dev-20260916-v1/{fixtures,pairs,summary,analytic_check}.json`.
- Main pilot command: `python scripts/graph_equivalence_pilot.py` in a fresh checkout. The script refuses to overwrite an existing run directory.
- Literature report: `/home/dbw/research-notes/medrax-graphs-20260916/RESEARCH.md`.

No raw private data or model weights were used. Pilot output is synthetic and preserved locally; the compact numerical summary is included in this report. Historical branches and results are unchanged.
