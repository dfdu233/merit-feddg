# Minimal experiment plan

**Goal:** decide the paper's core mechanism before spending on another full benchmark sweep.

## Stage 0 — bind method to reported rows (no GPU)

Required first. For every MERIT row in the main table, record:

- full code SHA;
- decoder/protocol;
- resolved config or launch command;
- receiver checkpoint/revision;
- dataset manifest/split;
- prompt/generation contract;
- scorer version.

If the main rows were not produced by the method currently described in the paper, stop and repair the manuscript/result binding before making mechanism claims.

## Stage 1 — cheapest mechanism decomposition

Use one development/source benchmark with useful expert coverage. Reuse the **exact same native evidence cache and receiver checkpoint**.

Compare:

1. **generalist**
2. **joint_all:** all delivered evidence in one receiver context
3. **isolated_mean:** source-isolated branches, arithmetic aggregation, no bounded commit
4. **isolated_geomedian:** source-isolated branches, geometric median, no bounded commit
5. **bard:** same isolated branches + anchored support rule

### Primary questions

- **joint_all → isolated_geomedian:** does source isolation / receiver mediation help when evidence delivery is matched?
- **isolated_geomedian → bard:** does the commit rule reduce harmful revisions without deleting useful corrections?

Do not compare joint and isolated arms until delivered evidence identities are audited; context-budget differences otherwise confound isolation with evidence quantity.

### Primary statistics

Report:

- paired per-example score delta versus generalist;
- positive score-change mass;
- negative score-change mass;
- fraction of changed outputs;
- cluster-bootstrap intervals by patient when available, otherwise by image;
- delivered source-group count (n=0/1/2/3+).

### Early stopping

Stop before a GPU scale-up if:

- expert coverage is effectively zero;
- wrapper parity against the intended baseline fails;
- all isolated arms reproduce the generalist.

## Stage 2 — closest-prior baseline (reuse branch scores)

If Stage 1 shows a nontrivial signal, add a **base-relative/quorum consensus** decoder inspired by *Inference-Time Consensus for Mitigating Hidden Behaviors from LLM Fine-Tuning* (arXiv:2607.23394).

Use the **same receiver-branch distributions**. This is a decoding baseline, not a new expert run.

Compare its rescue-harm curve and intervention coverage with BARD. Do not tune quorum/fault parameters on the final test set.

## Stage 3 — smallest failure diagnosis

Run the same fixed setup on one failure-domain benchmark, with PathVQA as the current natural candidate, only after Stage 1 is frozen.

Classify the failure into:

- **missing source coverage:** most cases have (nle1);
- **ineffective evidence:** isolated arms do not beat the generalist;
- **harmful aggregation/commitment:** unprotected isolated aggregation and BARD differ materially in rescue/harm behavior.

Do not infer one of these causes from aggregate accuracy alone.

## Full evaluation trigger

Only after one configuration survives Stages 1–3 should the complete benchmark suite be rerun.

The full run is **confirmation**, not another tuning round.
