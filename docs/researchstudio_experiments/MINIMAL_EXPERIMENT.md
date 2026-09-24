# Minimal experiment plan

**Goal:** decide the paper's core mechanism before spending on another full benchmark sweep.

This is an **ablation specification**, not a claim that the arms have been run. Keep the existing main-table raw results; a new arm never retroactively changes them. Freeze one receiver/checkpoint, native cache, image-question manifest, prompt, generation length, scorer, and code/config revision before the first paired comparison. Report HuatuoGPT-Vision-7B and LLaVA-Med separately; never average across receivers to hide a failure.

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

Create one immutable run ledger per reported cell: dataset and exact TEST denominator; output directory and per-ID provenance; source SHA plus a hash of any uncommitted code diff; model/processor weight revision; expert-cache identity; routing/schedule identity; resolved decoder parameters (including `max_new_tokens`, EOS and empty-answer policy); prompt template; scorer file hash and metric version. A Git commit alone is insufficient when the executed worktree was dirty. Mark missing provenance **unverified**, not equivalent. Reuse prior outputs only if the receiver, input/prompt/image SHA, native evidence, methods, and generation protocol match; otherwise keep them as a separately labelled historical comparison.

## Stage 1 — cheapest mechanism decomposition

Use one development/source benchmark with useful expert coverage. Reuse the **exact same native evidence cache and receiver checkpoint**.

Compare:

1. **generalist**
2. **joint_all:** all delivered evidence in one receiver context
3. **isolated_mean:** source-isolated branches, arithmetic aggregation, no bounded commit
4. **isolated_geomedian:** source-isolated branches, geometric median, no bounded commit
5. **bard:** same isolated branches + anchored support rule

The causal contrasts are deliberately narrower than the five-arm ranking:

| Contrast | Changed component | What it can support |
|---|---|---|
| generalist → joint_all | delivered evidence enters a single receiver context | evidence utility, **not** source isolation |
| joint_all → isolated_mean | separate receiver contexts plus arithmetic merge | the isolation-and-merge package; it does **not** identify isolation independently of merging |
| isolated_mean → isolated_geomedian | aggregation rule only | robustness of the center rule |
| isolated_geomedian → bard | anchored support/commit only | whether the commit rule improves the rescue–harm trade-off |

All isolated arms must use the **same source selection, native artifacts, rendering rule and receiver configuration**. At a *shared* committed prefix their branch distributions must agree; after free-generation answers diverge, the prefixes and hence the branch scores legitimately differ. Do not claim that separate free generations reuse one identical logit trace. Keep every available source group, its exact native output hash, deterministic rendered text, group order, and delivered/truncated status. `joint_all` must receive the same selected evidence, not a richer or shorter retrieval; compare context-token counts and disclose unavoidable budget differences. There is no direct test of "receiver mediation versus no receiver" in this chain: all evidence arms use the receiver. The defensible claim is that the frozen receiver provides a usable common comparison space, supported by heterogeneous-source strata and the paired contrasts above.

### Primary questions

- **joint_all → isolated_geomedian:** does source isolation / receiver mediation help when evidence delivery is matched?
- **isolated_geomedian → bard:** does the commit rule reduce harmful revisions without deleting useful corrections?

Do not compare joint and isolated arms until delivered evidence identities are audited; context-budget differences otherwise confound isolation with evidence quantity.

Run a real 8–32-ID canary through all five arms on each receiver first. Confirm nonempty outputs, exact prompt/image/evidence identities, complete provenance, comparable branch prefixes, no OOM/repetition, and the scorer's per-ID decision. Then run the **complete predeclared set**, not a score-selected subset. VQA-RAD (451 TEST IDs) is a low-cost descriptive engineering gate; SLAKE (2,094) adds language/source diversity. Because their TEST outcomes have already been inspected in this project, neither can be used to tune thresholds or called a pristine held-out validation set. Use an untouched development/source split for parameter choice when available, freeze parameters there, and label subsequent TEST analyses accordingly. PathVQA TEST is also previously inspected and is a post-hoc failure analysis, not an independent confirmatory holdout.

### Primary statistics

Report:

- paired per-example score delta versus generalist;
- positive score-change mass;
- negative score-change mass;
- fraction of changed outputs;
- cluster-bootstrap intervals by patient when available, otherwise by image;
- delivered source-group count (n=0/1/2/3+).

For item `i`, let `s_mi` be the **same upstream scorer's** value in `[0,1]` for method `m`, and `s_0i` the generalist value. On the identical `N` IDs report `rescue_m = mean(max(s_mi-s_0i,0))`, `harm_m = mean(max(s_0i-s_mi,0))`, and `net_m = rescue_m-harm_m`. For binary CE, also report the paired counts **wrong→right** and **right→wrong**; for OE short-answer recall, use score masses, not a binary-correctness label. Report exact-text change rate separately from parsed-decision change rate and from the nonempty/parseable rate. A lower change rate alone is not an improvement. The BARD claim needs lower harm **without an unacceptable loss of rescue**, or a better predeclared rescue–harm operating curve against both isolated_geomedian and Stage-2 consensus; state the development-set tolerance before examining TEST.

Use paired cluster bootstrap (10,000 resamples, fixed public seed) at deidentified patient level if available, otherwise image level; resample the **same clusters across arms** and give 95% intervals for score, rescue, harm, and each prespecified contrast. Report patient/image grouping, number of clusters and singleton fraction. Apply Holm correction to the three mechanistic contrasts above; other source/modality subgroups are exploratory. Preserve all attempted samples, including empty or truncated outputs, in the denominator and show raw versus deterministic, answer-blind repair as separate rows. Never pick a repair by its reference score.

Minimum paper display: (A) a five-arm table per receiver/dataset with `N`, coverage, score, rescue, harm, net, exact/decision change, parse rate, forward calls and GPU time; (B) a paired-contrast table with effect and 95% cluster interval; (C) one rescue–harm plot including the consensus baseline. Stratify descriptively by 0/1/2/3+ delivered groups, source family, CE/OE, and modality where counts permit. Include negative findings and representative cases selected by a **predeclared error category**, not by the desired answer. Account for native-expert acquisition time separately from receiver decoding time so cached evidence does not appear cost-free.

### Early stopping

Stop before a GPU scale-up if:

- expert coverage is effectively zero;
- wrapper parity against the intended baseline fails;
- all isolated arms reproduce the generalist.

Do not stop because an interim TEST score is low or high. A failed canary or missing identity field is an engineering failure to repair; a valid negative mechanism result remains a reportable scientific result.

## Stage 2 — closest-prior baseline (reuse branch scores)

If Stage 1 shows a nontrivial signal, add a **base-relative/quorum consensus** decoder inspired by *Inference-Time Consensus for Mitigating Hidden Behaviors from LLM Fine-Tuning* (arXiv:2607.23394).

Use the **same receiver-branch distributions**. This is a decoding baseline, not a new expert run.

Compare its rescue-harm curve and intervention coverage with BARD. Do not tune quorum/fault parameters on the final test set.

At each matched prefix reuse the **same cached base and isolated branch scores**, candidate mask and evidence renderings. Predefine any quorum threshold, weighting and tie-break on development data. Record the exact token rule and whether it needs extra receiver forwards; if logits/branch scores were not saved—or the consensus output diverges into a new prefix—run the required receiver branches rather than claiming a zero-compute free-generation replay. A consensus win or tie means the paper should foreground the heterogeneous receiver interface and narrow any unique claim about BARD commitment.

## Stage 3 — smallest failure diagnosis

Run the same fixed setup on one failure-domain benchmark, with PathVQA as the current natural candidate, only after Stage 1 is frozen.

Classify the failure into:

- **missing source coverage:** most cases have zero or one delivered source group (`n ≤ 1`);
- **ineffective evidence:** isolated arms do not beat the generalist;
- **harmful aggregation/commitment:** unprotected isolated aggregation and BARD differ materially in rescue/harm behavior.

Do not infer one of these causes from aggregate accuracy alone.

For each class report the paired `joint_all`, `isolated_mean`, `isolated_geomedian`, and `bard` rescue/harm masses, plus evidence availability and a trace of candidate, anchor, selected token and commit reason. The existing PathVQA false-negative/invalid-answer observations are hypotheses for this analysis, not tuning labels. Freeze any revised route or gate on development data before evaluating TEST again; keep the original result visible.

## Full evaluation trigger

Only after one configuration survives Stages 1–3 should the complete benchmark suite be rerun.

The full run is **confirmation**, not another tuning round.

The confirmatory matrix is the predeclared pair of receivers × the paper's complete eligible datasets, using one frozen configuration and scorer per task. Short-answer VQA uses the established CE/OE composite; MIMIC-CXR report generation remains separate with BLEU-4, ROUGE-L and METEOR, never pooled with VQA. Include all full TEST IDs, coverage/empty/unfinished/repetition counts, per-source breakdown, and compute cost. If a method is incompatible with a receiver or lacks a faithful implementation, mark it **not applicable/unverified**, not zero or an approximate substitute. Do not announce SOTA unless the full matched protocol supports the comparison.
