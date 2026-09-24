# ResearchStudio-guided MERIT experiment plan

This directory records the shortest falsification path for the current MERIT paper.

It contains **no new benchmark results** and should not be cited as evidence that a proposed mechanism works. The purpose is to bind the reported results to a reproducible method configuration, isolate the smallest missing mechanism experiments, and stop early when the hypothesis is unsupported.

## Read in this order

1. [RESEARCHSTUDIO_IDEA.md](RESEARCHSTUDIO_IDEA.md): the current receiver-mediated collaboration hypothesis.
2. [NOVELTY_AUDIT.md](NOVELTY_AUDIT.md): scoped prior-art collision check and the defensible novelty delta.
3. [MINIMAL_EXPERIMENT.md](MINIMAL_EXPERIMENT.md): staged experiment plan designed to avoid another full benchmark sweep before the mechanism is validated.
4. [DATA_REQUEST_MINIMAL_CN.md](DATA_REQUEST_MINIMAL_CN.md): minimum server-side data needed to run the paired analysis.
5. [ROW_SCHEMA.json](ROW_SCHEMA.json): normalized per-example result schema for matched comparisons.
6. [MAIN_TABLE_BINDING_AUDIT.md](MAIN_TABLE_BINDING_AUDIT.md): current full-result provenance/scorer caveats and rows that are not yet paper-ready.

The [minimal plan](MINIMAL_EXPERIMENT.md) now includes the paper-ready five-arm contrast matrix, paired rescue/harm definitions, statistical unit, reporting tables/plot, compute accounting, and a guard against turning previously inspected TEST splits into tuning data. [DATA_REQUEST_MINIMAL_CN.md](DATA_REQUEST_MINIMAL_CN.md) identifies the exact provenance and per-example fields needed to populate those displays. These are specifications; no ablation result is asserted by this documentation commit.

The live VQA-RAD execution is tracked on the separate experimental branch in the [run ledger](https://github.com/dfdu233/merit-feddg/blob/experiments/huatuo-adaptive-bard-validation-v1/reports/researchstudio_ablation/VQARAD_RUN_LEDGER.md). Its [paired analysis code](https://github.com/dfdu233/merit-feddg/blob/experiments/huatuo-adaptive-bard-validation-v1/scripts/analyze_researchstudio_ablation.py) rejects incomplete or evidence-mismatched runs. Both are preparation artifacts, not a completed result or a change to `main`.

## Current decision rule

Do **not** tune a new gate, fault budget, quorum, or expert pool on the final benchmark before the mechanism experiment is complete.

The immediate questions are:

- Does source isolation help when the delivered expert evidence is held fixed?
- Does the anchored commit rule reduce harmful revisions without deleting useful corrections?
- Can a simpler base-relative/quorum consensus baseline explain the same effect?

Only a configuration that survives these checks should be promoted to a new full benchmark run.
