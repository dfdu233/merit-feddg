# ResearchStudio-guided MERIT experiment plan

This directory records the shortest falsification path for the current MERIT paper.

The plan began as a specification; it now also links the first **full-TEST descriptive** mechanism result. Do not read one completed receiver–dataset cell as proof of the mechanism across models or datasets. The purpose is to bind reported results to a reproducible configuration, run the smallest falsifiable comparisons, and keep negative findings visible.

## Read in this order

1. [RESEARCHSTUDIO_IDEA.md](RESEARCHSTUDIO_IDEA.md): the current receiver-mediated collaboration hypothesis.
2. [NOVELTY_AUDIT.md](NOVELTY_AUDIT.md): scoped prior-art collision check and the defensible novelty delta.
3. [MINIMAL_EXPERIMENT.md](MINIMAL_EXPERIMENT.md): staged experiment plan designed to avoid another full benchmark sweep before the mechanism is validated.
4. [DATA_REQUEST_MINIMAL_CN.md](DATA_REQUEST_MINIMAL_CN.md): minimum server-side data needed to run the paired analysis.
5. [ROW_SCHEMA.json](ROW_SCHEMA.json): normalized per-example result schema for matched comparisons.
6. [MAIN_TABLE_BINDING_AUDIT.md](MAIN_TABLE_BINDING_AUDIT.md): current full-result provenance/scorer caveats and rows that are not yet paper-ready.
7. [MANUSCRIPT_ABLATION_DRAFT.md](MANUSCRIPT_ABLATION_DRAFT.md): paper-ready protocol language, both full-coverage VQA-RAD five-arm results, the LLaVA strict-consensus comparator, and the exploratory PathVQA diagnostic; SLAKE and Huatuo consensus remain pending.
8. [CASE_APPENDIX_VQARAD_LLAVA.md](CASE_APPENDIX_VQARAD_LLAVA.md): deterministic CE rescue, harm, protected-harm and missed-rescue examples from that complete run.

The [minimal plan](MINIMAL_EXPERIMENT.md) includes the five-arm contrast matrix, paired rescue/harm definitions, statistical unit, reporting tables/plot, compute accounting, and a guard against turning previously inspected TEST splits into tuning data. [DATA_REQUEST_MINIMAL_CN.md](DATA_REQUEST_MINIMAL_CN.md) identifies the exact provenance and per-example fields. The first completed result is documented separately and remains descriptive until the other prespecified cells and closest-prior comparator finish.

Execution is tracked on the separate experimental branch in the [VQA-RAD run ledger](https://github.com/dfdu233/merit-feddg/blob/experiments/huatuo-adaptive-bard-validation-v1/reports/researchstudio_ablation/VQARAD_RUN_LEDGER.md). Its [paired analysis code](https://github.com/dfdu233/merit-feddg/blob/experiments/huatuo-adaptive-bard-validation-v1/scripts/analyze_researchstudio_ablation.py) rejects incomplete or evidence-mismatched runs. The full five-arm VQA-RAD summaries for [LLaVA-Med](https://github.com/dfdu233/merit-feddg/blob/experiments/huatuo-adaptive-bard-validation-v1/reports/researchstudio_ablation/llava_vqarad_raw_full/summary.json) and [HuatuoGPT-Vision](https://github.com/dfdu233/merit-feddg/blob/experiments/huatuo-adaptive-bard-validation-v1/reports/researchstudio_ablation/huatuo_vqarad_raw_full/summary.json), and the [LLaVA strict-consensus comparison](https://github.com/dfdu233/merit-feddg/blob/experiments/huatuo-adaptive-bard-validation-v1/reports/researchstudio_ablation/llava_vqarad_consensus_strict_full/summary.json), are complete. [PathVQA's 58-case diagnostic](https://github.com/dfdu233/merit-feddg/blob/experiments/huatuo-adaptive-bard-validation-v1/reports/researchstudio_ablation/pathvqa_stage3_58_full/summary.json) is exploratory, not its full TEST result. Huatuo consensus and both SLAKE cells are still running. Nothing here changes `main`.

## Current decision rule

Do **not** tune a new gate, fault budget, quorum, or expert pool on the final benchmark before the mechanism experiment is complete.

The immediate questions are:

- Does source isolation help when the delivered expert evidence is held fixed?
- Does the anchored commit rule reduce harmful revisions without deleting useful corrections?
- Can a simpler base-relative/quorum consensus baseline explain the same effect?

Only a configuration that survives these checks should be promoted to a new full benchmark run.
