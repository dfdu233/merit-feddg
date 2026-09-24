# Strict base-relative consensus comparator

Status at 2026-09-24 11:40 UTC: **LLaVA-Med VQA-RAD full 451-case run ongoing; not a result.** This is Stage 2 of `docs/researchstudio_experiments/MINIMAL_EXPERIMENT.md`, motivated by the completed Stage-1 LLaVA VQA-RAD paired contrast. It is an evidence-conditioned adaptation of the strict `q=m` rule, not the prior paper's trained-reference-model result. The relaxed-quorum rule is deliberately excluded because the cited equation and released code differ for its downward branch.

| Item | Frozen protocol |
|---|---|
| Source / denominator | Official VQA-RAD full TEST 451 IDs, source SHA256 `eda8463126dcb89929b18681823cc661c0c5c8e221ffd2e5297ed59650b54c03` |
| Receiver / native cache | Same LLaVA-Med-7B checkpoint, config, prompt, manifest, source schedule and cache identity as `VQARAD_RUN_LEDGER.md`; no new native expert inference |
| Arms | `joint_all` (context/evidence identity control) and `consensus_strict`; Stage-1 generalist/isolation/BARD raw outputs stay untouched and are joined by exact ID only after all 451 consensus cases complete |
| Strict decoder | At each self-generated prefix, compare each isolated branch's normalized next-token probability against the common base probability; only unanimously signed changes are retained, using the weakest up/down shift, otherwise revert to base, renormalize and choose greedy argmax; stop at EOS or original 64-token answer budget |
| Independence | Branches use the same matched delivered native evidence as Stage 1; consensus runs its own free-generation trajectory and therefore re-evaluates branch scores at each new prefix, not a zero-cost replay of divergent prior answers |
| Executed code | Runner SHA256 `9f38387fafa4bc9adedf674129c24b6486462dea531a5685a1dd5f65449988bc`; BARD adapter `bf8b0a53814d986200b5beac8c8917718f581936061776b6f8328353bdac0bc3`; strict core `c35d4439b8336e19e64883221fb17b54d49e8d026e53168b69b0b98ef193e3dc`. Runner/adapter worktree was dirty at launch; **HEAD alone does not identify execution**. |
| Runtime | Host `xcyd-Z790-UD-AX` GPU1, co-located with independent SLAKE English run; `runs/researchstudio-vqarad-llava-consensus-strict-canary-v1/`, full log `runs/researchstudio-vqarad-llava-consensus-strict-full-v1.log`, detached process group `862062` at launch; exact `[2094,2545)` manifest slice |
| Scoring | After full coverage: same CE strict / OE reference-token recall scorer as Stage 1; preserve nonempty unfinished text, report empty/unfinished/parse and paired rescue/harm separately; no TEST-tuned quorum or answer-selected repair |

The first real two-case canary exited 0. For IDs `vqa-rad-test-0000` and `0001`, frozen prompt and row matched Stage 1, `joint_all` token IDs and evidence matched Stage 1 exactly, consensus evidence matched the same delivered evidence, and both consensus answers reached EOS. The full run resumes those two preserved outputs. This canary validates wiring only; its score is not a result. Four CPU tests in `tests/test_consensus_baseline.py` plus the existing BARD tests passed (28 total), including exact probability-rule behavior, common-support validation and independent-prefix decoding.

Before any paper comparison, require exact 451-ID coverage, per-ID prompt/image/cache/config/evidence parity with Stage 1, full output quality counts and the same paired scorer/cluster bootstrap. Full native acquisition time remains unmeasured for two producer stages, so do not call cached decoder time full inference cost.
