# LLaVA-Med VQA-RAD decode-length repair (separate exploratory arm)

**Status at 2026-09-24 11:18 UTC: ongoing, not scored or substituted for raw.** This is a deterministic, answer-blind decode-only extension for raw five-arm samples with at least one `finished=false` arm. It is not a new main-table method or a correction chosen by reference correctness. All five arms are regenerated for each selected ID so exact token-prefix parity can be checked; raw artifacts in `runs/researchstudio-vqarad-llava-matched-v1/` remain untouched.

| Item | Frozen value |
|---|---|
| Raw run/source | VQA-RAD full 451 TEST; source, manifest, cache/model/scorer identities in `VQARAD_RUN_LEDGER.md` |
| Selection | 36 IDs from the checksum-backed raw snapshot at 11:15 UTC; selected **only** if any of the five original arm artifacts had `finished=false` and a complete provenance file |
| Extended run | `runs/researchstudio-vqarad-llava-length-probe-v1/`, same five arms, same prompt/image/native cache/receiver config/matched delivered evidence, `--decode-max-tokens 128`; original allowance 64 |
| Host GPU | GPU1 (48 GiB), co-located with the independent SLAKE English run; no change to the raw run's code or outputs |
| First canary | `vqa-rad-test-0012` completed previously: all five original 64-token prefixes exactly matched; all five extended outputs finished |
| Concurrent pilot | `0017`, `0019` finished under co-location without OOM; SLAKE English continued writing |
| Pause/resume | Initial 36-ID batch was SIGTERM-stopped after four new complete IDs to investigate a tokenizer warning; completed files retained. Resumed same selection/output/protocol in `runs/researchstudio-vqarad-llava-length-probe-v1-batch-resume.log`. No raw run was interrupted. |
| Scoring rule | Score original nonempty unfinished text as generated. Present any validated extended result **beside** raw, never as a silent replacement. Reject per-arm repair if its original token sequence is not an exact prefix or provenance/evidence differs. |

Frozen selected IDs (the still-running raw VQA job may add later `finished=false` IDs, which require a subsequent separately recorded selection):

```text
vqa-rad-test-0012  vqa-rad-test-0017  vqa-rad-test-0019  vqa-rad-test-0029
vqa-rad-test-0047  vqa-rad-test-0051  vqa-rad-test-0052  vqa-rad-test-0059
vqa-rad-test-0068  vqa-rad-test-0092  vqa-rad-test-0125  vqa-rad-test-0126
vqa-rad-test-0169  vqa-rad-test-0181  vqa-rad-test-0183  vqa-rad-test-0184
vqa-rad-test-0188  vqa-rad-test-0192  vqa-rad-test-0199  vqa-rad-test-0211
vqa-rad-test-0215  vqa-rad-test-0222  vqa-rad-test-0252  vqa-rad-test-0277
vqa-rad-test-0296  vqa-rad-test-0302  vqa-rad-test-0312  vqa-rad-test-0334
vqa-rad-test-0360  vqa-rad-test-0361  vqa-rad-test-0381  vqa-rad-test-0396
vqa-rad-test-0399  vqa-rad-test-0403  vqa-rad-test-0408  vqa-rad-test-0414
```

## Context-length warning investigation

The concurrent log warned once that tokenizer input length `2065 > 2048`. This is **not** by itself proof that generated history was clipped, but required inspection. The checkpoint declares `tokenizer_model_max_length=2048` and `max_position_embeddings=32768`. Official LLaVA-Med `llava/model/llava_arch.py` truncates multimodal **prefill embeddings** at 2048. The cached receiver's `LlavaMedIncrementalStream._prefill` first calls `_validate_context(inputs, 1)`, which rejects an actual expanded prefill over 2048 before that helper; later tokens use `commit` on the KV cache and do not call the multimodal prefill helper again. Thus an answer crossing the tokenizer's 2048 marker after prefill is not silently clipped by that helper, although it remains subject to the model's 32768 position limit. The seven complete probe IDs examined at the pause had maximum initial expanded contexts of 1372–1944, maximum context-plus-output of 1444–2057, and **all five raw token prefixes matched exactly**. Continue checking every newly completed ID; the warning alone is not an acceptance gate.

This ledger does not assert that all future IDs pass parity, that extended text is clinically better, or that a full repaired score exists. The original full 451-ID raw analysis remains the primary result.
