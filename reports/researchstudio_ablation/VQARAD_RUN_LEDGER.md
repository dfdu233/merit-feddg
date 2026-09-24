# ResearchStudio VQA-RAD matched-evidence run ledger

This records the **raw, ongoing** five-arm experiment launched 2026-09-24 on cloud node `connect.nmb1.seetacloud.com:42865` (two RTX 4090). It is not a final metric table. The old dynamic `joint_all` runs are excluded: they delivered different native evidence than the isolated arms.

| Item | Frozen identity |
|---|---|
| Dataset | Official VQA-RAD TEST, 451 IDs (251 CE, 200 OE; 203 unique image clusters), `/home/dbw/ANCHOR/data/vqa_rad/official_test_full_v1.json`, SHA256 `eda8463126dcb89929b18681823cc661c0c5c8e221ffd2e5297ed59650b54c03` |
| Split/index | Frozen short manifest indices `[2094,2545)`; all 451, no sampled subset |
| Arms | `generalist`, `joint_all`, `isolated_mean`, `isolated_geomedian`, `bard` |
| Scientific label | `BARD (matched evidence)`, not the earlier `BARD (full native schedule)` headline |
| Native cache | `4665a34cfc1cad8b2883e493bd599b7f43415cbfe8b5ea5d0aab863aa2042c85`; `protocol.json` SHA256 `dad86fc2c9cce5399c4f7cabcc6fe64b15d95ddc9a0979260772d0eb6eeed2d2` |
| Runtime base revision | `92a9d26b760c2ad7e1dd4a1b854f6b1cedba763a`; worktree was dirty, so this revision alone is **not** the executed code identity |
| Executed runner | `scripts/run_adaptive_bard_canary.py` SHA256 `01670098767b3d3a172e2371238977d210e395caaf1805672e1b3402cc19bce5` |
| Executed BARD protocol | `merit_feddg/bard_protocol.py` SHA256 `91b458624fdbfe5a71d6ef09852ccac3e00735c27fa915a445e7f413bc37f21b` |
| Executed capability runtime | `merit_feddg/capability_runtime.py` SHA256 `f8f2d6cd7410b53765143e592b44c52160e2836f7154ab3fa4ed528ef6c1cb9c` |
| Receiver loader | `merit_feddg/generalist_factory.py` SHA256 `8579b21550991d0b3592f421d12f91364b217b690e306a943897b3f6c215668e`, identical local/cloud; `.py` source trees for both model implementations showed no file-content difference in a read-only checksum comparison (the cloud copies lack Git metadata) |
| Relevant dirty diff | Runner, BARD protocol, capability runtime and matched evaluation combined diff SHA256 `41c92acf5366f34cbd2570dee7d76b2c0ce2a60123533afb2a1a01ef5f3cd8f4`; per-file hashes above are the more direct identity |
| Decoder | `--matched-joint-evidence --skip-stress --cached-receiver --compress-artifacts --reference-native-evidence`; one shard, complete `[2094,2545)`, max-forwards ceiling 200,000,000 |
| Empty/repetition policy | Raw run, no EOS suppression, no answer-selected repair. Preserve empty, unfinished and repetition outputs in denominator; separate deterministic repair experiment only. |
| Scorer | ANCHOR mixed CE/OE v4 (`0832828aae177f143d17ddda52641e0cf4e98cc721cb32edc7fce1637a0455bd`); underlying parser SHA256 `497831651bfb9e178c7a79b9b5312ef578e6e36c3fa413c92cb12d3c63132163`. CE strict 0/1; OE reference-token recall; sample-weighted mean. |
| Inferential status | TEST outcomes were previously inspected; paired analysis is descriptive, not pristine held-out tuning evidence. No SOTA claim until all 451 IDs and quality audit pass. |

## Receiver-specific identities and outputs

| Receiver | Model/checkpoint in resolved config | Config SHA256 | Manifest SHA256 | GPU | Raw output/log |
|---|---|---|---:|---:|---|
| HuatuoGPT-Vision-7B | `FreedomIntelligence/HuatuoGPT-Vision-7B`, `/home/dbw/models/HuatuoGPT-Vision-7B`; `bfloat16`; decoder max 1024 | `fac80988b1d0377bf81755abff0e53ad07a508e262836496f1f82d3d4c20c50f` | `9d9cdb56e8c8a8ab1023602bb9d453cd11d146d3e5b2d53a5250240bd9034439` | 0 | `runs/researchstudio-vqarad-huatuo-matched-v1/`, `runs/researchstudio-vqarad-huatuo-matched-v1-full.log` |
| LLaVA-Med-7B | `microsoft/llava-med-v1.5-mistral-7b`, `/home/dbw/ANCHOR/hf_cache/llava-med-v1.5-mistral-7b`; `float16`; decoder max 64 | `0d3f8c62b6ba02d944dbcd7d59a33f585777b8c5e7093bb692d4edb8285147a7` | `b9ec5d6a8e252eb07cca1d88e73c5f18e3054a427602f57a0966898e40ae460a` | 1 | `runs/researchstudio-vqarad-llava-matched-v1/`, `runs/researchstudio-vqarad-llava-matched-v1-full.log` |

The local and cloud copies agree on checkpoint metadata **and all loaded weight-shard bytes**. Metadata: Huatuo `config.json` `3d96adabfe401f3f533f98c66478324c01159370e55c58bfdea1191a642f9fb9`, safetensors index `9326fc7c300ba6e894d50a8f9acd6f5b80e5ed8665648119d5005eaee2daf47a`; LLaVA `config.json` `f6ae889c5488ef86895e78f641339062962dd6b434666019fa119ab09d2bd8b3`, safetensors index `d5ecec60dba218c6621cfa524d739de942b4552bcbb25efe63848c18a731b2f6`; both vision preprocessors `d253881f65322dc546df59cf925a408e5538b8ecb5a1b496cdd36af9992686d4`.

Weight SHA256 values, in filename shard order `00001`–`00004`:

| Checkpoint | Shard 1 | Shard 2 | Shard 3 | Shard 4 |
|---|---|---|---|---|
| HuatuoGPT-Vision-7B | `da1f9b7837a6c54e7803a8a36dca76ba2b09f116b044225f1bcf04c6ab18db4d` | `5ab6f7191d5d35b13f22bd238155b83d15a683a56d9a6b46d038ef6e4d3b897b` | `cf5442ead64542c60c0ecc647fee679b7f18b531d47d169cac08d496e8b78be3` | `9df68cee325562ea25cba56f5bac5027000fd288031fbeb1f66000a4c8dafaf4` |
| LLaVA-Med-7B | `ef2190dc6c2a940e60f03f5fdb4dddb2320eb87801aeca5c40b0a28ce8aa420e` | `2b229607fecd98b8111320178e5bf3e2c527b05a942c85d65b5b507c76c1ed00` | `12b18ecdf8924d5fe28ada797fe6697fa60e62cba630759fbeb52975b261c4e2` | `1d2063fcd429d3f0f0a8a091b0522f0e02f2d85fe0e5b0eeb4ae168183a603bc` |

The Huatuo and LLaVA CLIP vision-tower binaries each hash to `c6032c2e0caae3dc2d4fba35535fa6307dbb49df59c7e182b1bc4b3329b81801`.

Both raw runs use the existing `scripts/run_adaptive_bard_canary.py` with `--cache`, the receiver-specific `--manifest` and `--config`, receiver-specific `--output`, the five `--methods` listed above, the decoder flags above, and `--start-index 2094 --end-index 2545 --shard-index 0 --shard-count 1 --max-forwards 200000000`. Their cloud processes are detached from the SSH terminal. The local copy is an incremental backup, not the source of truth while jobs remain active.

Scoring/export after completion: `scripts/analyze_researchstudio_ablation.py` requires exact full source-ID coverage, identical delivered native evidence across evidence arms, and an explicit `--executed-code-identity` from the runner/BARD/runtime hashes above rather than hashing the analysis worktree. It then emits `paired_rows.jsonl` and `summary.json` with raw scores, rescue/harm, paired cluster bootstrap and three Holm-corrected mechanistic contrasts. It must not run as a final report on partial outputs.
