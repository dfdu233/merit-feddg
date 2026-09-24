# ResearchStudio SLAKE matched-evidence run ledger

Raw five-arm experiment started 2026-09-24 on host `xcyd-Z790-UD-AX`, GPUs 0 and 1 (48 GiB each). **Ongoing; not a final metric table.** The preceding eight-case canary on each receiver passed the nonempty, finished, source-provenance and equal-delivered-evidence checks; full jobs resumed the same directories and reused those eight cases.

| Frozen item | Identity |
|---|---|
| Source | Native `/home/dbw/data/SLAKE/test.json`, SHA256 `6be8f7b4c5a46cdbc713a5210a25b6ed5aa1fd1574c83cefb4f998131f17c2c3`; 2,094 TEST qids, 836 CE/1,258 OE, EN1,061/ZH1,033, CT865/X-Ray740/MRI489 |
| Exact run slice | Manifest indices `[0,2094)` with identical native qid/question/image-path order; the paired analyzer verifies actual image SHA256 bytes against the frozen manifest |
| Arms | `generalist`, `joint_all`, `isolated_mean`, `isolated_geomedian`, `bard` under `--matched-joint-evidence` |
| Scientific label | `BARD (matched evidence)`, not the older full-native-schedule BARD headline |
| Native cache | Identity `4665a34cfc1cad8b2883e493bd599b7f43415cbfe8b5ea5d0aab863aa2042c85`; protocol SHA256 `dad86fc2c9cce5399c4f7cabcc6fe64b15d95ddc9a0979260772d0eb6eeed2d2` |
| Executed runner | `scripts/run_adaptive_bard_canary.py` SHA256 `266491b789aa0cb4dc6bc28abe02b8041a4c3d87db03f5d1cd7f5580ceefa55b` |
| Executed BARD protocol | `merit_feddg/bard_protocol.py` SHA256 `5191265bb63505539423020c100ba232585c7ba9af8c418ec99546d475911ec7` |
| Executed capability runtime | `merit_feddg/capability_runtime.py` SHA256 `f8f2d6cd7410b53765143e592b44c52160e2836f7154ab3fa4ed528ef6c1cb9c` |
| Receiver loader | `merit_feddg/generalist_factory.py` SHA256 `8579b21550991d0b3592f421d12f91364b217b690e306a943897b3f6c215668e` |
| Git caveat | Experimental branch HEAD at launch `9206bb4115c6cd8902f1567f3acc003b37abc3f6`, with dirty runtime files; exact executed file hashes above, **not** HEAD alone, identify code. The local-only `--decode-max-tokens` capability existed but was **not supplied** to these raw runs. |
| Decoding/transport | `--skip-stress --cached-receiver --compress-artifacts --reference-native-evidence --max-forwards 200000000`; no answer-selected repair or EOS suppression; original empty/unfinished text, if any, stays in denominator |
| Scorer | ANCHOR mixed CE/OE v4 with pinned parser file SHA256 `497831651bfb9e178c7a79b9b5312ef578e6e36c3fa413c92cb12d3c63132163`: CE strict 0/1, OE reference-token recall, sample-weighted over all 2,094 IDs. A `finished=false` nonempty answer is scored from its text, not automatically wrong. |

| Receiver | Config SHA256 | Manifest SHA256 | Answer max | Host GPU | Raw directory / log |
|---|---|---|---:|---:|---|
| HuatuoGPT-Vision-7B | `fac80988b1d0377bf81755abff0e53ad07a508e262836496f1f82d3d4c20c50f` | `9d9cdb56e8c8a8ab1023602bb9d453cd11d146d3e5b2d53a5250240bd9034439` | 1024 | 0 | `runs/researchstudio-slake-huatuo-host-matched-v1/`, `runs/researchstudio-slake-huatuo-host-matched-v1-full.log` |
| LLaVA-Med-7B | `0d3f8c62b6ba02d944dbcd7d59a33f585777b8c5e7093bb692d4edb8285147a7` | `b9ec5d6a8e252eb07cca1d88e73c5f18e3054a427602f57a0966898e40ae460a` | 64 | 1 | `runs/researchstudio-slake-llava-host-matched-v1/`, `runs/researchstudio-slake-llava-host-matched-v1-full.log` |

Model/vision-tower weight-byte SHA256 values are identical to the checkpoint identities documented in [VQARAD_RUN_LEDGER.md](VQARAD_RUN_LEDGER.md); the same resolved config files are used. The host full-run PIDs at launch were Huatuo `704760` and LLaVA `701980`, detached with `setsid`; process state and artifacts, not these PID numbers alone, establish liveness. The eight canary cases scored successfully with the pinned CE/OE evaluator, but their scores are **not** representative paper results. The full run must pass exact 2,094-ID, prompt/image/evidence/receiver identity, blank, truncation and repetition audits before the paired analyzer emits any manuscript metric. SLAKE TEST outcomes were inspected previously; paired inference is descriptive, not pristine held-out tuning evidence.

## LLaVA partition update, 2026-09-24 10:25 UTC

The original host LLaVA full `[0,2094)` process was normally stopped after its first 246 new `COMPLETE` records (plus the eight reused canary IDs). Its outputs were retained; no directory was removed. The same five-arm command and same executed code/config/cache identity resumed from those files on GPU1 as **English-only `[0,1061)`**, PID `742851`, log `runs/researchstudio-slake-llava-host-matched-v1-english.log`. Its first new record was manifest index 254, confirming no first-eight or prior full-run duplication. The Chinese `[1061,2094)` interval is reserved for cloud GPU1 after its VQA-RAD run and queued eight-case cross-machine parity check finish, provided code, model, cache and scorer identity are reconciled. If cloud GPU1 is unavailable, the host output can instead resume the same interval. Do not combine any shard until disjoint full-ID coverage and identical protocol identities are audited.
