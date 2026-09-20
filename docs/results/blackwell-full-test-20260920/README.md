# Full official TEST evaluation — execution in progress

Fixed candidate: C Fisher-weighted layout-nuisance projection. All four arms use the same HuatuoGPT-Vision-7B checkpoint, original images and native expert packets. No algorithm or generation-setting changes were made for TEST.

Scope: VQA-RAD451 and SLAKE2094. Original host GPU1 completed235 VQA cases (75712 forwards, including interrupted work); its unfinished216-case suffix migrated to cloud GPU0 after that device finished SLAKE shard0. Original full VQA job remains honestly interrupted; a derived235-record artifact is used only for offline prefix scoring. cloud RTX5090 GPUs run SLAKE modulo4 shards0/1 then3; user-authorized original host GPU0 shares with an identified unrelated MRI training job and runs shard2. Every execution shard has native generalist, compact, candidate and layout-average control; cloud native controls are regenerated in the same runtime. Device/runtime strata must be retained in final reporting.

Cloud actual environment: PyTorch2.8.0+cu128, transformers4.37.2, eager Qwen2 attention. Original host: PyTorch2.0.1+cu117, same transformers and attention class. Model weight hashes match. Certified pre-rendering preserves original prompt hashes; candidate projection and decoding are unchanged.

Host GPU0 sharing was explicitly authorized. Its first attempt failed on input ownership before any model forward; permissions were repaired, the unchanged job restarted, and both native/off/audit canaries passed. First observed peak allocation22.38GiB. Other training is preserved. Cloud observed initial peak22.32GiB on32GB5090.

Global authorized cap is1,000,000 actual language-model forwards, including historical44885, stopped native-refresh42016, all active jobs and failed attempts. Per-job ceilings are not expenditure reservations. Aggregate counters are supervised across hosts; no atomic distributed cap is claimed.

`historical-vqa-baselines.json` rescored eleven existing baseline prediction files with the same frozen CLOSED parser and OPEN token-recall functions. Their runtime is historical, so they provide context rather than a strict matched-current-runtime leaderboard. Candidate full-TEST metrics remain pending.

Final analysis will retain the frozen scoring protocol and add separately labelled official-source-typed SLAKE metrics, language/task breakdowns, image-cluster paired intervals, harm/retention, token lengths, forward counts and memory. SLAKE's historical manifest incorrectly labels all questions open; the frozen aggregate is therefore token recall, not accuracy. See the pre-scoring evaluation notes and innovation review in the adjacent official-test report directory.
