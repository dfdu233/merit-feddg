# Active full TEST evaluation — Huatuo Adaptive BARD

Latest user instruction: directly evaluate official TEST and compare effects. Authorized GPUs: cloud 5090 GPU0+GPU1, host GPU1 only. No host GPU0 jobs.

Source canary COMPLETE: reports/huatuo-spatial-v1/RESULTS.md, results.json. SLAKE16 baseline=BARD65.625%; VQA16 baseline34.583%, mean=median=BARD42.5%. BARD has 2 VQA rescues and 2 harms. No BARD-specific source advantage over median. Do not tune from these outcomes.

Active root: /home/dbw/merit-feddg-huatuo-spatial/runs/huatuo-test-v1. Full manifest2545 (SLAKE2094=1061English+1033Chinese, VQA451), all image hashes and historical Huatuo benchmark prompts checked. Rows are from /home/dbw/merit-feddg-chain-5066828/runs/official-test2545-inputs-v2/{slake,vqa_rad}/protocol.json. Frozen image-only Huatuo routes reused from /home/dbw/merit-feddg-expert-coverage/runs/huatuo-merit-full-v1/{dataset}/routing. Old expert cache lacks sufficiently explicit weight provenance; recomputing native outputs, no raw-mask stripping.

Expert cache identity4665a34cfc1cad8b2883e493bd599b7f43415cbfe8b5ea5d0aab863aa2042c85. All6987 native requests COMPLETE, exact identity/request keys validated on both host and cloud (zero missing); protocol complete. All masks/CAMs retained. HostGPU1 native stage done, no host receiver job. Cloud disk350GB expanded, plenty free.

Fresh Generalist COMPLETE2545. BARD first384 complete; active cloudGPU0 bard-main0 andGPU1 bard-main1 cover global indices384:2200 stride2. Both use200k per-worker ceiling, parity-validated cached KV, diagnostic stress disabled without changing decisions. MUST launch tail2200:2545 stride2/index0 and1 on GPUs as each main worker completes. No TEST label tuning. Logs and outputs under runs/huatuo-test-v1. Cloud /home/dbw symlink: rsync --keep-dirlinks.

Latest scored snapshot: BARD1134/2545, includes ALL1061EnglishSLAKE: freshGeneralist56.1603%,BARD55.5540% (48better46worse967equal). Chinese73/1033:59.2466→64.1553%, partial only. No VQA candidate in that snapshot. Detailed same-sample historical baseline reference tables: reports/huatuo-test-v1/interim-results.json. SLAKE metric answer-token recall, not plain accuracy. Historical numeric runtime differs, hence legacy comparisons are reference-only.

Opt-in KV wrapper scripts/huatuo_cached_receiver.py: real score parity bitwise on host/cloud ordinary+spatial vectors, and exact full tokens for15 arms across3cloudcases. Host still fails historical cloud-origin long token parity on4090 even under Torch2.8; do not pool its receiver predictions. Host new venv /home/dbw/venvs/huatuo-bard-torch28 has torch2.8 cu128,torchvision0.23,bitsandbytes0.47. Old env unchanged for native specialists.

Committed execution/source checkpoint4a69eb9 pushed. Current uncommitted batch: optional read-only fault-diagnostic skip + regression; routing_source metadata correction (TEST actually uses Huatuo routes, inherited old LLaVA label was wrong); protocol/legacy/interim reports. Need commit/push this checkpoint and final TEST results later. No main merge.
