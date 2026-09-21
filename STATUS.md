# Active full TEST evaluation — Huatuo Adaptive BARD

Latest user instruction: directly evaluate official TEST and compare effects. Authorized GPUs: cloud 5090 GPU0+GPU1, host GPU1 only. No host GPU0 jobs.

Source canary COMPLETE: reports/huatuo-spatial-v1/RESULTS.md, results.json. SLAKE16 baseline=BARD65.625%; VQA16 baseline34.583%, mean=median=BARD42.5%. BARD has 2 VQA rescues and 2 harms. No BARD-specific source advantage over median. Do not tune from these outcomes.

Active root: /home/dbw/merit-feddg-huatuo-spatial/runs/huatuo-test-v1. Full manifest2545 (SLAKE2094=1061English+1033Chinese, VQA451), all image hashes and historical Huatuo benchmark prompts checked. Rows are from /home/dbw/merit-feddg-chain-5066828/runs/official-test2545-inputs-v2/{slake,vqa_rad}/protocol.json. Frozen image-only Huatuo routes reused from /home/dbw/merit-feddg-expert-coverage/runs/huatuo-merit-full-v1/{dataset}/routing. Old expert cache lacks sufficiently explicit weight provenance; recomputing native outputs, no raw-mask stripping.

Expert cache identity4665a34cfc1cad8b2883e493bd599b7f43415cbfe8b5ea5d0aab863aa2042c85. HostGPU1 runs prepare_bard_expert_cache.py --skip-expert medcpt_pubmed with old working Huatuo env + HF shared offline vars + BiomedParse safe.directory. CloudGPU0 runs run_medcpt_requests.py against real /root/merit-medcpt-artifacts KB. CloudGPU1 runs run_adaptive_bard_canary.py --methods generalist --skip-stress over all2545 (150k forward ceiling). Logs native-host.log (local), medcpt.log/generalist.log (cloud). Cloud /home/dbw is symlink: rsync --keep-dirlinks. Cloud data disk now350GB/303GBfree, root8.9GBfree.

Next: merge exact request-keyed MedCPT cache to host, native cache to cloud; verify all requests complete under frozen identity, mark protocol complete. Start full BARD paired with saved fresh Generalist (no labels during inference), two cloud GPUs. Score by official dataset and language; use original reference metadata and report metric contracts. Full TEST primary comparison is Generalist vs BARD; 5-arm ablation already completed source32. Never use incomplete/absent BARD outputs as fallback predictions.

Opt-in KV wrapper scripts/huatuo_cached_receiver.py: real score parity bitwise on host/cloud ordinary+spatial vectors, and exact full tokens for15 arms across3cloudcases. Host still fails historical cloud-origin long token parity on4090 even under Torch2.8; do not pool its receiver predictions. Host new venv /home/dbw/venvs/huatuo-bard-torch28 has torch2.8 cu128,torchvision0.23,bitsandbytes0.47. Old env unchanged for native specialists.

Uncommitted coherent batch: cached wrapper, runner scheduling/method selection, expert-stage split, sourcefinalreports. Prior commitf0e9d7b pushed on experiments/huatuo-adaptive-bard-validation-v1. No main merge. Need commit/push current checkpoint and TEST results later.
