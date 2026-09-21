# Active official TEST: candidate-only multi-dataset evaluation

User scope: new Adaptive BARD on Huatuo + LLaVA-Med, prioritize short datasets where old method underperformed existing best baselines. Latest constraints: DO NOT rerun any baselines; DO NOT continue LLaVA-Med VQA-RAD (user says already SOTA). GPUs authorized: cloud5090 GPU0+GPU1 and HOST GPU1 ONLY. No hostGPU0 jobs. Frozen algorithm upstream5b15b16, experiment branch experiments/huatuo-adaptive-bard-validation-v1; latest pushed09bbd93. Current uncommitted checkpoint includes LLaVA KV wrapper, explicit frozen MCQ prompt support and results below. No main merge.

## Completed Huatuo evidence

Source32 complete: reports/huatuo-spatial-v1/RESULTS.md. VQA16 baseline34.583→BARD42.5%, equal mean/median; SLAKE16 unchanged65.625%. No BARD-specific source advantage.

Official VQA-RAD COMPLETE451: fresh baseline63.8910%, BARD63.0648% mixed metric. CLOSED251 77.2908→77.6892%; OPEN200 token recall47.0743→44.7112%.9better13worse429equal; image-cluster95%deltaCI[-2.499,+.750]pp. reports/huatuo-test-v1/vqa-final.json, RESULTS.md. Full baseline2545 already complete; no further baseline inference. Existing archived methods rescored offline in vqa-final.json, numerical runtime mismatch caveat.

SLAKE English COMPLETE1061: baseline56.1603%, BARD55.5540%;48better46worse967equal; CI[-2.323,+1.168]pp. Native CLOSED416 74.7596→70.9135%; OPEN645 44.1645→45.6477%.528 structural fallbacks,226 cases commit.25 harmful exact No→Yes versus10 beneficial. reports/huatuo-test-v1/english-diagnostics.json and english-changed-scores.json.

## Active cloud Huatuo final SLAKE remainder

Root /home/dbw/merit-feddg-huatuo-spatial/runs/huatuo-test-v1. Manifest2545 orderSLAKE2094 thenVQA451. Nativecache4665a34cfc1cad8b2883e493bd599b7f43415cbfe8b5ea5d0aab863aa2042c85 COMPLETE6987 exactkeys checked bothmachines; all masks/CAMs retained. Cloud /home/dbw is symlink: rsync --keep-dirlinks. Disk350GB.

Completed outputs: generalist2545; bard-batch0000 indices0:128; bard-batch0128 128:384; bard-main0 even384:1554 (586 cases,10432 completed-case forwards); bard-main1 odd385:1489 (553 cases,9676 forwards). Main workers intentionally terminated to prioritizeVQA, partial in-flight files excluded unless provenance exists. bard-vqa0 even2094:2545 and bard-vqa1 odd2094:2545 COMPLETE451. Do NOT rerun any of these.

ACTIVE: cloudGPU0 bard-slake-rest0 PID55803 even1556:2094 (269 cases); cloudGPU1 bard-slake-rest1 PID55704 odd1491:2094 (302 cases). Each --methods bard --skip-stress --cached-receiver --max-forwards200000. When complete, all2545 candidate cases should be present: scan bard-*/*/provenance.json, deduplicateassert. Refresh compact predictions and score wholeSLAKE en/zh/all against existing fresh Generalist. Cloud Python /root/autodl-tmp/merit-env/bin/python torch2.8cu128. Huatuo1024/rep1.2/min1/eager, KV bitwisevalidated. HostHuatuo receiver excluded due historical long token mismatch, evenhosttorch2.8.

## Active HOST GPU1 LLaVA expansion

Root runs/llava-test-v1. Config frozen LLaVA64 matches existing baseline budget. KV wrapper reuses existing native incremental stream; 3 actualVQA cases ordinary+spatial branches all inspected fullscorevectors bitwise equal production. reports/llava-test-v1/kv-parity.json. Wrapper unwraps vision-cache source. Current runner preserves explicit benchmark_prompt from raw answer-blind manifest, crucial for MMMU/PMC choice options; expertrequests unchanged. Huatuo running jobs have no benchmark_prompt and are unaffected.

LLaVA VQA CANCELLED peruser,184 completed cases retained across vqa-rad (56 hadfreshbaseline before userclarification) +vqa-rad-candidate. PID785411 terminated. NEVER resumeVQA. ACTIVE SLAKE2094 candidate-only PID809320, log slake-candidate.log, output slake-candidate; hostGPU1 newenv /home/dbw/venvs/huatuo-bard-torch28/bin/python. Input short-test-manifest.jsonl contains frozen original LLaVA benchmark prompts; --start-index0 --end-index2094 --max-forwards400000. Reuses complete nativeHuatuocache4665 with sameexperts andrequests, explicitly shared Huatuoimage-onlyroutes acrossreceivers (not historical LLaVA routing).

PathVQA full6719 input pathvqa-manifest.jsonl + frozen originalLLaVA pathvqa-routing.json. Cache b2fb387f19bd079203a85db91599dd4c0b08489d11a1b63a60c560a361f03084 in pathvqa-expert-cache. Native preparation ACTIVE hostGPU1 PID780905, oldworkingpython /home/dbw/.runtime/miniconda3/envs/huatuo/bin/python, --skip-expert medcpt_pubmed. Log pathvqa-native.log. Scheduled CONCH3503,MedCPT6719,BiomedParse5802,CheXagent248,XRVfindings227,XRVanatomy21. CONCH/CheX done; BiomedParse lastactive. OldHF cacheenv/safe.directory as previous nativepreparation. Runs concurrently withLLaVA, measured17GBreceiver/31GBfree before expertload; monitorOOM.

MedCPT fullKB downloading cloud→host via rsync execsession46158 (~2.9GB/4.4GB lastcheck), destination /home/dbw/merit-feddg/artifacts/knowledge/medcpt-pubmed. Wasmanifestonly; DO NOT retrieval before all3 fileSHAs matchmanifest. Cloudsource /root/merit-medcpt-artifacts/medcpt-pubmed,851839docs,chunk36. Once complete, run missing PathMedCPT6719 on hostGPU1 with originalauto-device config (smallmodel). Easiest rerun prepare_bard_expert_cache.py withoutskip once native finishes: allvisualcachehits, onlyMedCPTlive, then automatically markscomplete. Preserve sameidentity; no manifest/confignormalizationchanges to cachepreparer.

Path receiver NOT launched. After cache complete, candidate-only with --cached-receiver on availableGPU(s), full6719, preservebenchmarkprompts. Native pool frozen frommatched_bard, not oldexpandedQuiltpool. Forcloud need transferimages/nativecache/config/code; preferlosslesstar/zstd sortedbyimagehash. Full nativeKB hosttransfer may take severalminutes more.

Next priorities: LLaVA SLAKE short complete, PathVQA6719(old31.6572 vsOPERA33.1439), MMMU10500(old20.4286 vsMedRAG22.10). PMC33430/Omni88995 later per readiness/cost; no fabricated oldscore forpartialruns. Metadata reports/llava-test-v1/dataset-priority.json. Certifiedmatrix localbaseline file directories may be stale/missing (MMMU fullgreedy absentlocally); useexistingcertifiedaggregate, do notmistake32-case gate forfullbaseline. MCQ manifestprompts mustretainchoices. No externalSOTA claim fromlocaltables.

SSH host: merit-runner@172.17.0.1 key/root/.ssh/merit_host_gpu0_ed25519. Cloud root@connect.weste.seetacloud.com port51493 key/root/.ssh/merit_5090_ed25519. No new agents. Read research-ops skill alreadyapplied. Finish jobs, updatepairedreports, commit/pushcoherentbatch. Actualresults only; no silentpartialfallbacks.
