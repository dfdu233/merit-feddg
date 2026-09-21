# Cross-session deployment review

Read-only audit at 2026-09-21 09:21 UTC of benchmark thread 01a05d29-9d0b-7121-8a53-488b8cd1a125, local artifacts and all three additional servers.

## Confirmed progress

Huatuo VQA-RAD451 and SLAKE2094 complete. LLaVA SLAKE2094 complete. Snapshot LLaVA Path: cloud256+700, host134 =1090/6719; MMMU489/10500. Counts require per-case provenance and exclude in-flight cases. Path retrieval6551/6719, MMMU retrieval7901/10500 (last progress lines, lower bounds).

All six additional GPUs (oldcloud40297,44450,42865) idle during preparation. Live tar/gzip receivers observed, no Adaptive BARD receiver/probe process or parity artifact found. This is staging, not formal evaluation. Cache disk snapshots63MB/45MB/35MB; no completeness claim.

## Adaptation review

Across all three nodes, SHA256 matches current files for bard.py, bard_protocol.py, generalist_factory.py, run_adaptive_bard_canary.py, huatuo_cached_receiver.py, config.yaml and full PathVQA manifest. Therefore no detected algorithm/config/manifest substitution in these inspected files. Model weights and full native cache transfer were not independently certified in this audit.

Existing Path ranges [0,256),[256,1536),[1536,3072); reserved additional [3072,4288),[4288,5504),[5504,6719). Exact disjoint partition of6719. Original session must not schedule3072+ while benchmark session owns it. New servers each have two4090/4090D GPUs; recorded allocation is currently server-level, not proof both GPU workers launched.

Additional servers existing Huatuo env torch2.0.1+cu117/transformers4.37.2, versus5090 torch2.8+cu128. Code equality does not prove numeric/generation parity. Require bounded same-input real LLaVA ordinary+spatial cached-vs-production check and cross-node output comparison before merging formal results; Huatuo needs separate architecture validation. Existing host LLaVA checks do not certify these new nodes.

Runner validates expert config/cache identity, exact scheduled keys for bounded incomplete caches, per-image SHA and frozen benchmark prompts. Reusing exact native expert outputs is appropriate; preserve full masks and MCQ options. Last PathMedCPT progress6551 means tail transfer must be refreshed and all6719keys verified before admitting last shard. File transfer processes alone are insufficient evidence.

## Scope still pending

Only LLaVA PathVQA deployment is concretely staged on additional nodes in this snapshot. No evidence of completed all-model/all-dataset queue deployment, Huatuo Path/PMC/MMMU/Omni expansion, or report-specific MIMIC adaptation. Do not reuse VQA64-token scoring contract for report evaluation without its own frozen report protocol. Keep Adaptive BARD results distinct from historical MERIT compact-rows. LLaVA VQA-RAD remains excluded under original user instruction. Historical baselines only; no baseline regeneration.

No other-session workers, transfers, configuration or method were changed during this review.
