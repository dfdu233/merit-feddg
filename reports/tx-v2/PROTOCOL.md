# MERIT-Tx v2 GPU validation

Requested implementation: `1845507872872e969aa63298daef686e0cff4489`, isolated checkout `/home/dbw/merit-tx-v2`. Prior Adaptive BARD results are preserved. Priority is PathVQA, where both LLaVA-Med and HuatuoGPT underperformed historical Greedy. No target qualification or target-label selection is permitted.

Source panel: 256 PathVQA official TRAIN images, selected by fixed image hash without correctness filtering, plus an existing 48-case external PathoROB four-medical-center development panel. The latter is development data here, not an untouched target evaluation. PathVQA TRAIN images are checked against target TEST file and decoded-pixel hashes; zero overlapping rows found. Separate references are never supplied to candidate generation. One question per selected PathVQA image. PathoROB split grouping uses center plus slide ID.

Qualification: 224 cases / 216 image-or-slide groups. Independent canary: 80 cases / 76 groups. No group crosses the split. PathVQA lacks patient identifiers: its counterfactual controls are different image groups, not verified different patients. Do not overstate patient-level independence.

Both models generate frozen Generalist answers and Adaptive BARD proposals using the submitted MERIT-Tx configuration, 256-token limit, and no target-specific tuning. This source protocol differs from the old 64-token LLaVA target experiment and must not be compared as if the sample sets or generation budgets matched. Shared native requests use frozen image-only LLaVA routing; the same route/cache is reused across the two receiver models. Initial route outputs are carried into qualification/canary manifests. The proposal still uses actual frozen native evidence and spatial transport; references do not affect proposals.

Hardware: 5090 pair initially generated all304 Huatuo baselines; three4090 pairs generated all304 LLaVA baselines. HostGPU1 prepared the native cache and handled Huatuo proposal shard0/2 and LLaVA proposal shard0/6. Other source proposal shards run on the clouds. HostGPU0 is excluded. Lossless native-cache transfer overlaps proposal execution, validating each complete request before consumption. Full raw evidence remains retained. Local compact exports may omit duplicate payloads only while retaining original raw artifacts/cache provenance.

Qualification cards are fitted separately for each proposal model, with the v2 schema and submitted thresholds unchanged: at least two actual support/veto domains; positive utility lower bound; support harm upper bound at most0.25; specificity lower bound at least0.5; veto precision lower bound at least0.5. Fitter/canary use the submitted default token-F1 metric. Any additional benchmark-style CE/OE score will be identified separately. Cards are frozen before independent-canary decisions; canary references are read only after outputs are frozen. Advancement requires nonzero accepted coverage, bounded harm, and positive gain over immutable Generalist; zero-change equality is not success.

Execution fixes, not performance tuning:

- Submitted disabled vector-gate setting used probe_tokens256, but its config validator permits at most64. Set the inactive parameter to64; actual generation limit remains256.
- Huatuo's submitted budget check rejected min_new_tokens=max_new_tokens=1, breaking the ordinary first-token receiver path. Restore the valid equality boundary; add a regression. It does not change256-token generation.
- Preserve source domain/group metadata in native-cache preparation instead of hardcoding target metadata.
- Source scripts accept an exact per-case proposer map, preserving actual variable proposal provenance rather than applying an unnecessarily broad global union.
- Qualification explicitly records/skips cells with fewer than four matched controls, rather than crashing or fabricating controls. The observed fit panel has only three CXR cases.
- Verification config assigns the same fault group to BiomedCLIP anatomy and claim-verifier interfaces (and CONCH aliases); separate interface IDs must not authorize self-validation. Frozen proposal generation remains unchanged.
- Shared local dependencies and host directory ownership were repaired. Early failures produced no candidate results; all logs retained.

Validation:54 targeted tests pass. Real CONCH, PLIP, and BiomedCLIP scored a real source transaction plus four controls each, without references. New-branch Huatuo SDPA cache parity was checked on two real source examples: ordinary/spatial branches, four prefix positions each, score vectors exactly equal; host/cloud original token sequences also equal. This is a bounded runtime check, not an accuracy claim or proof of all-case hardware identity.

Current measurements and final promotion decision will be recorded separately after qualification and canary complete. Previously inspected target benchmarks cannot be described as fresh untouched confirmation.
