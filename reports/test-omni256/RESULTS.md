# Huatuo OmniMedVQA fixed verifier pilot

Reproduces thread01a0bc6e-126d-7cd3-80b1-8cfb06235b25's unqualified exploratory filter on branch experiments/merit-tx-v3-host-validation. 256 SHA256-ID-ordered rows selected from the frozen completed Huatuo pool, not outcome-selected and not full88995. Historical raw Baseline/BARD reused; native verifier ran on hostGPU1 using unchanged run_test_verifier_ablation.py, four fixed source controls and unchanged route. Reference labels loaded only by the scorer. All256 exact IDs completed.

| Arm | Correct/256 | Accuracy % | Accepted |
|---|---:|---:|---:|
| Baseline |184|71.8750|0|
| Original BARD |184|71.8750|6|
| BiomedCLIP filter |183|71.4844|1|
| CONCH filter |184|71.8750|0|
| PLIP filter |184|71.8750|0|
| Joint filter |183|71.4844|1|

Six textual changes include four unambiguously-letter-only mapping failures, one all-verifiers-outside-scope case, and one actual BiomedCLIP comparison. The accepted RadImageNet_10470 change breaks a correct B into A: real margin -0.88933, control median -1.04280, differential +0.15347. Thus the current difference-only rule accepts despite the real image preferring the baseline. This is faithful to the existing rule, not an implementation exception. No rule or threshold was adjusted using this test result.

Original BARD fixes one and breaks one baseline answer; filtering retains the harm but not the benefit in this small pilot. Does not establish population degradation or superiority. Most cases require no verifier inference, so pilot wall time cannot represent full-method speed. Results and effects: results.json; frozen inputs/source paths: runs/test-omni256. Existing original benchmark generation and native cache production were not stopped. No formal source-qualified v3 claim.
