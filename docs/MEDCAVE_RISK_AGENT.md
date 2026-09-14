# MedCAVE risk-controlled specialist agent

This branch adds a backward-compatible research path on top of the frozen Med-DEFER experiments. Existing `MedDeferEngine` behavior is intentionally unchanged.

## Motivation

The stopped VQA-RAD native-claims study showed that evidence dependence and local-removal gain are not correctness or utility. A specialist can strongly influence the VLM and still turn a correct baseline answer into an incorrect one. The new path therefore treats specialist use as an **intervention-risk** problem rather than a saliency/reliance problem.

## New modules

### `merit_feddg/intervention_risk.py`

Defines separate risk axes:

- capability/applicability mismatch;
- source reliability;
- pre-call and post-call OOD;
- cross-evidence conflict;
- answer instability;
- semantic coverage;
- visual consistency;
- expert uncertainty.

The default aggregation is the maximum risk component. This is intentionally non-compensatory: a severe capability mismatch cannot be cancelled by high confidence on another axis.

`fit_source_risk_thresholds()` performs source-only calibration over frozen intervention outcomes. It selects the widest score threshold whose one-sided Wilson upper bound on the harmful-intervention rate remains under the requested risk budget. It fits no neural parameters and fails closed when calibration support is insufficient.

The calibrated controller exposes three actions:

- `ACCEPT`: permit the specialist intervention;
- `ACQUIRE`: request independent evidence before changing the baseline;
- `FALLBACK`: keep the untouched generalist output.

This is a source-calibrated risk-control prototype, not yet a theorem-backed conformal guarantee under arbitrary domain shift. Target labels must never be used to fit thresholds.

### `merit_feddg/sequential_agent.py`

Adds a deterministic sequential specialist loop:

1. filter registered experts by modality and requested capability;
2. require source qualification and a domain signal;
3. rank candidates by source trust, expected gain and cost;
4. call one specialist lazily;
5. build intervention-risk signals;
6. `ACCEPT`, `ACQUIRE` another independent expert, or `FALLBACK`;
7. if the acquisition budget expires, preserve the baseline exactly.

The LLM is not asked to select checkpoints by name. A planner may request a capability, while the runtime chooses a qualified model.

## Integration with existing code

The implementation reuses rather than replaces:

- `ClaimRequest`;
- `NativeEvidence`;
- `ExpertCard`;
- `DomainSignal` / `DomainTrustCalibrator`;
- `LazyExpertPool`;
- existing evidence bridges and specialist adapters.

A first experiment can therefore wrap the current source-qualified experts without modifying their model code.

## Required next wiring for a paper experiment

The current branch provides the control kernel and CPU tests. The next experiment-facing wiring should provide claim-specific values in `NativeEvidence.provenance`:

```text
coverage
conflict
instability
visual_consistency
cross_expert_conflict
```

Recommended definitions:

- `coverage`: whether every requested semantic proposition is explicitly linked by the evidence bridge;
- `conflict`: support/contradiction disagreement with the current evidence state;
- `instability`: response sensitivity to matched control evidence rather than evidence correctness;
- `visual_consistency`: three-view consistency over original / expert-rendered / counterfactual-rendered images;
- `cross_expert_conflict`: independent specialist disagreement on the same claim.

These diagnostics should remain features of intervention risk, not be individually interpreted as correctness probabilities.

## Evaluation contract

For every source calibration and target evaluation run, record:

- base correctness/utility;
- guided correctness/utility;
- `rescue`, `neutral`, or `harm` intervention label after predictions are frozen;
- risk score and dominant risk component;
- ACCEPT / ACQUIRE / FALLBACK action;
- selected specialist sequence;
- call count and latency;
- OOD and qualification provenance.

Primary safety metrics should include harmful-accept rate, rescue-minus-harm net benefit, selective risk/coverage, tool calls and latency in addition to final task accuracy.

## Unseen-specialist experiment

A convincing plug-and-play experiment should freeze the agent policy, then register a specialist not present during agent development. The new specialist may bring its own source qualification/risk artifact, but the agent itself must not be fine-tuned. Measure tool discovery, invocation success, harmful-accept rate and net task gain.
