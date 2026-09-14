# MedCAVE: calibration-only intervention control

PR #4 now has an opt-in free-answer execution path. The VLM and specialists stay
frozen. Statistical validation requires source data: **calibration-only**, not
data-free, conformal coverage or arbitrary-hospital safety. This remains Draft:
no valid source-cal certificate or medical benefit has been established.

## Executed inference path

`medcave_run` reuses CapabilityPool, SharedExpertPool, CapabilityRequest,
validate_result, NativeSession and LLaVA-Med. It does not route free VQA through
the historical closed-set logits engine.

1. Save the original answer, token IDs and generation configuration.
2. Filter tools by modality, capability, question attribute and request contract;
   require positive prespecified scheduling utility. Example priorities are not
   estimates of clinical gain.
3. Call a tool, validate result identity/capability/scope, accumulate its native
   evidence, and generate a fresh candidate from the original image.
4. Verify the exact candidate and accumulated evidence. ACCEPT returns that
   candidate directly. ACQUIRE retains the first observation and re-generates
   and re-verifies the joint candidate. FALLBACK returns the saved baseline.
5. Budgets: at most two expert requests and two candidates; zero visual probes
   in v1. Cache hits consume the logical request budget. Model aliases sharing
   checkpoint provenance cannot count as additional model sources. Different
   sources are not assumed to have independent errors.

The existing semantic/compact channel retains the original image and native
spatial provenance. A candidate cannot be accepted if any acquired packet is
missing from its tokenizer delivery audit. No new image layout or invented ROI.

## Correctness changes

- Missing risk values are `None`; invalid numbers/ranges are rejected.
- Structural failure is independent of aggregation and threshold.
- Historical Wilson threshold search is source-dev selection only and always
  returns `acceptance_disabled=True`. Empty empirical harm is undefined.
- Zero max_bias_norm forbids logits changes. Nonpositive utility forbids a call.
- LazyExpertPool keys include the full request/card and counts provider attempts.
- The legacy sequential logits path cannot certify joint free-text generation.
- Free-answer runtime counts logical/actual/cache/candidate calls, records
  exceptions, checks candidate/state identity, and exactly preserves fallback.

## Signals: implemented adapters and honest gaps

CandidateSignalBuilder uses EvidenceBridgeRegistry only for explicitly mapped
candidate propositions. Liver localization does not imply normal liver;
localization failure does not imply disease absence. Retrieved cases are not
direct patient evidence. Native confidence remains a raw score.

SourceArtifactVerifier loads optional `qualification_path` entries through the
existing QualificationArtifact and QualificationGate. It requires matching
model/adapter, modality, capability and task. Candidate-bound adapter feature
payloads can use the existing pre/post OOD methods. Missing artifacts/features
remain unknown. Explicit assertions are compared only within the same image,
proposition, location and polarity. Disagreement with the generalist is not
itself an error. Classifiers without spatial checks receive not_applicable for
that check, rather than an invented pass or failure.

The configured real CheXagent adapter has **no validated mapping from its prose
to candidate propositions**, matching source qualification artifact or usable
pre/post OOD and matched-control sensitivity features. These remain unknown.
No visual probe is claimed to have run. These are concrete blockers to a useful
certified ACCEPT path; CPU mock-verifier ACCEPT tests do not resolve them.

## Two-stage source validation

`source-dev` freezes the supplied full configuration into policy.json and runs
trajectories, optionally with offline descriptive statistics. It does not
automatically optimize thresholds. `source-cal` requires that exact binding
and separate source groups/images, executes the complete policy and prospective
stopping decision, then opens reference labels.

Certification uses the smallest case ID per declared independent group, selected
without consulting acceptance/outcome. All rows also have descriptive metrics.
Patient/slide/study IDs are preferred; grouping by image cannot establish patient
independence when IDs are missing.

A fixed one-sided exact Clopper-Pearson bound checks P(harm | policy accepted).
Defaults: at least 60 accepted independent groups, risk budget .05, alpha .05.
The binomial interpretation assumes independently, identically sampled source
groups. No source-cal threshold search or repeated rule adjustment is performed.
Insufficient support, failed validation and incomplete trajectories disable
ACCEPT. Zero acceptances means conditional harmful-accept rate is null.
P(harm and accepted) and final task quality are separate metrics; closed exact
correctness and open continuous token F1 are reported separately.

Bindings cover policy/configuration, code, risk schema, generalist/vision/source
provenance, prompt/decoder, expert versions, qualification file bytes, budgets,
grouping rule and evaluator version. Changed bindings are rejected. Digests
detect accidental mutation; local files/adapters remain a trusted boundary,
not a signed adversarial attestation or a target-domain guarantee.

No independent previously unexposed source-cal manifest plus qualification
package was established in this execution. Existing VQA-RAD TRAIN data had
already appeared in development and was used only for engineering smoke.
Official tests were not split or run.

## Actual executable CLI

```bash
cd /home/dbw/merit-feddg-medcave-risk
export PYTHONPATH="$PWD"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
PY=/home/dbw/merit-feddg/.venv/bin/python
MANIFEST=/home/dbw/merit-feddg/runs/vqarad-official-protocol-v3/data/train/manifest.jsonl
ARTIFACTS=/home/dbw/merit-feddg/artifacts

"$PY" -m merit_feddg.medcave_run --stage dry-run \
  --config configs/medcave_source.yaml --manifest "$MANIFEST" \
  --artifacts "$ARTIFACTS" --output runs/medcave-dry-run

"$PY" -m merit_feddg.medcave_run --stage source-smoke --limit 2 \
  --config configs/medcave_source.yaml --manifest "$MANIFEST" \
  --artifacts "$ARTIFACTS" --output runs/medcave-source-smoke \
  --references /home/dbw/merit-feddg/runs/vqarad-official-protocol-v3/data/train/references.json

# Set SMOKE_ROOT to the output_root printed above.
"$PY" -m merit_feddg.medcave_run --stage replay \
  --config configs/medcave_source.yaml --manifest "$MANIFEST" \
  --artifacts "$ARTIFACTS" --replay-run "$SMOKE_ROOT" --output runs/medcave-replay
```

Replay validates exact complete candidate/state trajectories. Old single-tool
arms and missing joint candidates return replay_unavailable. It never invents
new joint answers or claims new-method accuracy.

The following implemented commands require additional audited source paths;
those artifacts are **not claimed to exist** on this server. Set DEV_MANIFEST,
DEV_REFERENCES, CAL_MANIFEST and CAL_REFERENCES to independent source inputs,
POLICY to source-dev output_root/policy.json, CERTIFICATE to source-cal
output_root/certificate.json, and FULL_EVAL_MANIFEST/EVAL_REFERENCES to the
original frozen full evaluation inputs.

```bash
"$PY" -m merit_feddg.medcave_run --stage source-dev \
  --config configs/medcave_source.yaml --manifest "$DEV_MANIFEST" \
  --references "$DEV_REFERENCES" --artifacts "$ARTIFACTS" --output runs/medcave-dev

"$PY" -m merit_feddg.medcave_run --stage source-cal \
  --config configs/medcave_source.yaml --policy "$POLICY" \
  --manifest "$CAL_MANIFEST" --references "$CAL_REFERENCES" \
  --artifacts "$ARTIFACTS" --output runs/medcave-cal

"$PY" -m merit_feddg.medcave_run --stage evaluate \
  --config configs/medcave_source.yaml --policy "$POLICY" --certificate "$CERTIFICATE" \
  --manifest "$FULL_EVAL_MANIFEST" --references "$EVAL_REFERENCES" \
  --artifacts "$ARTIFACTS" --output runs/medcave-evaluation
```

Formal evaluation has no subset limit. References are read after generation.
The example evaluator is a lexical VQA diagnostic, not an ANCHOR clinical scorer
or report-factuality evaluator. Changing evaluator or prompts invalidates the
binding and requires independent validation again.

## Verification status (2026-09-14)

- Initial PR df7edce: six risk tests passed, despite the untested safety holes.
- Full delivery regression: 740 passed. Repository ruff, shell syntax and
  whitespace checks passed. No type-check job is configured.
- All six CLI stages have CPU fake-backend integration tests; these are contract
  tests, not medical or statistical efficacy evidence.
- Real two-row TRAIN smoke: one CheXagent call, one 21-token candidate with
  evidence delivered, two exact baseline fallbacks. Acceptance 0/2 and conditional
  harmful-accept rate undefined. No clinical gain or calibrated safety claim.
- Failed startup/cache-API attempts remain locally recorded. Server compatibility
  fixes are a separate commit; shared dependencies/checkpoints were not changed.

See the [execution report](results/medcave_pr4_2026-09-14/README.md). No patient
images, raw case outputs, weights or credentials are included in Git.
