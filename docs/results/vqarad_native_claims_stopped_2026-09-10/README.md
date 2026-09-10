# VQA-RAD native-claims stopped snapshot

This is the complete Git-tracked snapshot of the `native_claims` evaluation at
the point it was stopped by request on 2026-09-10. It is **not** a completed
451-case result and must not be reported as one.

The six arms jointly completed 362/451 official-test questions (206 closed and
156 open). Available cache counts at stop time were 364 `generalist`, 364
`semantic_all`, 364 `entry_all`, 364 `entry_filtered`, 363 `hybrid_all`, and
362 `hybrid_gate`. The 362-case common prefix is completion-order selected, not
a prespecified subset, so its metrics are diagnostic only.

## Files

- `per-case-results.jsonl`: all 451 manifest rows in official order. Every row
  contains the question, answer type, image path/hash, offline reference,
  image-only routing output, availability flag, and every available arm output.
  Outputs retain answer text, token IDs, generation configuration, controller
  and expert events, native expert values, entry checks, evidence transport,
  local-removal Gate audits, perturbation geometry, and timings. Missing arms
  are explicit `null` values.
- `evaluation-summary.json`: fixed-evaluator metrics on the 362 six-arm-common
  cases, paired ablations, transport and Gate counts, baseline identity check,
  and every strict good/bad outcome with its expert/Gate trace.
- `snapshot-provenance.json`: run identity, commit/config inputs, raw log
  hashes, and the exact payload-omission rule.
- `GATE_FAILURE_AND_CONFIDENCE_RESEARCH_2026-09-10.md`: mechanism diagnosis,
  representative cases, literature/code comparison, and a source-only
  improvement protocol. No proposed change was implemented in this snapshot.
- `SHA256SUMS`: integrity hashes for the published files.

## Result at stop time

On the 362 common cases, strict mixed scores were 0.5106 Generalist, 0.5238
`semantic_all`, 0.4797 `entry_all`, 0.4797 `entry_filtered`, 0.4769
`hybrid_all`, and 0.4845 `hybrid_gate`. Gate recovered 0.0076 over
`hybrid_all`, but remained 0.0261 below Generalist. Mean recorded engine time
was 0.676 s for Generalist and 3.140 s for `hybrid_gate` (4.65x).

The regenerated Generalist is exactly identical to the dedicated ANCHOR
Greedy baseline on every available case (364/364), so the observed delta is not
caused by baseline drift.

## Deliberate payload omission

The raw local run is about 14 GB because identical compressed soft masks are
embedded repeatedly. The Git bundle omits only each
`float32-zlib-base64.data` string. In place, it records
`data_omitted_from_git: true`, the exact character count, and SHA-256 of that
string. Shape, encoding, labels, native scores, transform metadata, provenance,
Gate decisions and all non-mask fields remain. Raw mask bytes, model weights,
images and expert caches remain in the ignored local run at:

`runs/matched-native-claims-anchor/edcf48420dd394d764533ec9cc24be2e555887d916adc7fa29a9d74181ac603d`

References were joined only after generation for offline evaluation. No target
answer was available to routing, experts, Gate, evidence transport, or answer
generation.

