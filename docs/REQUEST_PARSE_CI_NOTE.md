# CI status and scope (2026-09-16)

Inspection of PR #11 at fe95743, run 35106768312:
- `open-generation-cpu`: 181 existing tests passed; global Ruff failed.
- `capability-contract-cpu`: tests and scoped lint passed.
- `test`: collection stopped in five historical torch-dependent test files because
  this job installs only .[dev] and does not install torch. No new parser test
  failure was reported; collection never reached the full test run.

The three introduced I001 import-layout findings (two in the new runner, one in
its new tests) have been corrected without behavioral changes. The remaining
reported lint findings refer to older files; no old experiment scripts, test
semantics, thresholds or workflow configuration are modified in this Vector.
A follow-up CI run must verify the corrected formatting. Full CI is NOT claimed
green, and the base's 936 tests are not a new full-regression result.

Local isolated test rerun after the import-only correction: 32 passed, 1 skipped
(real upstream bridge unavailable locally). Compile and CLI help passed. No real
medical checkpoint loaded; no parser fidelity or clinical efficacy measured.
