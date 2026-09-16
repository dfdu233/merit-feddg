# Current status

MedRAX reuse novelty and workload audit completed on 2026-09-16.

- Worktree: `/home/dbw/merit-feddg-reuse-audit`.
- Branch: `research/medrax-reuse-audit-20260916`; base `21003dc`.
- Audit: `medrax-reuse-structure-20260916-v1`.
- Fixed ChestAgentBench metadata: 2500 questions, 609 cases, 4629 image references, 1346 distinct references; zero exact case/image-set/question repeats.
- These are metadata counts, not observed tool calls or speedups. No gold answers used in analysis; no models or GPU jobs ran.
- Tool signatures and implementation show strong simple-cache baselines. APC, KGCache, TVCache, and SmartCache constrain novelty.
- Report and reproducible audit: [MEDRAX_REUSE_AUDIT](docs/MEDRAX_REUSE_AUDIT.md).
- Current risk: whether real trajectories retain a meaningful evidence-composition problem after exact and full-image-feature caching. Not yet verified; no graph method selected.
- Next discriminating step: inspect matched real tool trajectories with the same actor/tools, if pursuing this candidate; do not claim graph novelty or train before residual opportunity is established.
- Prior graph-reification direction remains stopped. Historical results are preserved in their branches.
- No active job remains.
