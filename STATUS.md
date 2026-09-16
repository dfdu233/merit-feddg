# Current status

Graph representation mechanism pilot completed on 2026-09-16.

- Worktree: `/home/dbw/merit-feddg-graph-equivalence`.
- Branch: `experiments/graph-equivalence-20260916`.
- Experiment: `graph-equivalence-dev-20260916-v1`; protocol `1653769`; code `f0b7c8e`.
- 64 synthetic development fixtures, 192 schema cases, all semantic/control checks passed.
- Raw PPR Top-6 changes: full reification 27/64, partial 20/64. Canonicalization eliminates all changes; semantic-hop accounting eliminates reachability differences.
- Decision: pre-registered stop rule fired. No GPU or real-data escalation. No active job remains.
- Result, limitations, and re-vector rules: [GRAPH_EQUIVALENCE_RESULTS](docs/GRAPH_EQUIVALENCE_RESULTS.md).
- Next action: return to literature affinity map before selecting another scientific hypothesis; do not scale relation-reification experiments.
- Prior project status is preserved at base commit `adeda00589ba67d97adc634bf8f18deab20b9ac2`; other worktrees are untouched.
