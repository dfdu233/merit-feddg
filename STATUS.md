# Current research status

Scope: user requests sustained staged experiments toward a defensible main graph-related innovation; heterogeneous tool substrate follows MedRAX. No innovation confirmed yet. Gate/expert fusion remains frozen.

- Worktree `/home/dbw/merit-feddg-relational-pilot`, branch `experiments/medrax-relational-pilot-20260916`.
- Four stages complete: SLAKE predicted-graph24, acquisition13, MedSG correspondence28, source-identity diagnostic28. Latest evidence and limitations: [MULTIMODAL_GRAPH_RESEARCH](docs/MULTIMODAL_GRAPH_RESEARCH.md).
- No stable graph benefit. Correct source-name oracle also fails to recover target localization. Need adequate grounding before identifying relation-transport effects.
- Current Vector: does a capable task-specialist target tool expose relation-specific headroom under identical source evidence?
- Active Huatuo-Qwen replication: `scripts/run_correspondence_huatuo.py run --limit 1`, session7756. Inspect canary then run remaining27; evaluate only after completion. Source descriptions copied byte-identically from Qwen v2.
- MedSeq-Grounder snapshot download active, Python PID3049994, session38623; revision7c18e944897e8fdb86e93dcf919ad78517330df4, destination `/home/dbw/models/MedSeq-Grounder`. Network slow but making progress. After complete run `scripts/run_correspondence_specialist.py run --limit 1`, inspect and finish then evaluate. Protocol and manifest already frozen. Do not score incomplete download as model failure.
- Native coordinate/schema v1 failure is preserved; v2 corrects processor-resized coordinates, all28 cases/168boxes valid. Original and changed task/split semantics explicitly documented.
- Host GPU1 maps container0 UUID GPU-3846413a-4238-d307-b1f3-10c2dfbe002c. Other user's Huatuo process uses ~25GB; do not stop it. Our sequential7B jobs fit remaining space; recheck before launch.
- Runtime: `/home/dbw/.venvs/qwen25vl-v2/bin/python`, CUDA_VISIBLE_DEVICES=0, HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1, TOKENIZERS_PARALLELISM=false, OMP_NUM_THREADS=4.
- Source audits: `/home/dbw/research-notes/medrax-composition-20260916/`, especially `relation_transport_audit.md`. Observation-conditioned relation compatibility is a question, not verified novelty. Task7 cannot establish latent relation truth versus unobservability without additional controlled evidence.
