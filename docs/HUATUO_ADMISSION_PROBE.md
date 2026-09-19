# Huatuo context admission pilot

This experiment targets the Huatuo degradation, not the previously completed
LLaVA-Med TRAIN run. Existing full-test artifacts show VQA-RAD generalist62.82%
vs MERIT57.16%, and SLAKE54.69% vs54.48%. These historical aggregate scores are
motivation, not selection criteria for individual TRAIN examples.

Hypothesis: a separate same-model question-conditioned usefulness decision can
exclude out-of-capability specialist packets without contaminating the final
answer context. The prior LLaVA experiment accepted everything; that result is
not assumed to generalize to Huatuo. Reuse the already frozen relevance/scope
prompts, grounded in Self-RAG's relevance annotation, S2A context separation and
the tool-review precedent in MedAgent-Pro. See CONTEXT_ADMISSION_RESEARCH.md.
No new novelty claim, training, numerical admission threshold or yes/no-specific
gate rule. Original formal answer instructions are retained for every arm.

Fixed initial sample: first4 IDs of the existing24 TRAIN-only-image probe,
selected before this experiment and without score-based filtering. Pixel identity
and test-image exclusion are checked. The generation input contains no reference
answers. Huatuo recomputes its own original image-only routing; no LLaVA route or
answer is reused. Expanded expert registry and native all_evidence acquisition
remain unchanged. Original formal1024-token answer budget, native4096 input
policy and Huatuo preprocessing/repetition settings are retained.

Arms: generalist, original compact MERIT, relevance, scope. Gates see the original
image/question, specialist definition and unchanged packet; no generalist answer
or reference. Only admitted packets enter a fresh answer context. All/none views
reuse exact controls and are explicitly logged, not counted as new candidates.
First compact answer must reproduce exactly in text and token IDs. Runtime,
budget and delivery failures stop the affected experiment, not medical evidence.

Independent worktree `/home/dbw/merit-feddg-huatuo-admission`, branch
`experiments/huatuo-context-admission-v1`, based on `8d1742a`. Existing dirty
expert-coverage worktree is read-only: pinned imports reuse its verified native
Huatuo adapter and runner plus the unchanged formal MERIT core. Every stage
checks hashes of the dependency files/config and image identities. Checkpoints,
environment and data are reused; no upgrades/downloads or other-job termination.

GPU0 host UUID `GPU-809e1541-5fe0-e1a6-d360-d0ea647e9023`; GPU1 had insufficient
conservative headroom for another Huatuo worker. Background host tmux runs
route -> experts -> admission. Root `runs/huatuo-train-admission-v3`; v1/v2 are
preflight-only snapshots. Initial expert stage stopped on a missing relative
artifacts link; non-overwriting links to existing artifacts/upstream were added,
then the same frozen run resumed. Original failure log is preserved.

Run with the existing `/home/dbw/.runtime/miniconda3/envs/huatuo/bin/python`:

```bash
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export PYTHONPATH=/home/dbw/ANCHOR:/home/dbw/merit-feddg-expert-coverage/scripts
export HF_HOME=/home/dbw/ANCHOR/hf_cache HF_HUB_CACHE=/home/dbw/ANCHOR/hf_cache/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python scripts/run_huatuo_admission_probe.py --stage check --dataset vqa_rad --gpu-uuid GPU-809e1541-5fe0-e1a6-d360-d0ea647e9023 --output runs/huatuo-train-admission-v3
# Same arguments with --stage route, experts, admission in sequence.
```

Only aggregate outcomes may be committed. Do not copy raw questions, answers,
images, expert caches or tokens into public reports. This four-case plumbing
probe cannot establish efficacy, SOTA or general safety; no score-conditioned
prompt tuning is permitted. Expand TRAIN only after inspecting actual decisions,
delivery and candidates. SLAKE remains the next target, not already evaluated.
