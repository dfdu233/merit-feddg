# Active Huatuo spatial validation

User scope: cloud GPU0+GPU1 (5090), host GPU1 only. Host GPU0 untouched.

Worktree based on 5b15b16; experiment branch experiments/huatuo-adaptive-bard-validation-v1. Method frozen. Runtime repairs and preflight evidence are in reports/huatuo-spatial-v1/METHOD.md.

Cloud eager attention passes historical 4/13/92-token parity plus native spatial canary. Host Torch 2.0.1 fails one token of the cloud-origin 92-token record, including under the original adapter; do not pool its efficacy outputs. Isolated host Torch2.8 environment preparation is underway at /home/dbw/venvs/huatuo-bard-torch28.

Cloud run root: /home/dbw/merit-feddg-huatuo-spatial/runs/huatuo-spatial-v1. worker-0 is processing long vqarad-train-554. worker-1 completed its 16 cases. supplemental on GPU1 processes the seven remaining even-index VQA cases in a separate supplemental-results directory. Never count duplicate cases. workers have 200k/200k/100k forward limits; method/config identical.

SLAKE 16/16 complete: mixed CE/OE score Generalist=BARD 65.625%, joint/mean/median 62.5%. No final VQA claim yet. No full TEST run started. Offline references stay outside receiver inputs.

Next: finish all 32, collect slim analysis exports, paired harm/rescue and stratified stress, report runtime costs/limits, commit results. Do not replace incomplete outputs with baseline guesses.
