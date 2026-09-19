#!/usr/bin/env bash
# Public, immutable HF artifacts. Preserve HTTP partials and verify official LFS SHA256.
set -euo pipefail
critic_root=/home/dbw/models/llava-critic-7b-498f2d7
revision=498f2d719b83e50e48787c6958afe7100503c23f
log_root=/home/dbw/merit-feddg-huatuo-critic/runs/critic-preparation
mkdir -p "$critic_root" "$log_root"
digests=(a3c2dac0e595e5b40072a604ff401df2e781d702a42d3175a7e6737aaa44d437 0b30c8b75a6c81a454f0e3eae6c4106d7a6cd79b17b55f24c57c440cd6ab4703 fd3b0171bd7dca40dc4cff06524dc5b9979ef0f4905a1084be0768ae1c8cd316 6611121dc8a845f7b85d003b76bc2cf59ad4f2938099e23ce82072d69bacb5a6)
pids=()
for index in 0 1 2 3; do
    name="model-0000$((index+1))-of-00004.safetensors"
    digest="${digests[$index]}"
    # Only our stopped HF transfer owns these unique digest-specific partials.
    shopt -s nullglob
    partials=("$critic_root/.cache/huggingface/download/"*".$digest.incomplete")
    if [[ ! -e "$critic_root/$name" && ${#partials[@]} -eq 1 ]]; then
        mv -n -- "${partials[0]}" "$critic_root/$name"
    fi
    aria2c --continue=true --max-connection-per-server=8 --split=8 --min-split-size=10M \
        --file-allocation=none --auto-file-renaming=false --check-integrity=true \
        --checksum="sha-256=$digest" --summary-interval=30 --console-log-level=warn \
        --dir="$critic_root" --out="$name" \
        "https://huggingface.co/lmms-lab/llava-critic-7b/resolve/$revision/$name" \
        > "$log_root/aria2-$index.log" 2>&1 &
    pids+=("$!")
done
failed=0
for download_pid in "${pids[@]}"; do
    wait "$download_pid" || failed=1
done
exit "$failed"
