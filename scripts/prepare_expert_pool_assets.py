"""Audit/download literature-selected MERIT-Tx expert assets.

Open checkpoints can be downloaded explicitly. Gated checkpoints are never
silently fetched; the script prints the required Hugging Face command after the
user has accepted the upstream model terms.
"""
from __future__ import annotations

import argparse
from pathlib import Path

OPEN_MODELS = {
    "plip": ("vinid/plip", "artifacts/models/vinid--plip"),
    "medsam": ("wanglab/medsam-vit-base", "artifacts/models/wanglab--medsam-vit-base"),
}
GATED_MODELS = {
    "conch": ("MahmoodLab/CONCH", "artifacts/models/MahmoodLab--CONCH"),
    "maira2": ("microsoft/maira-2", "artifacts/models/microsoft--maira-2"),
    "medsiglip": ("google/medsiglip-448", "artifacts/models/google--medsiglip-448"),
}
RESEARCH_MODELS = {
    # These are literature-selected coverage candidates, not active MERIT-Tx
    # experts. Downloading them does not grant commit authority.
    "musk": ("xiangjx/musk", "artifacts/models/xiangjx--musk", "gated"),
    "merlin": ("stanfordmimi/Merlin", "artifacts/models/stanfordmimi--Merlin", "open"),
}


def present(path):
    value = Path(path)
    return value.is_dir() and any(value.iterdir())


def commands():
    rows = []
    for name, (repo, target) in OPEN_MODELS.items():
        rows.append(f"# {name}: open checkpoint")
        rows.append(f"hf download {repo} --local-dir {target}")
    for name, (repo, target) in GATED_MODELS.items():
        rows.append(f"# {name}: gated; accept upstream terms first")
        rows.append(f"hf download {repo} --local-dir {target}")
    rows.append("# Literature-selected research candidates; no runtime authority yet")
    for name, (repo, target, access) in RESEARCH_MODELS.items():
        label = "gated; accept upstream terms first" if access == "gated" else "open"
        rows.append(f"# {name}: {label}; adapter/source qualification still required")
        rows.append(f"hf download {repo} --local-dir {target}")
    rows.append(
        "# EyeCLIP/EchoCLIP use their official repositories/releases; run the "
        "coverage audit before downloading because current manifests may not "
        "contain ophthalmic or echocardiography-native inputs."
    )
    return rows


def download_open():
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("install/use an environment containing huggingface_hub") from exc
    for name, (repo, target) in OPEN_MODELS.items():
        if present(target):
            print(f"present: {name} -> {target}")
            continue
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {repo} -> {target}", flush=True)
        snapshot_download(repo_id=repo, local_dir=target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-open", action="store_true")
    parser.add_argument("--print-commands", action="store_true")
    args = parser.parse_args()

    print("MERIT-Tx expert asset audit:")
    for name, (repo, target) in {**OPEN_MODELS, **GATED_MODELS}.items():
        status = "present" if present(target) else "MISSING"
        print(f"  {name:8s} {status:7s} {target}  [{repo}]")
    for name, (repo, target, _access) in RESEARCH_MODELS.items():
        status = "present" if present(target) else "OPTIONAL"
        print(f"  {name:8s} {status:8s} {target}  [{repo}]  (research candidate)")

    if args.download_open:
        download_open()
    if args.print_commands or not args.download_open:
        print()
        print("\n".join(commands()))


if __name__ == "__main__":
    main()
