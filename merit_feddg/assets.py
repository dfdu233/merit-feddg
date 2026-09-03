from __future__ import annotations

import os
from pathlib import Path

from .io import load_yaml, save_json


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _entries(profile: str, registry: Path, key: str) -> list[dict]:
    payload = load_yaml(registry)
    if profile not in payload["profiles"]:
        choices = ", ".join(sorted(payload["profiles"]))
        raise ValueError(f"unknown profile {profile!r}; choose from {choices}")
    return list(payload["profiles"][profile][key])


def download_profile(
    profile: str,
    root: str | Path,
    dry_run: bool = False,
    include_gated: bool = False,
) -> dict:
    root = Path(root)
    model_registry = repository_root() / "configs" / "models.yaml"
    dataset_registry = repository_root() / "configs" / "datasets.yaml"
    models = _entries(profile, model_registry, "models")
    datasets = _entries(profile, dataset_registry, "datasets")
    plan = {
        "profile": profile,
        "root": str(root.resolve()),
        "models": models,
        "datasets": datasets,
        "downloaded": [],
        "skipped": [],
        "failed": [],
    }
    if dry_run or (not models and not datasets):
        return plan

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("install the research extra before downloading assets") from exc

    token = os.getenv("HF_TOKEN")
    model_root = root / "models"
    dataset_root = root / "datasets"
    model_root.mkdir(parents=True, exist_ok=True)
    dataset_root.mkdir(parents=True, exist_ok=True)

    for entry in models:
        if entry.get("gated") and not include_gated:
            plan["skipped"].append(
                {"id": entry["id"], "reason": "gated; pass --include-gated after accepting terms"}
            )
            continue
        destination = model_root / entry["id"].replace("/", "--")
        try:
            snapshot_download(
                repo_id=entry["id"],
                local_dir=destination,
                token=token,
            )
            plan["downloaded"].append({"id": entry["id"], "path": str(destination)})
        except Exception as exc:  # noqa: BLE001 - isolate failures across independent assets
            plan["failed"].append({"id": entry["id"], "error": str(exc)})

    for entry in datasets:
        destination = dataset_root / entry["id"].replace("/", "--")
        try:
            snapshot_download(
                repo_id=entry["id"],
                repo_type="dataset",
                local_dir=destination,
                token=token,
            )
            plan["downloaded"].append({"id": entry["id"], "path": str(destination)})
        except Exception as exc:  # noqa: BLE001 - isolate failures across independent assets
            plan["failed"].append({"id": entry["id"], "error": str(exc)})

    save_json(root / "download-report.json", plan)
    return plan


def asset_plan(profile: str) -> dict:
    root = repository_root()
    return {
        "models": _entries(profile, root / "configs" / "models.yaml", "models"),
        "datasets": _entries(profile, root / "configs" / "datasets.yaml", "datasets"),
    }
