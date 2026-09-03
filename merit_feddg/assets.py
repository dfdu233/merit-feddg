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
    preferred_endpoint = os.getenv("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    fallback_endpoint = os.getenv("MERIT_HF_FALLBACK_ENDPOINT", "").rstrip("/")
    endpoints = [preferred_endpoint]
    if fallback_endpoint and fallback_endpoint not in endpoints:
        endpoints.append(fallback_endpoint)
    model_root = root / "models"
    dataset_root = root / "datasets"
    model_root.mkdir(parents=True, exist_ok=True)
    dataset_root.mkdir(parents=True, exist_ok=True)

    def download(entry: dict, destination: Path, repo_type: str | None = None) -> str:
        failures = []
        for endpoint in endpoints:
            try:
                snapshot_download(
                    repo_id=entry["id"],
                    repo_type=repo_type,
                    local_dir=destination,
                    token=token,
                    endpoint=endpoint,
                )
                return endpoint
            except Exception as exc:  # noqa: BLE001 - retry a configured independent endpoint
                failures.append(f"{endpoint}: {exc}")
        raise RuntimeError(" | ".join(failures))

    for entry in models:
        if entry.get("gated") and not include_gated:
            plan["skipped"].append(
                {"id": entry["id"], "reason": "gated; pass --include-gated after accepting terms"}
            )
            continue
        destination = model_root / entry["id"].replace("/", "--")
        try:
            endpoint = download(entry, destination)
            plan["downloaded"].append(
                {"id": entry["id"], "path": str(destination), "endpoint": endpoint}
            )
        except Exception as exc:  # noqa: BLE001 - isolate failures across independent assets
            plan["failed"].append({"id": entry["id"], "error": str(exc)})

    for entry in datasets:
        destination = dataset_root / entry["id"].replace("/", "--")
        try:
            endpoint = download(entry, destination, repo_type="dataset")
            plan["downloaded"].append(
                {"id": entry["id"], "path": str(destination), "endpoint": endpoint}
            )
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


_MODEL_WEIGHT_SUFFIXES = {".bin", ".pt", ".pth", ".safetensors"}
_DATASET_METADATA = {"readme.md", ".gitattributes", "license", "dataset_infos.json"}


def _asset_state(path: Path, kind: str) -> dict:
    files = []
    if path.is_dir():
        files = [
            candidate
            for candidate in path.rglob("*")
            if candidate.is_file() and ".cache" not in candidate.relative_to(path).parts
        ]
    if kind == "model":
        payloads = [candidate for candidate in files if candidate.suffix in _MODEL_WEIGHT_SUFFIXES]
    else:
        payloads = [
            candidate for candidate in files if candidate.name.lower() not in _DATASET_METADATA
        ]
    return {
        "path": str(path.resolve()),
        "files": len(files),
        "payload_files": len(payloads),
        "bytes": sum(candidate.stat().st_size for candidate in files),
    }


def verify_assets(
    profile: str,
    root: str | Path,
    include_gated: bool = False,
) -> dict:
    """Verify that every selected snapshot contains at least one payload file."""
    root = Path(root)
    plan = asset_plan(profile)
    report = {
        "profile": profile,
        "root": str(root.resolve()),
        "ready": True,
        "present": [],
        "missing": [],
        "skipped": [],
    }
    for kind, directory in (("models", "models"), ("datasets", "datasets")):
        for entry in plan[kind]:
            if entry.get("gated") and not include_gated:
                report["skipped"].append(
                    {
                        "kind": kind[:-1],
                        "id": entry["id"],
                        "reason": "gated asset was not requested",
                    }
                )
                continue
            path = root / directory / entry["id"].replace("/", "--")
            item_kind = kind[:-1]
            state = {"kind": item_kind, "id": entry["id"], **_asset_state(path, item_kind)}
            if state["payload_files"]:
                report["present"].append(state)
            else:
                report["missing"].append(state)
    report["ready"] = not report["missing"]
    return report
