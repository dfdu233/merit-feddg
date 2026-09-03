from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from .io import load_yaml, save_json

_COMPLETE_MARKER = ".merit-download-complete.json"
_COMPLETE_TEMP = f"{_COMPLETE_MARKER}.tmp"
_MARKER_SCHEMA = 1


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
    force_download: bool = False,
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
        "resumed": [],
        "reused": [],
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
                    revision=entry.get("revision"),
                    force_download=force_download,
                )
                return endpoint
            except Exception as exc:  # noqa: BLE001 - retry a configured independent endpoint
                failures.append(f"{endpoint}: {exc}")
        raise RuntimeError(" | ".join(failures))

    def ensure(entry: dict, destination: Path, kind: str) -> None:
        repo_type = "dataset" if kind == "dataset" else None
        if not force_download and _completion_matches(destination, entry, kind):
            plan["reused"].append({"id": entry["id"], "path": str(destination)})
            return

        existed = destination.exists()
        try:
            endpoint = download(entry, destination, repo_type=repo_type)
            state = _asset_state(destination, kind)
            if not state["payload_files"]:
                raise RuntimeError("snapshot completed without a usable payload file")
            _write_completion_marker(destination, entry, kind, state)
            bucket = "resumed" if existed and not force_download else "downloaded"
            plan[bucket].append(
                {"id": entry["id"], "path": str(destination), "endpoint": endpoint}
            )
        except Exception as exc:  # noqa: BLE001 - isolate failures across independent assets
            plan["failed"].append({"id": entry["id"], "error": str(exc)})

    for entry in models:
        if entry.get("gated") and not include_gated:
            plan["skipped"].append(
                {"id": entry["id"], "reason": "gated; pass --include-gated after accepting terms"}
            )
            continue
        destination = model_root / entry["id"].replace("/", "--")
        ensure(entry, destination, "model")

    for entry in datasets:
        destination = dataset_root / entry["id"].replace("/", "--")
        ensure(entry, destination, "dataset")

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


def _content_files(path: Path) -> list[Path]:
    if not path.is_dir():
        return []
    return [
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file()
        and candidate.name not in {_COMPLETE_MARKER, _COMPLETE_TEMP}
        and ".cache" not in candidate.relative_to(path).parts
    ]


def _content_fingerprint(path: Path, files: list[Path]) -> str:
    digest = hashlib.sha256()
    for candidate in sorted(files, key=lambda item: item.relative_to(path).as_posix()):
        relative = candidate.relative_to(path).as_posix()
        digest.update(f"{relative}\0{candidate.stat().st_size}\n".encode())
    return digest.hexdigest()


def _asset_state(path: Path, kind: str) -> dict:
    files = _content_files(path)
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
        "fingerprint": _content_fingerprint(path, files),
    }


def _write_completion_marker(path: Path, entry: dict, kind: str, state: dict) -> None:
    payload = {
        "schema": _MARKER_SCHEMA,
        "id": entry["id"],
        "kind": kind,
        "revision": entry.get("revision"),
        "files": state["files"],
        "payload_files": state["payload_files"],
        "bytes": state["bytes"],
        "fingerprint": state["fingerprint"],
    }
    marker = path / _COMPLETE_MARKER
    temporary = path / _COMPLETE_TEMP
    save_json(temporary, payload)
    temporary.replace(marker)


def _completion_matches(path: Path, entry: dict, kind: str) -> bool:
    marker = path / _COMPLETE_MARKER
    try:
        with marker.open("r", encoding="utf-8") as handle:
            recorded = json.load(handle)
    except (OSError, ValueError, TypeError):
        return False
    if (
        recorded.get("schema") != _MARKER_SCHEMA
        or recorded.get("id") != entry["id"]
        or recorded.get("kind") != kind
        or recorded.get("revision") != entry.get("revision")
    ):
        return False
    state = _asset_state(path, kind)
    return bool(
        state["payload_files"]
        and recorded.get("files") == state["files"]
        and recorded.get("payload_files") == state["payload_files"]
        and recorded.get("bytes") == state["bytes"]
        and recorded.get("fingerprint") == state["fingerprint"]
    )


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
            if _completion_matches(path, entry, item_kind):
                report["present"].append(state)
            else:
                state["reason"] = (
                    "payload is missing or the completed-download marker does not match"
                )
                report["missing"].append(state)
    report["ready"] = not report["missing"]
    return report
