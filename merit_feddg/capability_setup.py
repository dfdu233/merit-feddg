"""Explicit preparation of the two small TorchXRayVision experts.

No weight download happens when importing/using the inference adapters. This
module does not install Qwen research extras or upgrade an existing LLaVA stack.
Published release byte sizes are checked; SHA256 here is a locally recorded
integrity fingerprint, NOT an upstream-authenticated model hash.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import importlib.metadata
import importlib.util
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from time import monotonic

_RELEASE = "https://github.com/mlmed/torchxrayvision/releases/download/v1/"
XRV_ASSETS = (
    {
        "id": "xrv-densenet121-res224-all",
        "filename": "nih-pc-chex-mimic_ch-google-openi-kaggle-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt",
        "bytes": 28382008,
    },
    {
        "id": "xrv-chest-anatomy-pspnet",
        "filename": "pspnet_chestxray_best_model_4.pth",
        "bytes": 272988989,
    },
)
_INTEGRITY_NOTE = (
    "SHA256 is locally computed after expected-size validation; the upstream release "
    "does not provide an authenticated SHA256 in this registry. This detects later local "
    "corruption, not malicious upstream/mirror replacement. PyTorch pickle weights must "
    "only be loaded from a source you trust."
)

# module, distribution, auto-install-if-missing, role. --no-deps prevents pip
# from replacing torch, torchvision, transformers or their existing dependencies.
_DEPENDENCIES = (
    ("torch", "torch", None, "preserved_core"),
    ("torchvision", "torchvision", None, "preserved_core"),
    ("transformers", "transformers", None, "preserved_core"),
    ("llava", "llava", None, "existing_llava_source_or_environment"),
    ("yaml", "PyYAML", "PyYAML>=6,<7", "runner"),
    ("pyarrow", "pyarrow", "pyarrow>=15,<24", "real_parquet_data"),
    ("torchxrayvision", "torchxrayvision", "torchxrayvision==1.5.4", "xrv"),
    ("skimage", "scikit-image", "scikit-image>=0.21,<0.24", "xrv"),
    ("imageio", "imageio", "imageio>=2.27,<3", "xrv"),
    ("tifffile", "tifffile", "tifffile>=2022.8,<2025", "xrv"),
    ("lazy_loader", "lazy-loader", "lazy-loader>=0.3,<1", "xrv"),
    ("networkx", "networkx", "networkx>=2.8,<4", "xrv"),
    ("dateutil", "python-dateutil", "python-dateutil>=2.8,<3", "pandas"),
    ("pytz", "pytz", "pytz", "pandas"),
    ("tzdata", "tzdata", "tzdata", "pandas"),
    ("scipy", "scipy", "scipy>=1.10,<1.13", "xrv"),
    ("pandas", "pandas", "pandas>=1.5,<2.3", "xrv"),
    ("requests", "requests", "requests>=2.31,<3", "xrv"),
    ("tqdm", "tqdm", "tqdm>=4.64,<5", "xrv"),
    ("ftfy", "ftfy", "ftfy>=6.1,<7", "biomedclip"),
    ("regex", "regex", "regex", "biomedclip"),
    ("timm", "timm", "timm>=0.9.8,<1.1", "biomedclip"),
    ("safetensors", "safetensors", "safetensors>=0.4,<1", "biomedclip"),
    ("huggingface_hub", "huggingface-hub", "huggingface-hub>=0.20,<1", "biomedclip"),
    ("h5py", "h5py", "h5py>=3.8,<4", "conch"),
    ("conch", "conch",
     "git+https://github.com/Mahmoodlab/CONCH.git@141cc09c7d4ff33d8eda562bd75169b457f71a62",
     "official_conch_source"),
    ("open_clip", "open-clip-torch", "open_clip_torch==3.2.0", "biomedclip_local_directory"),
)


def dependency_report():
    result = []
    for module, distribution, install_spec, role in _DEPENDENCIES:
        try:
            available = importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):
            available = False
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            version = None
        item = {
            "module": module,
            "distribution": distribution,
            "present": available,
            "version": version,
            "role": role,
            "install_if_missing": install_spec,
            "check": "discovery_only_not_ABI_or_GPU_validation",
        }
        if not available and install_spec is None:
            item["hint"] = (
                "Use the existing compatible environment/source. Set PYTHONPATH to the "
                "official local LLaVA-Med/CONCH source if appropriate. This preparation "
                "step deliberately does not install or upgrade this package."
            )
        if module == "open_clip" and available and version:
            match = re.match(r"(\d+)\.(\d+)", version)
            if match and tuple(map(int, match.groups())) < (3, 2):
                item["hint"] = (
                    "Local-directory BiomedCLIP requires open_clip >= 3.2. Existing version "
                    "was preserved. In this environment explicitly run: python -m pip install "
                    "--no-deps open_clip_torch==3.2.0. This does not upgrade transformers."
                )
        result.append(item)
    return result


def install_missing_dependencies(report, *, mirror="global"):
    if mirror not in {"global", "cn"}:
        raise ValueError("mirror must be global or cn")
    if any(item["role"] == "preserved_core" and not item["present"] for item in report):
        raise RuntimeError("Activate the existing torch/torchvision/transformers environment first")
    index = (
        "https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple"
        if mirror == "cn"
        else "https://pypi.org/simple"
    )
    installed = []
    for item in report:
        if item["present"] or not item["install_if_missing"]:
            continue
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--index-url",
                index,
                item["install_if_missing"],
            ],
            check=True,
        )
        installed.append(item["install_if_missing"])
    return installed


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _marker_path(path):
    return path.with_name(path.name + ".merit-integrity.json")


def _atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


@contextmanager
def _download_lock(path):
    lock = path.with_name(path.name + ".download.lock")
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(
            f"Another download owns {lock}. If a previous process was killed, verify it has "
            "stopped before removing only this lock file; keep the .incomplete weights."
        ) from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump({"pid": os.getpid(), "weight": str(path)}, stream)
        yield
    finally:
        lock.unlink(missing_ok=True)


def _transport_url(source_url, github_proxy=None):
    # --mirror cn affects PyPI only. Pickle weights use the official release by
    # default because the registry lacks an independently authenticated hash.
    if not github_proxy:
        return source_url
    parsed = urllib.parse.urlsplit(github_proxy)
    if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
        raise ValueError("MERIT_GITHUB_PROXY must be an explicit HTTPS URL without credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("MERIT_GITHUB_PROXY must not contain query strings or fragments")
    return github_proxy.rstrip("/") + "/" + source_url


def _existing_status(path, asset):
    if not path.exists():
        return None
    if not path.is_file() or path.stat().st_size != int(asset["bytes"]):
        raise RuntimeError(f"Existing weight has wrong size; retained without overwrite: {path}")
    marker = _marker_path(path)
    if not marker.is_file():
        raise RuntimeError(
            f"Existing weight has no integrity manifest and was retained: {path}. "
            "Its origin cannot be inferred; validate it explicitly before reusing."
        )
    recorded = json.loads(marker.read_text(encoding="utf-8"))
    expected = {
        "schema": 1,
        "id": asset["id"],
        "source_url": _RELEASE + asset["filename"],
        "bytes": int(asset["bytes"]),
    }
    if any(recorded.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"Weight integrity manifest does not match registry: {path}")
    if recorded.get("sha256") != _sha256(path):
        raise RuntimeError(f"Weight SHA256 changed; retained without overwrite: {path}")
    return {**recorded, "path": str(path.resolve()), "status": "reused_verified_local"}


def _download_once(partial, url, expected_size, *, timeout=60):
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > expected_size:
        raise RuntimeError("Partial file exceeds expected size; retained without truncation")
    if offset == expected_size:
        return
    headers = {"Accept-Encoding": "identity", "User-Agent": "merit-feddg-capability-setup/1"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)
    print(f"XRV connecting: {partial.name}, {offset}/{expected_size} bytes", file=sys.stderr)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        code = response.status
        content_type = response.headers.get("Content-Type", "").lower()
        if "text/html" in content_type or "application/json" in content_type:
            raise RuntimeError("Download endpoint returned a page/error, not model weights")
        content_range = response.headers.get("Content-Range", "")
        if offset or code == 206:
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
            if code != 206 or not match:
                raise RuntimeError(
                    "Server did not honor Range; existing partial retained unchanged"
                )
            start, end, total = map(int, match.groups())
            if start != offset or total != expected_size or not start <= end < total:
                raise RuntimeError("Mismatched Content-Range; existing partial retained unchanged")
            response_size = end - start + 1
        elif code == 200:
            response_size = expected_size
        else:
            raise RuntimeError(f"Unexpected download HTTP status: {code}")
        content_length = response.headers.get("Content-Length")
        if content_length is not None and int(content_length) != response_size:
            raise RuntimeError("Unexpected Content-Length; no partial bytes changed")
        received = 0
        last_progress = monotonic()
        with partial.open("ab") as stream:
            while True:
                chunk = response.read(min(1024 * 1024, response_size - received + 1))
                if not chunk:
                    break
                if received + len(chunk) > response_size:
                    raise RuntimeError(
                        "Download exceeded declared size; excess bytes were not written"
                    )
                stream.write(chunk)
                received += len(chunk)
                if monotonic() - last_progress >= 20:
                    print(
                        f"XRV progress: {offset + received}/{expected_size} bytes", file=sys.stderr
                    )
                    last_progress = monotonic()
            stream.flush()
            os.fsync(stream.fileno())
        if received != response_size or partial.stat().st_size != expected_size:
            raise RuntimeError("Incomplete transfer; partial bytes preserved for the next retry")


def ensure_xrv_weight(artifacts, asset, *, check_only=False, github_proxy=None, attempts=3):
    name = asset["filename"]
    if Path(name).name != name or "/" in name or "\\" in name:
        raise ValueError("XRV asset filename must be a basename")
    path = Path(artifacts) / "models" / "xrv" / name
    existing = _existing_status(path, asset)
    if existing is not None:
        return existing
    partial = path.with_name(path.name + ".incomplete")
    if check_only:
        return {
            "id": asset["id"],
            "path": str(path.resolve()),
            "status": "missing",
            "expected_bytes": asset["bytes"],
            "partial_bytes": partial.stat().st_size if partial.is_file() else 0,
        }
    if attempts < 1:
        raise ValueError("at least one download attempt required")
    path.parent.mkdir(parents=True, exist_ok=True)
    with _download_lock(path):
        existing = _existing_status(path, asset)
        if existing is not None:
            return existing
        return _finish_download(path, partial, asset, github_proxy, attempts)


def _finish_download(path, partial, asset, github_proxy, attempts):
    name = asset["filename"]
    source_url = _RELEASE + name
    transport = _transport_url(source_url, github_proxy)
    started_at = partial.stat().st_size if partial.exists() else 0
    errors = []
    for _ in range(attempts):
        try:
            _download_once(partial, transport, int(asset["bytes"]))
            break
        except (OSError, RuntimeError, ValueError, http.client.HTTPException) as exc:
            errors.append(str(exc))
    else:
        raise RuntimeError("XRV download failed; keep .incomplete and rerun: " + " | ".join(errors))
    recorded = {
        "schema": 1,
        "id": asset["id"],
        "source_url": source_url,
        "transport_url": transport,
        "third_party_transport": bool(github_proxy),
        "bytes": int(asset["bytes"]),
        "sha256": _sha256(partial),
        "upstream_sha256": None,
        "integrity_note": _INTEGRITY_NOTE,
    }
    # A same-target lock prevents cooperating processes sharing one partial.
    if path.exists():
        raise RuntimeError(f"Another writer created {path}; partial retained, nothing overwritten")
    # Publish the manifest first: an interruption can then resume the still
    # complete .incomplete file without treating a completed file as untracked.
    _atomic_json(_marker_path(path), recorded)
    # Atomic exclusive creation prevents replacing a user-created file even if
    # it appeared after the existence check. Both paths live on the same volume.
    os.link(partial, path)
    partial.unlink()
    return {
        **recorded,
        "path": str(path.resolve()),
        "status": "resumed" if started_at else "downloaded",
    }


def prepare_capabilities(artifacts, *, mirror="global", check_only=False, install_deps=False):
    if mirror not in {"global", "cn"}:
        raise ValueError("mirror must be global or cn")
    if check_only and install_deps:
        raise ValueError("--check-only cannot be combined with --install-deps")
    dependencies = dependency_report()
    installed = install_missing_dependencies(dependencies, mirror=mirror) if install_deps else []
    if installed:
        importlib.invalidate_caches()
        dependencies = dependency_report()
    results, failed = [], []
    for asset in XRV_ASSETS:
        try:
            results.append(
                ensure_xrv_weight(
                    artifacts,
                    asset,
                    check_only=check_only,
                    github_proxy=os.getenv("MERIT_GITHUB_PROXY"),
                )
            )
        except (OSError, RuntimeError, ValueError) as exc:
            failed.append({"id": asset["id"], "error": str(exc)})
    return {
        "artifacts": str(Path(artifacts).resolve()),
        "check_only": check_only,
        "mirror": mirror,
        "xrv_default_transport": "official_github_release",
        "dependencies": dependencies,
        "installed": installed,
        "weights": results,
        "failed": failed,
        "integrity_note": _INTEGRITY_NOTE,
        "ready_weights": not failed and all(item["status"] != "missing" for item in results),
        "environment_note": "No upgrade of torch/torchvision/transformers/llava/CONCH/open_clip; "
        "dependency discovery is not a compatibility or clinical validation.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--mirror", choices=("global", "cn"), default="global")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--install-deps", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = prepare_capabilities(**vars(args))
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        print(json.dumps({"failed": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ready_weights"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
