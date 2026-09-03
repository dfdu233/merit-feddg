from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from pathlib import Path


def _torch_status() -> dict:
    try:
        import torch
    except ImportError:
        return {"installed": False, "cuda_available": False, "gpus": []}

    cuda_available = bool(torch.cuda.is_available())
    return {
        "installed": True,
        "version": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_runtime": torch.version.cuda,
        "gpus": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
        if cuda_available
        else [],
    }


def _nvidia_status() -> dict:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return {"available": False, "gpus": []}
    try:
        result = subprocess.run(
            [
                executable,
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "error": str(exc), "gpus": []}
    devices = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 3:
            try:
                memory_mib = int(fields[2])
            except ValueError:
                memory_mib = fields[2]
            devices.append({"name": fields[0], "driver": fields[1], "memory_mib": memory_mib})
    return {"available": True, "gpus": devices}


def _huggingface_status() -> dict:
    try:
        from huggingface_hub import get_token
    except ImportError:
        return {"client_installed": False, "authenticated": False}
    return {"client_installed": True, "authenticated": bool(get_token())}


def diagnostics(root: str | Path) -> dict:
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    disk = shutil.disk_usage(root)
    gib = 1024**3
    return {
        "platform": platform.platform(),
        "python": {"version": platform.python_version(), "executable": sys.executable},
        "artifact_root": str(root),
        "disk_gib": {
            "total": round(disk.total / gib, 1),
            "free": round(disk.free / gib, 1),
        },
        "torch": _torch_status(),
        "nvidia_smi": _nvidia_status(),
        "huggingface": _huggingface_status(),
    }
