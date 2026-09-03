from __future__ import annotations

import json
from pathlib import Path

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".dcm"}


def build_folder_manifest(
    root: str | Path,
    output: str | Path,
    modality: str,
    domain: str,
    prompt: str = "Describe the clinically relevant findings.",
) -> int:
    """Create a safe manifest without copying or uploading any medical images."""

    root = Path(root).resolve()
    output = Path(output)
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            rows.append(
                {
                    "id": f"{domain}-{len(rows):07d}",
                    "image": str(path),
                    "modality": modality,
                    "domain": domain,
                    "prompt": prompt,
                    "candidates": [],
                    "label": None,
                }
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def audit_domain_split(paths: list[str | Path], held_out: set[str]) -> dict:
    seen_ids: dict[str, str] = {}
    domains: dict[str, int] = {}
    leaks: list[dict[str, str]] = []
    for manifest in paths:
        with Path(manifest).open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                domain = str(row["domain"])
                sample_id = str(row["id"])
                domains[domain] = domains.get(domain, 0) + 1
                if sample_id in seen_ids and seen_ids[sample_id] != domain:
                    leaks.append({"id": sample_id, "left": seen_ids[sample_id], "right": domain})
                seen_ids[sample_id] = domain
    source = set(domains) - held_out
    if source & held_out:
        raise AssertionError("source and held-out domains overlap")
    return {"domains": domains, "held_out": sorted(held_out), "id_leaks": leaks}
