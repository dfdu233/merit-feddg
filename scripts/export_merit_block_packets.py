"""Export frozen BARD native expert caches into MERIT-Block real-evidence packets.

This is a zero-expert-inference conversion.  It reconstructs the exact
CapabilityRequest used by prepare_bard_expert_cache.py, verifies the cache
identity, and writes only native EvidenceItem dictionaries plus answer-blind
routing metadata.  References/answers are never loaded.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from merit_feddg.capabilities import CapabilityRequest
from merit_feddg.capability_runtime import ValueGenerationConfig
from merit_feddg.evidence_need import evidence_need
from merit_feddg.matched_evaluation import load_cached, load_manifest
from merit_feddg.open_study import fingerprint


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _request(row, descriptor, generation):
    need = evidence_need(row["question"], descriptor)
    return CapabilityRequest(
        sample_id=row["id"],
        image=row["image"],
        question=row["question"],
        modality=row["modality"],
        task=row["task"],
        domain="official-test",
        group_id=row["image_sha256"],
        capability=descriptor["capability"],
        scope=descriptor["scope"],
        query=need.query if generation.request_style == "need" else row["question"],
        generated_prefix="",
    )


def export_packets(cache_root, manifest):
    root = Path(cache_root)
    protocol = _read_json(root / "protocol.json")
    if protocol.get("schema") != "bard-expert-cache-v1":
        raise ValueError("cache root is not a completed BARD expert cache")
    if not protocol.get("shards_complete") or protocol.get("references_loaded"):
        raise ValueError("expert cache must be complete and reference-free")
    identity = str(protocol["identity"])
    config = protocol["config"]
    generation = ValueGenerationConfig(**config["capability_value"]["generation"])
    schedule = _read_json(root / "schedule.json")
    routes = _read_json(root / "routing.json")
    original = load_manifest(manifest)

    expected_ids = {row["id"] for row in original}
    if set(schedule) != expected_ids or set(routes) != expected_ids:
        raise ValueError("cache schedule/routing do not cover the exact manifest")

    output = []
    missing = []
    for row in original:
        sample_id = row["id"]
        modality = routes[sample_id].get("modality")
        if not isinstance(modality, str) or not modality:
            raise ValueError(f"{sample_id}: cached route has invalid modality")
        routed = {
            **row,
            "modality": modality,
        }
        experts = {}
        for descriptor in schedule[sample_id]:
            expert_id = descriptor["expert"]
            request = _request(routed, descriptor, generation)
            key = fingerprint(["infer", expert_id, asdict(request)])
            directory = root / "expert-cache" / fingerprint(sample_id)
            value = load_cached(directory / f"{key}.json", identity)
            if value is None:
                missing.append((sample_id, expert_id, descriptor["capability"]))
                continue
            items = value.get("items")
            if not isinstance(items, list):
                raise ValueError(f"{sample_id}/{expert_id}: cached items are invalid")
            if items:
                experts.setdefault(expert_id, []).extend(items)

        output.append(
            {
                "id": sample_id,
                "image_sha256": row["image_sha256"],
                "group_id": row["image_sha256"],
                "modality": modality,
                "task": row.get("task", "open_vqa"),
                "experts": experts,
                "source_cache_identity": identity,
                "references_read": False,
                "expert_inference_executed": False,
            }
        )

    if missing:
        preview = ", ".join("/".join(values) for values in missing[:5])
        raise FileNotFoundError(
            f"expert cache is incomplete for {len(missing)} scheduled request(s): {preview}"
        )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expert-cache", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = export_packets(args.expert_cache, args.manifest)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(target)


if __name__ == "__main__":
    main()
