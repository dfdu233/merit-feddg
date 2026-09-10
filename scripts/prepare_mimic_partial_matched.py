#!/usr/bin/env python3
"""Prepare the existing ANCHOR MIMIC-partial report set for matched inference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from merit_feddg.open_study import atomic_json


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--baseline-answers", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    source = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    answers = [json.loads(line) for line in args.baseline_answers.read_text(encoding="utf-8").splitlines()
               if line.strip()]
    source_ids = [str(row.get("qid", row.get("id"))) for row in source]
    answer_ids = [str(row["question_id"]) for row in answers]
    if len(source_ids) != len(set(source_ids)) or source_ids != answer_ids:
        raise ValueError("baseline answers must be the exact ordered full manifest")

    manifest, references, outputs = [], {}, {}
    for row, answer, sample_id in zip(source, answers, source_ids, strict=True):
        reference = str(row.get("answer", "")).strip()
        if not reference or str(answer.get("gt_ans", "")).strip() != reference:
            raise ValueError(f"reference mismatch for {sample_id}")
        image = args.image_root / str(row["img_name"])
        if not image.is_file():
            raise FileNotFoundError(image)
        manifest.append({
            "id": sample_id,
            "image": str(image.resolve()),
            "question": str(row["question"]).strip(),
            "answer_type": "report",
            "task": "report_generation",
            "official_index": len(manifest),
            "source_qid": sample_id,
            "image_sha256": sha256_file(image),
        })
        references[sample_id] = [reference]
        metadata = answer.get("metadata", {})
        token_ids = list(metadata.get("generated_token_ids") or [])
        if not token_ids:
            raise ValueError(f"baseline token IDs unavailable for {sample_id}")
        outputs[sample_id] = {
            "text": str(answer.get("text", "")),
            "token_ids": token_ids,
            "finished": metadata.get("stop_reason") != "max_new_tokens",
            "seconds": 0.0,
            "visual_evidence": [],
            "guidance_trace": [],
            "guidance_spent": 0.0,
            "evidence_transport": {"schema": "reused-generalist-v1", "presented": [], "omitted": []},
            "expert_calls": 0,
            "controller_calls": 0,
            "probe_model_calls": 0,
            "probe_seconds": 0.0,
            "controller_output_tokens": 0,
            "trace": [],
            "evidence": [],
            "adopted_evidence_count": 0,
            "presented_evidence_count": 0,
            "historical_runtime_not_replayed": True,
        }

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in manifest:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    atomic_json(args.output / "references.json", references)
    atomic_json(args.output / "baselines" / "llava-med-greedy-anchor-report-v1.json", {
        "schema": "matched-generalist-reuse-v1",
        "prompt_contract": "anchor-task-v1",
        "generalist_id": "microsoft/llava-med-v1.5-mistral-7b",
        "n": len(outputs),
        "source": {
            "path": str(args.baseline_answers.resolve()),
            "sha256": sha256_file(args.baseline_answers),
            "manifest": str(args.source_manifest.resolve()),
            "manifest_sha256": sha256_file(args.source_manifest),
            "labels_removed": True,
            "source_answers": len(outputs),
        },
        "outputs": outputs,
    })
    print(json.dumps({"n": len(manifest), "output": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
