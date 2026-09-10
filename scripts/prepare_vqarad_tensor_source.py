#!/usr/bin/env python3
"""Build an image-disjoint VQA-RAD source cache with real native expert outputs."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import yaml
from PIL import Image

from merit_feddg.capabilities import CapabilityRequest, tool_descriptors, validate_result
from merit_feddg.capability_experts import CapabilityPool
from merit_feddg.capability_study import _filter_optional_experts, _route_records
from merit_feddg.generalist_factory import load_generalist
from merit_feddg.io import load_experiment_yaml
from merit_feddg.open_study import atomic_json, fingerprint
from merit_feddg.tensor_evidence import TensorContract, compile_tensor_evidence


def rgb_image(value):
    if not isinstance(value, dict):
        raise ValueError("expected embedded Hugging Face image")
    source = io.BytesIO(value["bytes"]) if value.get("bytes") else value.get("path")
    if source is None:
        raise ValueError("image has neither bytes nor path")
    with Image.open(source) as image:
        return image.convert("RGB")


def rgb_digest(image):
    return hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-parquet", required=True)
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--config", default="configs/matched_vector_gate.yaml")
    parser.add_argument("--contract", default="configs/vqarad_tensor_contract.json")
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    images = output / "images"
    images.mkdir(exist_ok=True)
    test_manifest_rows = [json.loads(line) for line in Path(args.test_manifest).read_text().splitlines()
                          if line.strip()]
    # Only image identities are used. The manifest may carry questions, but no
    # question field participates in filtering/selection and it contains no answers.
    test_hashes = {row["image_sha256"] for row in test_manifest_rows}
    frame = pd.read_parquet(args.train_parquet)
    candidates = defaultdict(list)
    overlap_questions = 0
    for index, raw in frame.iterrows():
        image = rgb_image(raw["image"])
        digest = rgb_digest(image)
        if digest in test_hashes:
            overlap_questions += 1
            continue
        path = images / f"{digest}.png"
        if not path.exists():
            image.save(path, format="PNG")
        question, answer = str(raw["question"]).strip(), str(raw["answer"]).strip()
        if question and answer:
            candidates[digest].append((int(index), question, answer, str(path)))

    config = load_experiment_yaml(args.config)
    config["generalist"].pop("tensor_bridge_checkpoint", None)
    config["generalist"]["deterministic_image_padding"] = True
    train_config = output / "generalist-base.yaml"
    train_config.write_text(yaml.safe_dump({"generalist": config["generalist"],
                                            "experts": config["experts"]}, sort_keys=False))
    specs, excluded = _filter_optional_experts(config["experts"], args.artifacts)
    specs.pop("source_cases", None)
    specs = {name: {**spec, "capabilities": [cap for cap in spec.get("capabilities", [])
                                              if cap in {"classification", "segmentation", "detection"}]}
             for name, spec in specs.items()}
    specs = {name: spec for name, spec in specs.items() if spec["capabilities"]}
    probe = load_generalist(config["generalist"], args.artifacts)
    image_rows = [{"id": f"vqarad-source-image-{digest[:20]}", "image": values[0][3],
                   "question": "", "modality": "mixed", "capability": "classification",
                   "task": "open_vqa", "domain": "vqarad-official-train-image-disjoint",
                   "domain_kind": "official_train_filtered_by_test_image_identity", "role": "source",
                   "group_id": digest, "image_sha256": digest}
                  for digest, values in sorted(candidates.items())]
    routed, routing = _route_records(image_rows, config.get("routing", {}), lambda: probe,
                                     {"source_protocol": 1, "config": fingerprint(config)}, output)
    routed_by_hash = {row["image_sha256"]: row for row in routed}
    contract = TensorContract.from_dict(json.loads(Path(args.contract).read_text()))
    pool = CapabilityPool(specs, args.artifacts, source_records=())
    source, skipped = [], []
    try:
        for number, (digest, values) in enumerate(sorted(candidates.items()), 1):
            base = routed_by_hash[digest]
            choices = sorted(values, key=lambda value: (hashlib.sha256(value[1].encode()).hexdigest(),
                                                         value[0]))
            selected = None
            for index, question, answer, image_path in choices:
                row = {**base, "id": f"vqarad-source-{index:04d}", "question": question}
                descriptors = tool_descriptors(specs, row)
                if descriptors:
                    selected = (index, question, answer, image_path, row, descriptors)
                    break
            if selected is None:
                skipped.append({"image_sha256": digest, "reason": "no_applicable_vector_expert"})
                continue
            index, question, answer, image_path, row, descriptors = selected
            evidence = []
            for descriptor in descriptors:
                request = CapabilityRequest(row["id"], image_path, question, row["modality"],
                                            "open_vqa", row["domain"], digest,
                                            descriptor["capability"], query=question,
                                            scope=descriptor["scope"])
                result = validate_result(pool.infer(descriptor["expert"], request),
                                         descriptor["expert"], request)
                for item in result.items:
                    packet = compile_tensor_evidence((item,), Image.open(image_path).size, contract)
                    if len(packet) and not packet.rejected:
                        evidence.append(asdict(item))
            if not evidence:
                skipped.append({"image_sha256": digest, "source_index": index,
                                "reason": "no_fully_registered_nonempty_evidence"})
                continue
            source.append({"id": row["id"], "split": "source", "domain": row["domain"],
                           "image": image_path, "prompt": question + "\n" + config.get(
                               "prompt_suffix", "Answer concisely from the image."),
                           "answer": answer, "evidence": evidence})
            print(f"tensor source {number}/{len(candidates)} kept={len(source)} "
                  f"modality={row['modality']} tools={len(evidence)}", flush=True)
    finally:
        pool.clear()

    source_path = output / "source.jsonl"
    source_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in source))
    audit = {
        "protocol": "official_train_filtered_by_exact_test_rgb_identity",
        "official_train_questions": len(frame), "official_test_questions": len(test_manifest_rows),
        "test_unique_images": len(test_hashes), "overlap_train_questions_removed": overlap_questions,
        "candidate_source_questions": sum(map(len, candidates.values())),
        "candidate_source_unique_images": len(candidates), "selected_source_rows": len(source),
        "selected_source_unique_images": len({row["image"] for row in source}),
        "source_test_image_overlap": len({Path(row["image"]).stem for row in source} & test_hashes),
        "one_question_per_source_image": True, "question_selection_uses_answer": False,
        "test_manifest_loaded_for_image_identities": True,
        "test_question_fields_used": False, "test_answers_loaded": False,
        "target_annotations_used": False,
        "modality_counts": dict(Counter(routed_by_hash[Path(row["image"]).stem]["modality"]
                                        for row in source)),
        "expert_item_counts": dict(Counter(item["expert_id"] for row in source
                                           for item in row["evidence"])),
        "excluded_optional_experts": excluded, "skipped": skipped,
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "contract_sha256": hashlib.sha256(Path(args.contract).read_bytes()).hexdigest(),
    }
    if not source or audit["source_test_image_overlap"]:
        raise ValueError("source cache is empty or overlaps the complete test image set")
    atomic_json(output / "audit.json", audit)
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
