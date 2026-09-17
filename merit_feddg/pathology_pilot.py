"""Pure, answer-blind utilities for a frozen pathology-specialist pilot.

The five arms test specialist headroom, not a new gate/decoding algorithm.
File identities and matched questions bind offline predictions to their inputs.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path

SCHEMA = "pathology-quilt-pilot-v1"
ARMS = ("compact", "without_conch", "quilt_alone", "compact_quilt", "compact_wrong_image")
LABEL_KEYS = frozenset({"answer", "answers", "reference", "references", "label", "labels"})


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path, data):
    """Never overwrite historical results, including an incomplete prior attempt."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def select_rows(rows, incumbent, *, limit=12, split="train"):
    """Keep all rows in the denominator; schedule a fixed microscopy TRAIN subset.

This is an existing-modality label, not a new learned or ground-truth image router.
No answer/reference fields may be present in the inference manifest.
"""
    if split != "train" or not 2 <= limit <= 100:
        raise ValueError("this diagnostic requires TRAIN and 2 <= limit <= 100")
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)) or not ids:
        raise ValueError("nonempty unique-ID manifest required")
    if set(ids) != set(incumbent):
        raise ValueError("complete matching incumbent/manifest IDs required")
    selected, counts = [], Counter()
    for row in rows:
        if LABEL_KEYS & row.keys():
            raise ValueError("reference/answer labels forbidden in inference manifest")
        if row.get("split", "train") != "train":
            raise ValueError("non-TRAIN row; do not re-label a test split")
        key = str(row["id"])
        modality = incumbent[key].get("input_modality", row.get("modality", "unknown"))
        counts[modality] += 1
        if modality != "pathology":
            continue
        if row.get("task", "open_vqa") != "open_vqa":
            continue
        if not all(isinstance(row.get(k), str) and row[k].strip()
                   for k in ("image", "question", "image_sha256")):
            raise ValueError("image/question/image_sha256 required")
        selected.append({k: row[k] for k in ("id", "image", "question", "image_sha256")}
                        | {"id": key, "modality": "pathology", "task": "open_vqa",
                           "group_id": str(row.get("group_id") or row["image_sha256"])})
    selected.sort(key=lambda r: digest([SCHEMA, r["id"], r["image_sha256"], r["question"]]))
    selected = selected[:limit]
    if len(selected) < 2:
        raise ValueError("need at least two eligible microscopy TRAIN cases")
    return selected, {"full_manifest_n": len(rows), "modality_counts": dict(counts),
                      "scheduled_n": len(selected), "selection_uses_answers": False,
                      "scope": "microscopy subset; not whole PathVQA performance"}


def donor_map(rows):
    """Wrong-image control uses the SAME question/prompt, never another case's answer.

Donor image/group must differ. Donors may be reused; this is not a bijective
permutation. Identity remains visible in the audit but not leaked into prompts.
"""
    result = {}
    for row in rows:
        eligible = [r for r in rows if r["image_sha256"] != row["image_sha256"]
                    and r["group_id"] != row["group_id"]]
        if not eligible:
            raise ValueError("no different-image/group donor for negative control")
        result[row["id"]] = min(eligible, key=lambda r: digest([row["id"], r["id"]]))["id"]
    return result


def prediction_key(row, source, variant):
    return digest({"variant": variant, "target": row["id"], "question": row["question"],
                   "prompt": row["prompt"], "source_pixel_sha": source["image_sha256"],
                   "source_file_sha": source["image_file_sha256"]})


def prediction_jobs(frozen):
    lookup = {r["id"]: r for r in frozen["rows"]}
    jobs = []
    for row in frozen["rows"]:
        for variant, source in (("matched", row),
                                ("wrong_image", lookup[frozen["donors"][row["id"]]])):
            jobs.append({"key": prediction_key(row, source, variant), "variant": variant,
                         "target_id": row["id"], "source_id": source["id"],
                         "question": row["question"], "prompt": row["prompt"],
                         "image": source["image"], "image_sha256": source["image_sha256"],
                         "image_file_sha256": source["image_file_sha256"]})
    return jobs


def validate_predictions(frozen, cache):
    if (cache.get("schema") != SCHEMA or cache.get("identity") != digest(frozen)
            or cache.get("complete") is not True):
        raise ValueError("missing/incomplete/mismatched prediction cache")
    model = cache.get("model", {})
    if (model.get("declared_model") != "wisdomik/Quilt-Llava-v1.5-7b"
            or model.get("precision") not in {"fp16", "4bit", "8bit"}
            or not all(model.get(k) for k in ("checkpoint", "vision", "source"))):
        raise ValueError("missing Quilt checkpoint/vision/source provenance")
    jobs = {j["key"]: j for j in prediction_jobs(frozen)}
    records = cache.get("predictions", {})
    if set(records) != set(jobs):
        raise ValueError("prediction cache must contain ALL matched/control jobs")
    for key, job in jobs.items():
        record = records[key]
        if record.get("job") != job or record.get("status") != "ok":
            raise ValueError("prediction/source/question identity mismatch")
        if not isinstance(record.get("text"), str) or not record["text"].strip():
            raise ValueError("empty specialist output is not evidence")
        tokens = record.get("token_ids")
        if not isinstance(tokens, list) or not tokens or any(type(t) is not int or t < 0 for t in tokens):
            raise ValueError("invalid specialist token IDs")
        if len(tokens) > frozen["max_new_tokens"]:
            raise ValueError("specialist output exceeds frozen budget")
        if not math.isfinite(record.get("seconds", float("nan"))) or record["seconds"] < 0:
            raise ValueError("invalid timing")
    return records


def quilt_item(record, *, expert_id="quilt_pathology", scope="histology_question_observation"):
    """Transport original generated text, not a diagnosis probability or certificate."""
    from .capabilities import EvidenceItem

    job = record["job"]
    return EvidenceItem(
        "quilt:" + job["key"], expert_id, "generation", scope,
        {"text": record["text"], "question": job["question"],
         "output_semantics": "unverified_histopathology_answer"},
        summary="", confidence=None,
        provenance={"adapter": "cached_quilt_llava", "source_image_sha256": job["image_sha256"],
                    "source_prediction_key": job["key"], "target_annotations_used": False,
                    "prediction_is_unverified": True},
    )


def paired_summary(scores, baseline):
    if set(scores) != set(baseline) or not scores:
        raise ValueError("paired nonempty scores with identical IDs required")
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in [*scores.values(), *baseline.values()]):
        raise ValueError("scores must be finite and in [0,1]")
    changes = [scores[k] - baseline[k] for k in scores]
    return {"n": len(changes), "mean_score": sum(scores.values()) / len(scores),
            "mean_delta": sum(changes) / len(changes),
            "score_improved": sum(v > 0 for v in changes),
            "score_harmed": sum(v < 0 for v in changes),
            "unchanged": sum(v == 0 for v in changes),
            "warning": "continuous-score improvements are not binary clinical rescues"}
