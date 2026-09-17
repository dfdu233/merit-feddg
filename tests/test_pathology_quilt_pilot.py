"""Synthetic protocol tests. No weights, clinical labels, or efficacy claims."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

from merit_feddg.experts.quilt_cached import CachedQuiltExpert, build_cached_quilt
from merit_feddg.pathology_pilot import (
    ARMS, SCHEMA, digest, donor_map, file_sha, paired_summary, prediction_jobs,
    quilt_item, select_rows, validate_predictions, write_new,
)

ROOT = Path(__file__).resolve().parents[1]
worker_spec = importlib.util.spec_from_file_location("quilt_worker_test", ROOT / "scripts/run_quilt_worker.py")
worker = importlib.util.module_from_spec(worker_spec)
worker_spec.loader.exec_module(worker)
HAS_UPSTREAM = importlib.util.find_spec("merit_feddg.capabilities") is not None


@pytest.fixture
def source(tmp_path):
    rows, old = [], {}
    for i, modality in enumerate(("pathology", "pathology", "gross_pathology", "cxr")):
        image = tmp_path / f"{i}.bin"
        image.write_bytes(bytes([i, i + 1]))
        rows.append({"id": str(i), "image": str(image), "question": "What tissue is shown?",
                     "image_sha256": digest(i), "group_id": str(i), "modality": modality})
        old[str(i)] = {"input_modality": modality}
    return rows, old


@pytest.fixture
def prepared(tmp_path, source):
    rows, old = source
    selected, coverage = select_rows(rows, old)
    for row in selected:
        row.update(prompt=row["question"] + "\nAnswer briefly.", image_file_sha256=file_sha(row["image"]))
    frozen = {"schema": SCHEMA, "split": "train", "rows": selected, "coverage": coverage,
              "donors": donor_map(selected), "max_new_tokens": 64, "arms": list(ARMS)}
    records = {job["key"]: {"job": job, "text": "Glandular tissue.", "status": "ok",
                           "token_ids": [10, 20, 2], "seconds": 0.01, "output_tokens": 3}
               for job in prediction_jobs(frozen)}
    cache = {"schema": SCHEMA, "identity": digest(frozen), "complete": True, "predictions": records,
             "model": {"declared_model": "wisdomik/Quilt-Llava-v1.5-7b", "precision": "4bit",
                       "checkpoint": {"synthetic": "hash"}, "vision": {"synthetic": "hash"},
                       "source": {"synthetic": "hash"}}}
    write_new(tmp_path / "frozen.json", frozen)
    write_new(tmp_path / "quilt_predictions.json", cache)
    return tmp_path, frozen, cache


def test_selection_filters_gross_and_records_denominator(source):
    rows, old = source
    selected, coverage = select_rows(rows, old)
    assert len(selected) == 2 and coverage["full_manifest_n"] == 4
    assert all(r["modality"] == "pathology" for r in selected)
    assert select_rows(list(reversed(rows)), old)[0] == selected


@pytest.mark.parametrize("key", ["answer", "answers", "reference", "references", "label", "labels"])
def test_reference_leakage_rejected(source, key):
    rows, old = source
    rows[0][key] = "forbidden"
    with pytest.raises(ValueError, match="forbidden"):
        select_rows(rows, old)


@pytest.mark.parametrize("limit", [0, 1, 101])
def test_bounded_selection(source, limit):
    with pytest.raises(ValueError):
        select_rows(*source, limit=limit)


def test_no_test_split_relabeling(source):
    rows, old = source
    rows[0]["split"] = "test"
    with pytest.raises(ValueError, match="TRAIN"):
        select_rows(rows, old)


def test_duplicate_ids(source):
    rows, old = source
    rows.append(rows[0])
    with pytest.raises(ValueError, match="unique"):
        select_rows(rows, old)


def test_missing_incumbent(source):
    rows, old = source
    del old["0"]
    with pytest.raises(ValueError, match="matching"):
        select_rows(rows, old)


def test_control_preserves_question_changes_image(prepared):
    _, frozen, _ = prepared
    jobs = prediction_jobs(frozen)
    for row in frozen["rows"]:
        pair = [j for j in jobs if j["target_id"] == row["id"]]
        assert len(pair) == 2
        assert pair[0]["question"] == pair[1]["question"] == row["question"]
        assert pair[0]["prompt"] == pair[1]["prompt"]
        assert pair[0]["image_sha256"] != pair[1]["image_sha256"]


def test_no_same_group_donor(prepared):
    _, frozen, _ = prepared
    for row in frozen["rows"]:
        row["group_id"] = "same"
    with pytest.raises(ValueError, match="donor"):
        donor_map(frozen["rows"])


def test_no_same_pixels_donor(prepared):
    _, frozen, _ = prepared
    for row in frozen["rows"]:
        row["image_sha256"] = "same"
    with pytest.raises(ValueError, match="donor"):
        donor_map(frozen["rows"])


def test_valid_cache(prepared):
    _, frozen, cache = prepared
    assert len(validate_predictions(frozen, cache)) == 4


@pytest.mark.parametrize("change", ["incomplete", "missing", "text", "tokens", "question", "image", "identity", "provenance", "nan"])
def test_bad_cache(prepared, change):
    _, frozen, original = prepared
    cache = copy.deepcopy(original)
    key = next(iter(cache["predictions"]))
    row = cache["predictions"][key]
    if change == "incomplete": cache["complete"] = False
    elif change == "missing": del cache["predictions"][key]
    elif change == "text": row["text"] = " "
    elif change == "tokens": row["token_ids"] = [-200]
    elif change == "question": row["job"]["question"] = "Changed?"
    elif change == "image": row["job"]["image_file_sha256"] = "wrong"
    elif change == "identity": cache["identity"] = "wrong"
    elif change == "provenance": cache["model"].pop("vision")
    elif change == "nan": row["seconds"] = float("nan")
    with pytest.raises(ValueError):
        validate_predictions(frozen, cache)


def test_factory_protocol_accepts_model_id(prepared):
    root, _, _ = prepared
    obj = build_cached_quilt(model_id="wisdomik/Quilt-Llava-v1.5-7b", run_dir=root,
                             cache_sha256=file_sha(root / "quilt_predictions.json"))
    assert isinstance(obj, CachedQuiltExpert)


def test_factory_wrong_model(prepared):
    root, _, _ = prepared
    with pytest.raises(ValueError, match="model_id"):
        build_cached_quilt(model_id="not-quilt", run_dir=root, cache_sha256="wrong")


def test_cache_digest_mismatch(prepared):
    root, _, _ = prepared
    with pytest.raises(ValueError, match="SHA256"):
        CachedQuiltExpert(run_dir=root, cache_sha256="wrong")


def test_write_does_not_overwrite(tmp_path):
    path = tmp_path / "frozen.json"
    write_new(path, {"x": 1})
    with pytest.raises(FileExistsError):
        write_new(path, {"x": 2})
    assert json.loads(path.read_text()) == {"x": 1}


def test_worker_output_layout():
    assert worker.suffix_tokens([1, -200, 7, 8, 2], [1, -200, 7]) == [8, 2]
    with pytest.raises(RuntimeError):
        worker.suffix_tokens([8, 2], [1, -200, 7])


def test_checkpoint_shard_completeness(tmp_path):
    write_new(tmp_path / "config.json", {"model_type": "llava"})
    write_new(tmp_path / "model.safetensors.index.json", {"weight_map": {"x": "missing.safetensors"}})
    with pytest.raises(ValueError, match="shards"):
        worker.checkpoint_identity(tmp_path)


def test_projector_is_not_full_checkpoint(tmp_path):
    write_new(tmp_path / "config.json", {})
    (tmp_path / "mm_projector.bin").write_bytes(b"synthetic")
    with pytest.raises(ValueError, match="full merged"):
        worker.checkpoint_identity(tmp_path)


def test_paired_metrics():
    result = paired_summary({"a": 1.0, "b": 0.5}, {"a": 0.0, "b": 1.0})
    assert result["score_improved"] == 1 and result["score_harmed"] == 1
    assert result["mean_delta"] == 0.25


def test_pair_ids_must_align():
    with pytest.raises(ValueError):
        paired_summary({"a": 1.0}, {"b": 1.0})


@pytest.mark.skipif(not HAS_UPSTREAM, reason="complete upstream repository unavailable locally")
def test_real_capability_factory_binding(prepared):
    from merit_feddg.capabilities import CapabilityRequest, validate_result
    root, frozen, _ = prepared
    row = frozen["rows"][0]
    expert = CachedQuiltExpert(run_dir=root, cache_sha256=file_sha(root / "quilt_predictions.json"))
    req = CapabilityRequest(row["id"], row["image"], row["question"], "pathology", "open_vqa",
                            "train", row["group_id"], "generation", scope=expert.scope)
    result = expert.infer(req)
    assert result.items[0].payload["text"] == "Glandular tissue."
    assert result.items[0].confidence is None
    validate_result(result, expert.expert_id, req)
    from dataclasses import replace
    with pytest.raises(ValueError, match="cache miss"):
        expert.infer(replace(req, question="What color?"))
    assert not expert.infer(replace(req, modality="gross_pathology")).items


@pytest.mark.skipif(not HAS_UPSTREAM, reason="complete upstream repository unavailable locally")
def test_control_not_disclosed_in_prompt_fields(prepared):
    _, _, cache = prepared
    record = next(r for r in cache["predictions"].values() if r["job"]["variant"] == "wrong_image")
    before = copy.deepcopy(record)
    item = quilt_item(record)
    assert "synthetic_wrong_image_control" not in item.provenance
    assert item.provenance["source_image_sha256"] == record["job"]["image_sha256"]
    assert record == before
    from merit_feddg.compact_evidence import compact_prompt, compact_records
    prompt = compact_prompt("Question?", compact_records((item,)), columns=False)
    assert record["text"] in prompt
    assert "wrong_image" not in prompt
