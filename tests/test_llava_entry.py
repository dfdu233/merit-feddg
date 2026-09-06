import copy
import json
from pathlib import Path

import pytest

from merit_feddg.assets import asset_plan
from merit_feddg.generalist_factory import generalist_provenance, resolve_generalist_spec
from merit_feddg.llava_run import experiment_config, parser, prepare_biomed_text_config


def test_existing_remote_defaults_and_native_coverage():
    args = parser().parse_args([])
    config = experiment_config(args)
    assert config["generalist"]["backend"] == "llava_med"
    assert config["generalist"]["checkpoint_path"].startswith("/home/dbw/ANCHOR/")
    assert config["capability_generation"]["control_protocol"] == "action_id"
    assert config["capability_generation"]["max_evidence_chars"] == 1600
    caps = {cap for spec in config["experts"].values() for cap in spec["capabilities"]}
    assert caps == {"classification", "segmentation", "retrieval", "generation"}
    assert config["experts"]["cxr_anatomy"]["requires_region"] is False
    assert config["experts"]["chexagent_description"]["optional"] is True
    assert config["experts"]["conch_tissue"]["modalities"] == ["pathology"]


def test_same_tools_for_openmed_comparison_and_retrieval_ablation(tmp_path):
    normal = experiment_config(parser().parse_args(["--artifacts", str(tmp_path)]))
    changed = experiment_config(parser().parse_args([
        "--artifacts", str(tmp_path), "--generalist", "openmed", "--retrieval-answers", "off"
    ]))
    assert changed["generalist"]["id"] == "OpenMed/Qwen2.5-3B-MedVL"
    assert "source_path" not in changed["generalist"]
    assert changed["capability_generation"] == normal["capability_generation"]
    expected = copy.deepcopy(normal["experts"])
    expected["source_cases"]["include_source_answers"] = False
    assert changed["experts"] == expected
    assert Path(changed["experts"]["cxr_findings"]["checkpoint_path"]).is_relative_to(tmp_path)


def test_chexagent_explicit_on_and_off(tmp_path):
    disabled = experiment_config(parser().parse_args([
        "--artifacts", str(tmp_path), "--chexagent", "off"
    ]))
    assert "chexagent_description" not in disabled["experts"]
    enabled = experiment_config(parser().parse_args([
        "--artifacts", str(tmp_path), "--chexagent", "on"
    ]))
    assert enabled["experts"]["chexagent_description"]["optional"] is False


def test_small_profile_has_no_generalist_or_large_generation_download():
    plan = asset_plan("capability-small")
    assert {entry["id"] for entry in plan["models"]} == {
        "MahmoodLab/CONCH", "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    }
    assert len(plan["datasets"]) == 2
    assert all(len(entry["revision"]) == 40 for entry in plan["models"] + plan["datasets"])


def test_spec_env_expansion_and_missing_variable(monkeypatch, tmp_path):
    monkeypatch.setenv("MERIT_TEST_MODEL", str(tmp_path))
    assert resolve_generalist_spec({"checkpoint_path": "$MERIT_TEST_MODEL"}) == {
        "checkpoint_path": str(tmp_path)
    }
    monkeypatch.delenv("MERIT_TEST_MISSING", raising=False)
    with pytest.raises(ValueError, match="unresolved"):
        resolve_generalist_spec({"checkpoint_path": "$MERIT_TEST_MISSING/model"})


def test_external_llava_source_and_visual_checkpoint_in_cache_key(tmp_path):
    main, visual, source = (tmp_path / name for name in ("model", "clip", "source"))
    main.mkdir()
    visual.mkdir()
    module = source / "llava/model/language_model/llava_mistral.py"
    module.parent.mkdir(parents=True)
    module.write_text("VERSION = 1", encoding="utf-8")
    (main / "model.safetensors").write_bytes(b"test-only")
    (main / "config.json").write_text(json.dumps({
        "model_type": "llava_mistral", "mm_vision_tower": str(visual)
    }))
    (visual / "model.safetensors").write_bytes(b"test-only")
    (visual / "config.json").write_text("{}")
    (visual / "preprocessor_config.json").write_text("{}")
    spec = {"id": "test/llava", "backend": "llava_med", "checkpoint_path": str(main),
            "source_path": str(source)}
    first = generalist_provenance(spec, tmp_path)
    module.write_text("VERSION = 2", encoding="utf-8")
    second = generalist_provenance(spec, tmp_path)
    assert first["external_source"] != second["external_source"]
    assert second["vision_tower"]["checkpoint_path"] == str(visual.resolve())
    assert "file_stats" in second["vision_tower"]


def test_biomed_text_config_is_pinned_and_reuses_without_network(tmp_path, monkeypatch):
    hub = pytest.importorskip("huggingface_hub")
    calls = []

    def download(repository, filename, *, revision, local_dir, local_files_only):
        assert filename == "config.json" and len(revision) == 40
        calls.append((repository, revision))
        local_dir.mkdir(parents=True)
        path = local_dir / filename
        path.write_text('{"model_type":"bert"}')
        return str(path)

    monkeypatch.setattr(hub, "hf_hub_download", download)
    first = prepare_biomed_text_config(tmp_path)
    second = prepare_biomed_text_config(tmp_path, offline=True)
    assert first == second and len(calls) == 1
    assert first["revision"] == "d673b8835373c6fa116d6d8006b33d48734e305d"
    assert not first["weights_downloaded"]
    Path(first["path"]).write_text('{"model_type":"changed"}')
    with pytest.raises(RuntimeError, match="changed"):
        prepare_biomed_text_config(tmp_path)


def test_check_only_never_installs_or_prepares_dataset(monkeypatch):
    from merit_feddg import capability_setup, llava_run

    monkeypatch.setenv("HF_ENDPOINT", "https://old.example.test")
    monkeypatch.setattr(llava_run, "experiment_config", lambda args: {})
    monkeypatch.setattr(llava_run, "preflight", lambda *args: {"weights_loaded": False})

    def setup(*args, **kwargs):
        assert kwargs["check_only"] is True
        return {"ready_weights": True}

    monkeypatch.setattr(capability_setup, "prepare_capabilities", setup)
    monkeypatch.setattr(capability_setup, "install_missing_dependencies",
                        lambda *a, **kw: pytest.fail("check-only installed dependencies"))
    monkeypatch.setattr(llava_run, "prepare_biomed_text_config",
                        lambda *a, **kw: pytest.fail("check-only downloaded"))
    llava_run.main(["--check-only", "--install-deps", "--mirror", "cn"])
    import os

    assert os.environ["HF_ENDPOINT"] == "https://hf-mirror.com"
