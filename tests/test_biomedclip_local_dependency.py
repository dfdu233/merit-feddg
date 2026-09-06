"""Local BiomedBERT config overrides without a second text-model download."""

import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from merit_feddg.experts.biomedclip import BiomedClipAdapter, _local_text_config_kwargs


def snapshot(tmp_path):
    config = {
        "model_cfg": {
            "embed_dim": 512,
            "vision_cfg": {"timm_model_name": "vit_base_patch16_224"},
            "text_cfg": {
                "hf_model_name": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
                "hf_tokenizer_name": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
                "hf_proj_type": "mlp",
                "hf_pooler_type": "cls_last_hidden_state_pooler",
                "context_length": 256,
                "tokenizer_kwargs": {"test_setting": True},
            },
        },
        "preprocess_cfg": {"mean": [.1, .2, .3]},
    }
    (tmp_path / "open_clip_config.json").write_text(json.dumps(config), encoding="utf-8")
    dependency = tmp_path / ".cache" / "merit-text-config"
    dependency.mkdir(parents=True)
    (dependency / "config.json").write_text('{"model_type":"bert"}', encoding="utf-8")
    return config, dependency


def test_local_override_preserves_complete_text_configuration_and_original_file(tmp_path):
    original, dependency = snapshot(tmp_path)
    kwargs = _local_text_config_kwargs(tmp_path)
    text = kwargs["text_cfg"]
    assert text["hf_model_name"] == str(dependency.resolve())
    assert text["hf_model_pretrained"] is False
    for name, value in original["model_cfg"]["text_cfg"].items():
        if name != "hf_model_name":
            assert text[name] == value
    assert json.loads((tmp_path / "open_clip_config.json").read_text()) == original


def test_missing_sidecar_preserves_legacy_loading_without_reading_extra_config(tmp_path):
    assert _local_text_config_kwargs(tmp_path) == {}
    assert _local_text_config_kwargs(tmp_path / "non-local-hub-id") == {}


def test_adapter_forwards_text_cfg_override_to_openclip_and_keeps_tokenizer_local(
    tmp_path, monkeypatch
):
    _, dependency = snapshot(tmp_path)
    calls = []
    fake_openclip = ModuleType("open_clip")

    class Model:
        def to(self, device):
            return self

        def eval(self):
            return self

    def create(model_name, **model_kwargs):
        calls.append(("model", model_name, model_kwargs))
        assert model_kwargs["text_cfg"]["hf_model_name"] == str(dependency.resolve())
        assert model_kwargs["text_cfg"]["hf_model_pretrained"] is False
        return Model(), object()

    def tokenizer(model_name):
        calls.append(("tokenizer", model_name))
        return object()

    fake_openclip.create_model_from_pretrained = create
    fake_openclip.get_tokenizer = tokenizer
    fake_torch = ModuleType("torch")
    fake_torch.device = lambda name: name
    fake_torch.cuda = SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "open_clip", fake_openclip)
    BiomedClipAdapter(str(tmp_path))
    assert calls[0][1] == f"local-dir:{tmp_path.resolve()}"
    assert calls[1] == ("tokenizer", f"local-dir:{tmp_path.resolve()}")


@pytest.mark.parametrize("bad_config", [{}, {"model_cfg": {"text_cfg": {}}}, []])
def test_invalid_local_override_does_not_silently_fall_back_to_remote(tmp_path, bad_config):
    snapshot(tmp_path)
    (tmp_path / "open_clip_config.json").write_text(json.dumps(bad_config), encoding="utf-8")
    with pytest.raises(ValueError, match="complete HF text_cfg"):
        _local_text_config_kwargs(tmp_path)


def test_real_transformers_resolves_prepared_architecture_locally(tmp_path):
    transformers = pytest.importorskip("transformers")
    snapshot(tmp_path)
    text = _local_text_config_kwargs(tmp_path)["text_cfg"]
    config = transformers.AutoConfig.from_pretrained(text["hf_model_name"], local_files_only=True)
    assert config.model_type == "bert"
