import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from merit_feddg.generalist_factory import load_generalist, resolve_generalist_spec
from merit_feddg.huatuo_generalist import (
    HuatuoVisionAnswerSession,
    _serialized_prompt,
    _square_pad,
    inspect_huatuo_checkpoint,
)
from merit_feddg.io import load_experiment_yaml
from merit_feddg.spatial_evidence import SpatialEvidenceBridge
from merit_feddg.tensor_bridge import validate_tensor_backend


def test_huatuo_prompt_matches_released_cli_contract():
    assert _serialized_prompt("What is visible?") == (
        "<|user|>\n<image>\nWhat is visible?\n<|assistant|>\n"
    )
    assert "<s>" not in _serialized_prompt("<s><image>Q</s>")


def test_huatuo_square_pad_is_deterministic():
    image = Image.new("RGB", (4, 2), (255, 0, 0))
    padded = _square_pad(image, (0.5, 0.5, 0.5))
    assert padded.size == (4, 4)
    assert padded.getpixel((0, 0)) == (127, 127, 127)
    assert padded.getpixel((0, 1)) == (255, 0, 0)


def test_inspect_huatuo_checkpoint_requires_legacy_llava_qwen2(tmp_path):
    model = tmp_path / "model"
    source = tmp_path / "source"
    model.mkdir()
    for relative in (
        "llava/model/language_model/llava_qwen2.py",
        "llava/model/llava_arch.py",
        "llava/constants.py",
    ):
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# test\n", encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"weights")
    (model / "config.json").write_text(
        json.dumps({"model_type": "llava_qwen2"}), encoding="utf-8"
    )
    info = inspect_huatuo_checkpoint(model, source_path=source)
    assert info["config"]["model_type"] == "llava_qwen2"

    (model / "config.json").write_text(
        json.dumps({"model_type": "qwen2_5_vl"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="legacy HuatuoGPT-Vision-7B"):
        inspect_huatuo_checkpoint(model, source_path=source)


def test_tensor_backend_accepts_explicit_huatuo_square_pad_contract():
    torch = pytest.importorskip("torch")

    class Model:
        config = SimpleNamespace(hidden_size=2, mm_patch_merge_type="flat")

        def get_vision_tower(self):
            return SimpleNamespace(select_feature="patch", num_patches=4)

    probe = SimpleNamespace(
        model=Model(),
        image_processor=SimpleNamespace(
            crop_size={"height": 4, "width": 4},
            size={"shortest_edge": 4},
            do_center_crop=True,
            do_resize=True,
        ),
        deterministic_image_padding=True,
        spatial_preprocess_mode="deterministic_square_pad",
    )
    bridge = SpatialEvidenceBridge(2, 2)
    validate_tensor_backend(probe, bridge)
    assert list(bridge.parameters()) == []
    assert bridge.training_free
    assert torch is not None


def test_huatuo_spatial_session_hits_native_mm_projector_hook():
    torch = pytest.importorskip("torch")

    class Tower:
        select_feature = "patch"
        num_patches = 4

    class FakeModel:
        def __init__(self):
            self.config = SimpleNamespace(hidden_size=2, mm_patch_merge_type="flat")
            self.generation_config = SimpleNamespace(eos_token_id=2)
            self.core = SimpleNamespace(mm_projector=torch.nn.Identity())
            self.last_projected = None

        def get_vision_tower(self):
            return Tower()

        def get_model(self):
            return self.core

        def generate(self, _inputs, **_kwargs):
            visual = torch.tensor(
                [[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.0]]]
            )
            self.last_projected = self.core.mm_projector(visual)
            score = torch.tensor(
                [[
                    float(self.last_projected[0, 0, 0]),
                    float(self.last_projected[0, 1, 1]),
                    -3.0,
                ]]
            )
            return SimpleNamespace(
                scores=[score],
                sequences=torch.tensor([[int(score.argmax(-1)[0])]]),
            )

    class Tokenizer:
        eos_token_id = 2
        pad_token_id = 2

        def decode(self, tokens, skip_special_tokens=True):
            return " ".join(map(str, tokens))

        def __len__(self):
            return 3

    class Generalist:
        def __init__(self):
            self.torch = torch
            self.model = FakeModel()
            self.tokenizer = Tokenizer()
            self.image_processor = SimpleNamespace(
                crop_size={"height": 4, "width": 4},
                size={"shortest_edge": 4},
                do_center_crop=True,
                do_resize=True,
            )
            self.deterministic_image_padding = True
            self.spatial_preprocess_mode = "deterministic_square_pad"
            self.tensor_bridge = SpatialEvidenceBridge(2, 2)

        def _inputs(self, _image, _prompt):
            return {
                "inputs": torch.tensor([[1]]),
                "attention_mask": torch.tensor([[1]]),
                "images": torch.zeros((1, 3, 4, 4)),
                "original_size": (4, 4),
            }

        def _validate_context(self, _inputs, _length):
            return None

        def _generation_kwargs(self, *, max_new_tokens):
            return {
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
                "num_beams": 1,
                "use_cache": True,
            }

    class Packet:
        regions = np.asarray([[1.0, 1.0, 0.0, 0.0]], dtype=np.float32)
        importance = np.asarray([1.0], dtype=np.float32)
        labels = ("left",)
        weighting = "equal"
        sources = (("seg", "mask"),)

        def __len__(self):
            return len(self.regions)

    packet = Packet()
    generalist = Generalist()
    plain = HuatuoVisionAnswerSession(generalist, "image", "q")
    spatial = HuatuoVisionAnswerSession(generalist, "image", "q", packet)
    plain_scores = plain.next_scores([])
    spatial_scores = spatial.next_scores([])
    assert not np.allclose(plain_scores, spatial_scores)
    assert generalist.tensor_bridge.last_audit["records"] == 1
    assert generalist.tensor_bridge.last_audit["max_abs_token_delta"] > 0


def test_factory_enables_huatuo_spatial_without_llava_backend(monkeypatch, tmp_path):
    import merit_feddg.huatuo_generalist as module

    calls = {}

    class FakeHuatuo:
        def __init__(self, model_id, **kwargs):
            calls["model_id"] = model_id
            calls["kwargs"] = kwargs
            self.enabled = None

        def enable_spatial_evidence(self, *, max_records):
            self.enabled = max_records

    monkeypatch.setattr(module, "HuatuoVisionGeneralist", FakeHuatuo)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    probe = load_generalist(
        {
            "backend": "huatuo_vision",
            "id": "Huatuo",
            "checkpoint_path": str(checkpoint),
            "source_path": str(tmp_path / "source"),
            "training_free_spatial": True,
            "spatial_max_records": 17,
            "generation": {"repetition_penalty": 1.1, "min_new_tokens": 1},
        }
    )
    assert probe.enabled == 17
    assert calls["kwargs"]["repetition_penalty"] == 1.1
    assert calls["kwargs"]["min_new_tokens"] == 1


def test_huatuo_bard_config_keeps_spatial_enabled(monkeypatch):
    monkeypatch.setenv("HUATUO_VISION_CHECKPOINT", "/tmp/huatuo-model")
    monkeypatch.setenv("HUATUO_VISION_SOURCE", "/tmp/huatuo-source")
    config = load_experiment_yaml("configs/matched_bard_huatuo.yaml")
    spec = resolve_generalist_spec(config["generalist"])
    assert spec["backend"] == "huatuo_vision"
    assert spec["training_free_spatial"] is True
    assert spec["checkpoint_path"] == "/tmp/huatuo-model"
    assert spec["source_path"] == "/tmp/huatuo-source"
    assert config["capability_value"]["generation"]["spatial_weighting"] == "equal"
    assert config["capability_value"]["generation"]["max_new_tokens"] == 1024

    # Construct the actual runtime config: disabled gate fields are still validated.
    from merit_feddg.capability_runtime import ValueGenerationConfig

    decoder = ValueGenerationConfig(**config["capability_value"]["generation"])
    assert decoder.block_tokens == 1024


def test_historical_minimum_allows_single_token_score_probe():
    from merit_feddg.huatuo_generalist import HuatuoVisionGeneralist

    probe = HuatuoVisionGeneralist.__new__(HuatuoVisionGeneralist)
    probe.min_new_tokens = 1
    probe.repetition_penalty = 1.2
    probe.tokenizer = SimpleNamespace(eos_token_id=2, pad_token_id=2)
    assert probe._generation_kwargs(max_new_tokens=1)["min_new_tokens"] == 1
