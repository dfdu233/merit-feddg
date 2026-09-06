"""LLaVA-Med protocol tests; no clinical weights or network are required."""

import json
from types import SimpleNamespace

import pytest
from PIL import Image

import merit_feddg.llava_generalist as module


def checkpoint(tmp_path, **changes):
    model = tmp_path / "llava-med-v1.5-mistral-7b"
    model.mkdir()
    visual = model / "clip"
    visual.mkdir()
    config = {
        "model_type": "llava_mistral",
        "architectures": ["LlavaMistralForCausalLM"],
        "mm_vision_tower": "clip",
        "mm_use_im_patch_token": False,
    }
    config.update(changes)
    (model / "config.json").write_text(json.dumps(config))
    (model / "model.safetensors").write_bytes(b"test-weight-placeholder")
    (visual / "config.json").write_text('{"model_type":"clip_vision_model"}')
    (visual / "preprocessor_config.json").write_text("{}")
    (visual / "model.safetensors").write_bytes(b"test-vision-placeholder")
    return model, visual


def test_preflight_resolves_relative_visual_dependency_without_network(tmp_path, monkeypatch):
    model, visual = checkpoint(tmp_path)
    hub = pytest.importorskip("huggingface_hub")
    monkeypatch.setattr(
        hub, "snapshot_download", lambda *a, **kw: pytest.fail("local inference must not use Hub")
    )
    result = module.inspect_llava_checkpoint(str(model))
    assert result["model_path"] == str(model.resolve())
    assert result["vision_tower_path"] == str(visual.resolve())


@pytest.mark.parametrize("model_type", ["llava", "llama", "llava_llama", "mistral"])
def test_hf_converted_and_legacy_formats_are_not_guessed(tmp_path, model_type):
    model, _ = checkpoint(tmp_path, model_type=model_type)
    with pytest.raises(ValueError, match="official v1.5 Mistral"):
        module.inspect_llava_checkpoint(str(model))


def test_visual_dependencies_are_checked_before_language_model_load(tmp_path):
    model, visual = checkpoint(tmp_path)
    (visual / "model.safetensors").unlink()
    with pytest.raises(FileNotFoundError, match="No complete"):
        module.inspect_llava_checkpoint(str(model))


def test_missing_visual_cache_has_actionable_error_and_local_only(tmp_path, monkeypatch):
    model, _ = checkpoint(tmp_path, mm_vision_tower="openai/clip-vit-large-patch14-336")
    hub = pytest.importorskip("huggingface_hub")
    calls = []

    def missing(repo, **kwargs):
        calls.append((repo, kwargs))
        raise OSError("not cached")

    monkeypatch.setattr(hub, "snapshot_download", missing)
    with pytest.raises(FileNotFoundError, match="vision_tower_path"):
        module.inspect_llava_checkpoint(str(model))
    assert calls == [("openai/clip-vit-large-patch14-336", {"local_files_only": True})]


def test_incomplete_main_shards_fail_preflight(tmp_path):
    model, _ = checkpoint(tmp_path)
    (model / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": "model-00002-of-00002.safetensors"}})
    )
    with pytest.raises(FileNotFoundError, match="missing weight"):
        module.inspect_llava_checkpoint(str(model))


def test_bad_source_path_fails_without_import_side_effects(tmp_path):
    with pytest.raises(FileNotFoundError, match="official LLaVA-Med v1.5 source"):
        module._llava_runtime(str(tmp_path))


class Tokenizer:
    eos_token_id = 2
    bos_token_id = 1

    def __len__(self):
        return 32

    def add_tokens(self, values, special_tokens):
        self.added = values

    def decode(self, tokens, skip_special_tokens=True):
        return " ".join(
            {7: "lung", 8: "tissue", 9: "0", 10: "1"}.get(int(i), str(i))
            for i in tokens
            if int(i) not in {0, 1, 2}
        )

    def encode(self, text, add_special_tokens=False):
        return {"0": [9], "1": [10]}[text]


@pytest.fixture
def fake_backend(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    model_path, visual_path = checkpoint(tmp_path)
    logs = {"load": [], "tokenize": [], "generation": []}
    tokenizer = Tokenizer()

    class Model:
        def __init__(self, config):
            self.config = config
            self.generation_config = SimpleNamespace(eos_token_id=2)
            self.embedding = torch.nn.Embedding(32, 8)
            self.tower = SimpleNamespace(
                is_loaded=False,
                device=torch.device("cpu"),
                dtype=torch.float32,
                image_processor="local-clip-processor",
                load_model=self.load_visual,
                to=lambda **kwargs: logs.setdefault("visual_device", kwargs),
            )

        def load_visual(self):
            logs["loaded_visual_path"] = self.config.mm_vision_tower
            self.tower.is_loaded = True

        def get_input_embeddings(self):
            return self.embedding

        def get_vision_tower(self):
            return self.tower

        def resize_token_embeddings(self, size):
            logs["resized"] = size

        def eval(self):
            return self

        def generate(self, **kwargs):
            logs["generation"].append(kwargs)
            ids = [7, 2] if kwargs["max_new_tokens"] > 1 else [8]
            if "prefix_allowed_tokens_fn" in kwargs:
                # Official inputs_embeds generation may start with a synthetic
                # BOS on older Transformers, not with the image/text IDs.
                constrain = kwargs["prefix_allowed_tokens_fn"]
                chosen = constrain(0, torch.tensor([1]))[0]
                assert constrain(0, torch.tensor([1, chosen])) == [2]
                ids = [chosen, 2]
            # Include a synthetic BOS to catch prompt-length slicing bugs.
            return SimpleNamespace(
                sequences=torch.tensor([[1, *ids]]),
                scores=tuple(torch.zeros(1, 32) for _ in ids),
            )

        def compute_transition_scores(self, sequences, scores, beam_indices, normalize_logits):
            assert normalize_logits
            return torch.full((1, len(scores)), -0.25)

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(path, **kwargs):
            logs["tokenizer_load"] = path, kwargs
            return tokenizer

    class Config:
        @staticmethod
        def from_pretrained(path, **kwargs):
            logs["config_load"] = path, kwargs
            return SimpleNamespace(**json.loads((model_path / "config.json").read_text()))

    class ModelLoader:
        @staticmethod
        def from_pretrained(path, **kwargs):
            logs["load"].append((path, kwargs))
            return Model(kwargs["config"])

    class Conversation:
        roles = ("USER", "ASSISTANT")

        def copy(self):
            result = Conversation()
            result.messages = []
            return result

        def append_message(self, role, message):
            self.messages.append((role, message))

        def get_prompt(self):
            return "[INST] " + self.messages[0][1] + " [/INST]"

    def image_tokenize(prompt, token, image_token_index, return_tensors):
        logs["tokenize"].append(prompt)
        assert token is tokenizer and image_token_index == -200 and return_tensors == "pt"
        return torch.tensor([1, -200, 5, 6, 4])

    runtime = SimpleNamespace(
        torch=torch,
        AutoTokenizer=AutoTokenizer,
        Config=Config,
        Model=ModelLoader,
        constants=SimpleNamespace(
            DEFAULT_IMAGE_TOKEN="<image>",
            DEFAULT_IMAGE_PATCH_TOKEN="<im_patch>",
            DEFAULT_IM_START_TOKEN="<im_start>",
            DEFAULT_IM_END_TOKEN="<im_end>",
            IMAGE_TOKEN_INDEX=-200,
        ),
        conv_templates={"mistral_instruct": Conversation()},
        tokenizer_image_token=image_tokenize,
        process_images=lambda images, processor, config: torch.ones(1, 3, 4, 4),
    )
    monkeypatch.setattr(module, "_llava_runtime", lambda source_path: runtime)
    backend = module.LlavaMedGeneralist(str(model_path), dtype="float32", device_map="cpu")
    return backend, logs, visual_path


def test_constructor_uses_explicit_offline_model_and_visual_paths(fake_backend):
    backend, logs, visual_path = fake_backend
    options = logs["load"][0][1]
    assert options["local_files_only"] is True
    assert options["device_map"] == {"": "cpu"}
    assert options["config"].mm_vision_tower == str(visual_path.resolve())
    assert logs["loaded_visual_path"] == str(visual_path.resolve())
    assert logs["tokenizer_load"][1]["trust_remote_code"] is False
    assert logs["config_load"][1]["local_files_only"] is True
    assert backend.processor.tokenizer is backend.tokenizer


def test_generated_only_output_is_not_sliced_by_input_length(fake_backend):
    backend, logs, _ = fake_backend
    result = backend.generate_with_usage(Image.new("RGB", (16, 12)), "<image> What organ?", 8)
    assert result == {"text": "lung", "input_tokens": 5, "output_tokens": 2, "token_ids": [7, 2]}
    assert logs["tokenize"][-1].count("<image>") == 1
    assert logs["generation"][-1]["image_sizes"] == [(16, 12)]
    assert logs["generation"][-1]["images"].shape == (1, 3, 4, 4)


def test_answer_session_preserves_exact_prefix_ids_and_image(fake_backend):
    backend, logs, _ = fake_backend
    session = backend.new_answer_session(Image.new("RGB", (20, 10)), "Describe the image")
    first = session.propose((), 1, 1)[0]
    second = session.propose(first.tokens + (11, 12), 1, 3)[0]
    assert first.tokens == (8,) and first.finished is False
    assert second.tokens == (7, 2) and second.finished is True
    assert logs["generation"][-1]["inputs"].tolist() == [[1, -200, 5, 6, 4, 8, 11, 12]]
    assert logs["generation"][-1]["attention_mask"].tolist() == [[1] * 8]
    assert logs["generation"][-1]["images"] is logs["generation"][-2]["images"]
    # New evidence gets a fresh session; committed IDs are still appended raw.
    fresh = backend.new_answer_session(Image.new("RGB", (20, 10)), "New evidence")
    fresh.propose((8, 11, 12), 1, 1)
    assert logs["generation"][-1]["inputs"][0, -3:].tolist() == [8, 11, 12]
    assert len(logs["tokenize"]) == 2  # never tokenize decoded assistant text


def test_finite_tool_actions_are_constrained_not_medical_answers(fake_backend):
    pytest.importorskip("merit_feddg.constrained")
    backend, logs, _ = fake_backend
    result = backend.generate_with_usage(
        Image.new("RGB", (4, 4)), "Choose action 0 or 1", 4, allowed_texts=["0", "1"]
    )
    assert result["text"] in {"0", "1"}
    assert "prefix_allowed_tokens_fn" in logs["generation"][-1]
    backend.generate(Image.new("RGB", (4, 4)), "What do you see?", 4)
    assert "prefix_allowed_tokens_fn" not in logs["generation"][-1]


def test_unsupported_beam_path_is_explicit(fake_backend):
    backend, _, _ = fake_backend
    session = backend.new_answer_session(Image.new("RGB", (4, 4)), "Question")
    with pytest.raises(ValueError, match="greedy blocks"):
        session.propose((), 3, 4)


def test_context_overflow_never_silently_truncates_committed_prefix(fake_backend):
    backend, logs, _ = fake_backend
    backend.model.config.tokenizer_model_max_length = 7
    backend.model.tower.num_patches = 3
    session = backend.new_answer_session(Image.new("RGB", (4, 4)), "Question")
    with pytest.raises(ValueError, match="silent prefix truncation is forbidden"):
        session.propose((7, 8), 1, 2)
    assert not logs["generation"]


def test_real_tiny_mistral_inputs_embeds_continuation_has_visual_context():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "MistralForCausalLM"):
        pytest.skip("Mistral is unavailable in this test environment")
    from transformers import MistralConfig, MistralForCausalLM

    class TinyImageMistral(MistralForCausalLM):
        """Exercise the official generate contract with actual random CPU weights.

        This does not impersonate a trained LLaVA-Med or a CLIP forward: one image
        placeholder embedding is replaced by a pixel statistic solely to test
        the inputs_embeds protocol and exact-prefix continuation.
        """

        def generate(self, inputs=None, images=None, image_sizes=None, **kwargs):
            self.last_input_ids = inputs.detach().clone()
            safe = inputs.clamp_min(3)
            embeds = self.get_input_embeddings()(safe)
            embeds = embeds.clone()
            embeds[inputs == -200] = images.float().mean().to(embeds.dtype)
            self.last_visual_embedding = embeds[inputs == -200].detach().clone()
            return super().generate(inputs_embeds=embeds, **kwargs)

    torch.manual_seed(9)
    model = TinyImageMistral(
        MistralConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=128,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
    ).eval()
    backend = object.__new__(module.LlavaMedGeneralist)
    backend.model, backend.torch, backend.tokenizer = model, torch, Tokenizer()
    backend.runtime = SimpleNamespace(constants=SimpleNamespace(IMAGE_TOKEN_INDEX=-200))
    model.get_vision_tower = lambda: SimpleNamespace(num_patches=1)
    backend._inputs = lambda image, prompt: {
        "inputs": torch.tensor([[1, -200, 5, 6]]),
        "images": torch.full((1, 3, 4, 4), 0.25),
        "image_sizes": [(4, 4)],
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
    }
    session = backend.new_answer_session(Image.new("RGB", (4, 4)), "Question")
    block = session.propose((7, 8), 1, 2)[0]
    assert 1 <= len(block.tokens) <= 2
    assert model.last_input_ids.tolist() == [[1, -200, 5, 6, 7, 8]]
    assert torch.allclose(model.last_visual_embedding, torch.full((1, 16), 0.25))
    assert block.log_probability <= 0
