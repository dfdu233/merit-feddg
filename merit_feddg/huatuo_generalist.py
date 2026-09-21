"""Frozen HuatuoGPT-Vision-7B backend with native spatial-evidence support.

This targets the released LLaVA-Qwen2 HuatuoGPT-Vision-7B checkpoint used by
the existing MERIT Huatuo experiments.  It intentionally does not target the
newer Qwen2.5-VL rewrite of the upstream repository.

The backend preserves Huatuo's original <|user|>/<|assistant|> serialization,
single-image square padding, and native prepare_inputs_labels_for_multimodal_new
path.  Spatial evidence is injected only through the existing parameter-free
MERIT spatial operator at the model's mm_projector output.
"""
from __future__ import annotations

import importlib
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from .block_decode import Block
from .experts.base import load_rgb


def inspect_huatuo_checkpoint(model_id, *, source_path):
    """Validate the legacy HuatuoGPT-Vision LLaVA-Qwen2 runtime without loading it."""
    model_path = Path(str(model_id)).expanduser().resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"HuatuoGPT-Vision checkpoint is missing: {model_path}")
    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"HuatuoGPT-Vision config is missing: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("model_type") != "llava_qwen2":
        raise ValueError(
            "huatuo_vision backend requires the legacy HuatuoGPT-Vision-7B "
            f"LLaVA-Qwen2 checkpoint; found model_type={config.get('model_type')!r}"
        )
    source = Path(source_path).expanduser().resolve()
    required = (
        source / "llava/model/language_model/llava_qwen2.py",
        source / "llava/model/llava_arch.py",
        source / "llava/constants.py",
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(
                "Huatuo source_path must contain the legacy LLaVA-Qwen2 source; "
                f"missing {path}"
            )
    # Reuse the repository's local-weight validation instead of accepting a
    # partial HF cache that would fail only after a multi-GB model load.
    from .llava_generalist import _validate_weights

    _validate_weights(model_path)
    return {
        "model_path": str(model_path),
        "source_path": str(source),
        "config": config,
    }


def _huatuo_runtime(source_path):
    source = Path(source_path).expanduser().resolve()
    expected = source / "llava/model/language_model/llava_qwen2.py"
    if not expected.is_file():
        raise FileNotFoundError(f"legacy Huatuo LLaVA-Qwen2 source not found: {expected}")

    imported = sys.modules.get("llava")
    if imported is not None:
        imported_path = Path(getattr(imported, "__file__", "")).resolve()
        if not imported_path.is_relative_to(source):
            raise RuntimeError(
                "Another llava package is already imported from "
                f"{imported_path}. Start a fresh process for HuatuoGPT-Vision."
            )
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    try:
        import torch
        from transformers import AutoTokenizer

        constants = importlib.import_module("llava.constants")
        models = importlib.import_module("llava.model.language_model.llava_qwen2")
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "Cannot import the legacy HuatuoGPT-Vision LLaVA-Qwen2 runtime. "
            "Use the existing Huatuo Python environment and matching source checkout."
        ) from exc
    return SimpleNamespace(
        torch=torch,
        AutoTokenizer=AutoTokenizer,
        Model=models.LlavaQwen2ForCausalLM,
        constants=constants,
    )


def _square_pad(image, mean):
    width, height = image.size
    if width == height:
        return image
    side = max(width, height)
    background = tuple(int(float(value) * 255) for value in mean)
    canvas = Image.new("RGB", (side, side), background)
    canvas.paste(image, ((side - width) // 2, (side - height) // 2))
    return canvas


def _tokenizer_image_token(tokenizer, prompt, image_token_index, torch):
    """Exact tokenizer helper used by the released HuatuoGPT-Vision cli.py."""
    chunks = [
        tokenizer(chunk, add_special_tokens=False).input_ids
        for chunk in prompt.split("<image>")
    ]

    def insert_separator(values, separator):
        return [
            item
            for group in zip(values, [separator] * len(values))
            for item in group
        ][:-1]

    ids = []
    offset = 0
    if chunks and chunks[0] and chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        ids.append(chunks[0][0])
    for values in insert_separator(chunks, [image_token_index] * (offset + 1)):
        ids.extend(values[offset:])
    return torch.tensor(ids, dtype=torch.long)


def _serialized_prompt(prompt):
    text = str(prompt)
    for token in ("<image>", "<s>", "</s>"):
        text = text.replace(token, "")
    text = text.strip()
    return f"<|user|>\n<image>\n{text}\n<|assistant|>\n"


def _generated_rows(output, model, tokenizer):
    steps = len(output.scores)
    if steps < 1 or output.sequences.shape[1] < steps:
        raise RuntimeError("HuatuoGPT-Vision returned no generated token scores")
    generated = output.sequences[:, -steps:]
    eos = model.generation_config.eos_token_id
    if eos is None:
        eos = tokenizer.eos_token_id
    eos_ids = {
        int(value)
        for value in (eos if isinstance(eos, (list, tuple)) else [eos])
        if value is not None
    }
    rows = []
    for row_index, ids in enumerate(generated.detach().cpu().tolist()):
        end = next(
            (index + 1 for index, token in enumerate(ids) if token in eos_ids),
            len(ids),
        )
        ids = tuple(int(token) for token in ids[:end])
        if not ids:
            raise RuntimeError("HuatuoGPT-Vision returned an empty continuation")
        logps = []
        for step, token in enumerate(ids):
            scores = output.scores[step][row_index].float()
            value = scores.log_softmax(-1)[token]
            if not bool(value.isfinite()):
                raise RuntimeError("HuatuoGPT-Vision returned a nonfinite token score")
            logps.append(float(value))
        rows.append((ids, float(np.mean(logps)), bool(ids[-1] in eos_ids)))
    return rows


class HuatuoVisionGeneralist:
    """Legacy HuatuoGPT-Vision-7B receiver compatible with MERIT/BARD."""

    def __init__(
        self,
        model_id: str,
        *,
        source_path: str,
        dtype: str = "bfloat16",
        device: str = "cuda",
        deterministic_image_padding: bool = True,
        repetition_penalty: float = 1.0,
        min_new_tokens: int = 0,
    ):
        if not deterministic_image_padding:
            raise ValueError(
                "Huatuo backend requires deterministic square padding; "
                "the released cli.py always square-pads its single image"
            )
        if dtype not in {"float16", "bfloat16", "float32"}:
            raise ValueError("Huatuo dtype must be float16, bfloat16 or float32")
        if (
            not np.isfinite(repetition_penalty)
            or repetition_penalty <= 0
            or type(min_new_tokens) is not int
            or min_new_tokens < 0
        ):
            raise ValueError("invalid Huatuo deterministic generation settings")

        info = inspect_huatuo_checkpoint(model_id, source_path=source_path)
        runtime = _huatuo_runtime(source_path)
        torch = runtime.torch
        torch_dtype = getattr(torch, dtype)
        model, loading = runtime.Model.from_pretrained(
            info["model_path"],
            init_vision_encoder_from_ckpt=True,
            output_loading_info=True,
            torch_dtype=torch_dtype,
            attn_implementation="eager",
            local_files_only=True,
        )
        unexpected = loading.get("unexpected_keys", [])
        if any("vision_tower" not in key for key in unexpected):
            raise RuntimeError(
                "Huatuo checkpoint exposed unexpected non-vision weights: "
                + ", ".join(unexpected[:8])
            )
        tokenizer = runtime.AutoTokenizer.from_pretrained(
            info["model_path"],
            local_files_only=True,
            trust_remote_code=False,
        )
        tokenizer.pad_token_id = tokenizer.eos_token_id

        model = model.to(device).eval().requires_grad_(False)
        model.config.tokenizer_padding_side = "left"
        model.config.use_cache = True
        tower = model.get_vision_tower()
        if tower is None:
            raise RuntimeError("HuatuoGPT-Vision did not construct a vision tower")
        if not tower.is_loaded:
            tower.load_model()
            if hasattr(tower, "vision_tower"):
                tower.vision_tower = tower.vision_tower.from_pretrained(
                    info["model_path"], local_files_only=True
                )
        tower.to(device=model.get_input_embeddings().weight.device, dtype=torch_dtype)

        self.torch = torch
        self.runtime = runtime
        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = tower.image_processor
        self.processor = SimpleNamespace(tokenizer=tokenizer)
        self.checkpoint_info = info
        self.deterministic_image_padding = True
        self.spatial_preprocess_mode = "deterministic_square_pad"
        self.repetition_penalty = float(repetition_penalty)
        self.min_new_tokens = int(min_new_tokens)

    def _generation_kwargs(self, *, max_new_tokens):
        if max_new_tokens < 1 or self.min_new_tokens > max_new_tokens:
            raise ValueError("invalid Huatuo generation budget")
        return {
            "max_new_tokens": int(max_new_tokens),
            "min_new_tokens": self.min_new_tokens,
            "do_sample": False,
            "num_beams": 1,
            "repetition_penalty": self.repetition_penalty,
            "use_cache": True,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }

    def _inputs(self, image, prompt):
        if isinstance(image, (list, tuple)):
            if len(image) != 1:
                raise ValueError("Huatuo spatial/BARD backend supports one original image")
            image = image[0]
        original = load_rgb(image)
        padded = _square_pad(original, self.image_processor.image_mean)
        pixels = self.image_processor.preprocess(
            padded, return_tensors="pt"
        )["pixel_values"][0]
        tower = self.model.get_vision_tower()
        pixels = pixels.to(device=tower.device, dtype=tower.dtype).unsqueeze(0)

        serialized = _serialized_prompt(prompt)
        ids = _tokenizer_image_token(
            self.tokenizer,
            serialized,
            self.runtime.constants.IMAGE_TOKEN_INDEX,
            self.torch,
        ).unsqueeze(0)
        device = self.model.get_input_embeddings().weight.device
        ids = ids.to(device)
        if int((ids == self.runtime.constants.IMAGE_TOKEN_INDEX).sum()) != 1:
            raise RuntimeError("Huatuo prompt must contain exactly one image placeholder")
        return {
            "inputs": ids,
            "attention_mask": self.torch.ones_like(ids),
            "images": pixels,
            "original_size": original.size,
        }

    def _expanded_length(self, inputs):
        tower = self.model.get_vision_tower()
        patches = int(tower.num_patches)
        return int(inputs["inputs"].shape[1]) + patches - 1

    def _validate_context(self, inputs, max_new_tokens):
        expanded = self._expanded_length(inputs)
        limits = [
            int(value)
            for value in (
                getattr(self.model.config, "tokenizer_model_max_length", None),
                getattr(self.model.config, "max_position_embeddings", None),
            )
            if value is not None
        ]
        if limits and expanded + max_new_tokens > min(limits):
            raise ValueError("Huatuo image, evidence and answer exceed model context")

    def context_token_budget(self, image, prompt, reserve_tokens):
        if type(reserve_tokens) is not int or reserve_tokens < 0:
            raise ValueError("reserve_tokens must be a nonnegative integer")
        inputs = self._inputs(image, prompt)
        expanded = self._expanded_length(inputs)
        limits = [
            int(value)
            for value in (
                getattr(self.model.config, "tokenizer_model_max_length", None),
                getattr(self.model.config, "max_position_embeddings", None),
            )
            if value is not None
        ]
        if not limits:
            raise ValueError("Huatuo token-aware packing requires an explicit context limit")
        limit = min(limits)
        return {
            "input_tokens": expanded,
            "reserved_tokens": reserve_tokens,
            "context_limit": limit,
            "remaining_tokens": limit - expanded - reserve_tokens,
            "fits": expanded + reserve_tokens <= limit,
        }

    def new_answer_session(self, image, prompt):
        return HuatuoVisionAnswerSession(self, image, prompt)

    def enable_spatial_evidence(self, *, max_records=64):
        from .spatial_evidence import SpatialEvidenceBridge
        from .tensor_bridge import validate_tensor_backend

        tower = self.model.get_vision_tower()
        patches = int(tower.num_patches)
        grid = round(patches ** 0.5)
        if grid * grid != patches:
            raise ValueError("Huatuo spatial evidence requires a square patch grid")
        if not hasattr(self.model.get_model(), "mm_projector"):
            raise ValueError("Huatuo spatial evidence requires the native mm_projector")
        bridge = SpatialEvidenceBridge(
            int(self.model.config.hidden_size),
            grid,
            max_records,
        ).eval()
        validate_tensor_backend(self, bridge)
        self.model.eval().requires_grad_(False)
        self.tensor_bridge = bridge

    def tensor_packet(self, items, image, *, question="", weighting="relevance"):
        from .spatial_evidence import spatial_packet

        if not hasattr(self, "tensor_bridge"):
            raise ValueError("tensor evidence requires training_free_spatial")
        if isinstance(image, (list, tuple)):
            raise TypeError("Huatuo tensor evidence accepts one original image")
        return spatial_packet(
            items,
            load_rgb(image).size,
            grid_size=self.tensor_bridge.contract.grid_size,
            max_records=self.tensor_bridge.contract.max_records,
            question=question,
            weighting=weighting,
        )

    def new_tensor_answer_session(
        self,
        image,
        prompt,
        items,
        *,
        question="",
        weighting="relevance",
    ):
        return HuatuoVisionAnswerSession(
            self,
            image,
            prompt,
            self.tensor_packet(
                items,
                image,
                question=question,
                weighting=weighting,
            ),
        )

    def generate(
        self,
        image,
        prompt,
        max_new_tokens=64,
        logits_processor=None,
        *,
        allowed_texts=None,
    ):
        return self.generate_with_usage(
            image,
            prompt,
            max_new_tokens,
            logits_processor,
            allowed_texts=allowed_texts,
        )["text"]

    def generate_with_usage(
        self,
        image,
        prompt,
        max_new_tokens=64,
        logits_processor=None,
        *,
        allowed_texts=None,
    ):
        inputs = self._inputs(image, prompt)
        self._validate_context(inputs, max_new_tokens)
        options = self._generation_kwargs(max_new_tokens=max_new_tokens)
        if logits_processor is not None:
            options["logits_processor"] = [logits_processor]
        if allowed_texts is not None:
            from .constrained import finite_choice_constraint

            options["prefix_allowed_tokens_fn"] = finite_choice_constraint(
                self.tokenizer,
                allowed_texts,
                prompt_length=None,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        with self.torch.inference_mode():
            output = self.model.generate(
                inputs["inputs"],
                images=inputs["images"],
                return_dict_in_generate=True,
                output_scores=True,
                **options,
            )
        ids, _, _ = _generated_rows(output, self.model, self.tokenizer)[0]
        return {
            "text": self.tokenizer.decode(ids, skip_special_tokens=True).strip(),
            "input_tokens": self._expanded_length(inputs),
            "output_tokens": len(ids),
            "token_ids": list(ids),
        }


class HuatuoVisionAnswerSession:
    """Exact-prefix Huatuo receiver session used by NativeSession and BARD."""

    def __init__(self, generalist, image, prompt, tensor_packet=None):
        self.generalist = generalist
        self.inputs = generalist._inputs(image, prompt)
        self.tensor_packet = tensor_packet

    def _evidence_context(self):
        if self.tensor_packet is None:
            return nullcontext()
        from .tensor_bridge import tensor_projector_context

        return tensor_projector_context(self.generalist, self.tensor_packet)

    def decode(self, tokens):
        return self.generalist.tokenizer.decode(tokens, skip_special_tokens=True)

    @property
    def eos_ids(self):
        eos = self.generalist.model.generation_config.eos_token_id
        if eos is None:
            eos = self.generalist.tokenizer.eos_token_id
        return {
            int(value)
            for value in (eos if isinstance(eos, (list, tuple)) else [eos])
            if value is not None
        }

    def _generate(self, prefix, length):
        torch = self.generalist.torch
        inputs = dict(self.inputs)
        if prefix:
            extra = torch.tensor(
                [prefix],
                device=inputs["inputs"].device,
                dtype=inputs["inputs"].dtype,
            )
            inputs["inputs"] = torch.cat((inputs["inputs"], extra), dim=1)
            inputs["attention_mask"] = torch.ones_like(inputs["inputs"])
        self.generalist._validate_context(inputs, length)
        with torch.inference_mode(), self._evidence_context():
            return self.generalist.model.generate(
                inputs["inputs"],
                images=inputs["images"],
                return_dict_in_generate=True,
                output_scores=True,
                **self.generalist._generation_kwargs(max_new_tokens=length),
            )

    def next_scores(self, prefix):
        """Return Huatuo's production generation score vector at an exact prefix."""
        if not prefix:
            output = self._generate((), 1)
        else:
            calls = 0
            vocabulary = list(range(len(self.generalist.tokenizer)))

            def force_prefix(batch_id, _input_ids):
                nonlocal calls
                if batch_id != 0 or calls > len(prefix):
                    raise RuntimeError(
                        "Huatuo bounded replay requires one deterministic sequence"
                    )
                allowed = (
                    [int(prefix[calls])]
                    if calls < len(prefix)
                    else vocabulary
                )
                calls += 1
                return allowed

            options = self.generalist._generation_kwargs(
                max_new_tokens=len(prefix) + 1
            )
            options["prefix_allowed_tokens_fn"] = force_prefix
            with self.generalist.torch.inference_mode(), self._evidence_context():
                output = self.generalist.model.generate(
                    self.inputs["inputs"],
                    images=self.inputs["images"],
                    return_dict_in_generate=True,
                    output_scores=True,
                    **options,
                )
            if calls != len(prefix) + 1:
                raise RuntimeError(
                    "Huatuo production replay stopped before the requested prefix"
                )
        if len(output.scores) != len(prefix) + 1:
            raise RuntimeError("Huatuo next_scores did not cover the exact prefix")
        return output.scores[-1][0].detach().float().cpu().numpy()

    def propose(self, prefix, count, length):
        if count != 1:
            raise ValueError("Huatuo capability generation supports greedy blocks only")
        if length < 1:
            raise ValueError("block length must be positive")
        output = self._generate(prefix, length)
        ids, score, finished = _generated_rows(
            output,
            self.generalist.model,
            self.generalist.tokenizer,
        )[0]
        if len(ids) > length:
            raise RuntimeError("Huatuo exceeded its block token budget")
        return [Block(ids, self.decode(ids), score, finished)]
