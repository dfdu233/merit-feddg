"""Offline-first backend for the official LLaVA-Med v1.5 Mistral checkpoint.

This is the Microsoft ``llava_mistral`` format, not a Transformers-converted
``llava`` checkpoint. It uses the installed official source and its image-token
and conversation helpers. In particular, official ``generate`` delegates using
``inputs_embeds``: its return value must NOT be sliced by the text prompt length.
See microsoft/LLaVA-Med's llava_mistral.py and eval/model_vqa.py.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import numpy as np
from PIL import Image

from .block_decode import Block
from .experts.base import load_rgb


def _snapshot_path(identifier, *, local_files_only=True, relative_to=None):
    candidate = Path(str(identifier)).expanduser()
    if candidate.is_dir():
        return candidate.resolve()
    if relative_to and not candidate.is_absolute():
        relative = Path(relative_to) / candidate
        if relative.is_dir():
            return relative.resolve()
    if candidate.is_absolute() or str(identifier).startswith((".", "~")):
        raise FileNotFoundError(f"Local model directory does not exist: {identifier}")
    try:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(str(identifier), local_files_only=local_files_only)).resolve()
    except Exception as exc:
        raise FileNotFoundError(
            f"No usable local checkpoint for {identifier!r}. LLaVA-Med inference is "
            "offline by default. Supply an existing checkpoint_path or vision_tower_path; "
            "download missing files explicitly in a separate preparation step."
        ) from exc


def _validate_weights(directory):
    """Reject incomplete cached snapshots before loading a multi-GB language model."""
    for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index = directory / name
        if not index.is_file():
            continue
        weights = json.loads(index.read_text(encoding="utf-8")).get("weight_map", {})
        if not isinstance(weights, dict) or not weights:
            raise ValueError(f"Invalid model weight index: {index}")
        for relative in set(weights.values()):
            if not isinstance(relative, str):
                raise TypeError(f"Invalid weight filename in {index}")
            path = PurePosixPath(relative)
            if (
                path.is_absolute()
                or Path(relative).is_absolute()
                or ".." in path.parts
                or "\\" in relative
            ):
                raise ValueError(f"Unsafe weight filename in {index}: {relative}")
            item = directory / relative
            if not item.is_file() or item.stat().st_size == 0:
                raise FileNotFoundError(f"Incomplete model snapshot; missing weight: {item}")
        return
    if not any(
        (directory / name).is_file() and (directory / name).stat().st_size > 0
        for name in ("model.safetensors", "pytorch_model.bin")
    ):
        raise FileNotFoundError(f"No complete PyTorch/safetensors weights in {directory}")


def inspect_llava_checkpoint(model_id, *, vision_tower_path=None, local_files_only=True):
    """Resolve language and CLIP dependencies without importing/loading model weights."""
    model_path = _snapshot_path(model_id, local_files_only=local_files_only)
    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing LLaVA-Med config: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("model_type") != "llava_mistral":
        raise ValueError(
            "llava_med backend requires the official v1.5 Mistral checkpoint "
            "(model_type='llava_mistral'). Legacy v1.0/delta and HF-converted 'llava' "
            f"checkpoints need their own backend; found {config.get('model_type')!r}."
        )
    architectures = config.get("architectures", [])
    if architectures and "LlavaMistralForCausalLM" not in architectures:
        raise ValueError(f"Unsupported LLaVA-Med architecture: {architectures}")
    _validate_weights(model_path)
    visual_id = vision_tower_path or config.get("mm_vision_tower") or config.get("vision_tower")
    if not visual_id:
        raise ValueError("LLaVA-Med config has no vision tower; set vision_tower_path explicitly")
    visual_path = _snapshot_path(
        visual_id, local_files_only=local_files_only, relative_to=model_path
    )
    for name in ("config.json", "preprocessor_config.json"):
        if not (visual_path / name).is_file():
            raise FileNotFoundError(f"Missing CLIP dependency: {visual_path / name}")
    _validate_weights(visual_path)
    return {"model_path": str(model_path), "vision_tower_path": str(visual_path), "config": config}


def _llava_runtime(source_path=None):
    if source_path is not None:
        source = Path(source_path).expanduser().resolve()
        expected = source / "llava" / "model" / "language_model" / "llava_mistral.py"
        if not expected.is_file():
            raise FileNotFoundError(
                f"source_path must contain the official LLaVA-Med v1.5 source: {expected}"
            )
        imported = sys.modules.get("llava")
        if imported is not None:
            imported_path = Path(getattr(imported, "__file__", "")).resolve()
            if not imported_path.is_relative_to(source):
                raise RuntimeError(
                    f"Another llava package is already imported from {imported_path}. "
                    "Start a fresh process using the configured LLaVA-Med environment/source."
                )
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
    try:
        import torch
        from transformers import AutoTokenizer

        constants = importlib.import_module("llava.constants")
        conversations = importlib.import_module("llava.conversation")
        helpers = importlib.import_module("llava.mm_utils")
        models = importlib.import_module("llava.model.language_model.llava_mistral")
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "Cannot import official LLaVA-Med v1.5. Use the existing compatible LLaVA-Med "
            "Python environment and set source_path (or PYTHONPATH). Do not upgrade its "
            "torch/transformers merely to install this repository's optional Qwen extras."
        ) from exc
    return SimpleNamespace(
        torch=torch,
        AutoTokenizer=AutoTokenizer,
        Config=models.LlavaMistralConfig,
        Model=models.LlavaMistralForCausalLM,
        constants=constants,
        conv_templates=conversations.conv_templates,
        tokenizer_image_token=helpers.tokenizer_image_token,
        process_images=helpers.process_images,
    )


def _eos_ids(model, tokenizer):
    eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if eos is None:
        eos = tokenizer.eos_token_id
    return {int(x) for x in (eos if isinstance(eos, (list, tuple)) else [eos]) if x is not None}


def _generated_rows(output, model, tokenizer):
    """Use generation steps, not prompt length or guessed BOS behavior."""
    steps = len(output.scores)
    if steps == 0 or output.sequences.shape[1] < steps:
        raise RuntimeError("LLaVA-Med returned no generated token scores")
    generated = output.sequences[:, -steps:]
    transitions = model.compute_transition_scores(
        output.sequences,
        output.scores,
        getattr(output, "beam_indices", None),
        normalize_logits=True,
    )
    if transitions.shape[1] != steps:
        raise RuntimeError("LLaVA-Med greedy transition scores do not cover generated tokens")
    eos = _eos_ids(model, tokenizer)
    rows = []
    for ids, scores in zip(
        generated.detach().cpu().tolist(), transitions.float().detach().cpu().tolist(), strict=True
    ):
        end = next((i + 1 for i, token in enumerate(ids) if token in eos), len(ids))
        ids, scores = tuple(ids[:end]), scores[:end]
        if not ids or len(ids) != len(scores) or not np.isfinite(scores).all():
            raise RuntimeError("LLaVA-Med returned malformed token transitions")
        rows.append((ids, float(np.mean(scores)), bool(ids[-1] in eos)))
    return rows


class LlavaMedGeneralist:
    def __init__(
        self,
        model_id: str,
        source_path: str | None = None,
        dtype: str = "float16",
        device_map="auto",
        vision_tower_path: str | None = None,
        local_files_only: bool = True,
        conv_mode: str = "mistral_instruct",
        deterministic_image_padding: bool = False,
    ):
        info = inspect_llava_checkpoint(
            model_id,
            vision_tower_path=vision_tower_path,
            local_files_only=local_files_only,
        )
        runtime = _llava_runtime(source_path)
        self.torch, self.runtime = runtime.torch, runtime
        if dtype not in {"float16", "bfloat16", "float32"}:
            raise ValueError("LLaVA-Med dtype must be float16, bfloat16 or float32")
        if conv_mode not in runtime.conv_templates:
            raise ValueError(f"Unknown LLaVA conversation template: {conv_mode}")
        self.conv_mode = conv_mode
        self.deterministic_image_padding = deterministic_image_padding
        self.checkpoint_info = info
        self.tokenizer = runtime.AutoTokenizer.from_pretrained(
            info["model_path"], use_fast=False, local_files_only=True, trust_remote_code=False
        )
        config = runtime.Config.from_pretrained(info["model_path"], local_files_only=True)
        # Official CLIPVisionTower omits local_files_only in its own helper calls.
        # Supplying a fully checked local directory makes all those calls local.
        config.mm_vision_tower = info["vision_tower_path"]
        mapping = {"": device_map} if device_map in ("cpu", "cuda") else device_map
        self.model = runtime.Model.from_pretrained(
            info["model_path"],
            config=config,
            torch_dtype=getattr(self.torch, dtype),
            device_map=mapping,
            local_files_only=True,
            low_cpu_mem_usage=mapping is not None,
            use_flash_attention_2=False,
        ).eval()
        constants = runtime.constants
        if getattr(config, "mm_use_im_patch_token", True):
            self.tokenizer.add_tokens([constants.DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if getattr(config, "mm_use_im_start_end", False):
            self.tokenizer.add_tokens(
                [constants.DEFAULT_IM_START_TOKEN, constants.DEFAULT_IM_END_TOKEN],
                special_tokens=True,
            )
        self.model.resize_token_embeddings(len(self.tokenizer))
        tower = self.model.get_vision_tower()
        if tower is None:
            raise RuntimeError("LLaVA-Med did not construct its CLIP vision tower")
        if not tower.is_loaded:
            tower.load_model()
        input_device = self.model.get_input_embeddings().weight.device
        if input_device.type == "meta":
            raise RuntimeError(
                "LLaVA-Med input embeddings are offloaded to meta; use a GPU device map"
            )
        tower.to(device=input_device, dtype=getattr(self.torch, dtype))
        self.image_processor = tower.image_processor
        self.processor = SimpleNamespace(tokenizer=self.tokenizer)
        self.model.config.use_cache = True

    def _inputs(self, image, prompt):
        views = image if isinstance(image, (list, tuple)) else [image]
        if not 1 <= len(views) <= 2:
            raise ValueError("Use one original image and at most one predicted evidence view")
        native = [load_rgb(view) for view in views]
        constants = self.runtime.constants
        # One token per image. The single-image baseline retains its exact prompt.
        question = str(prompt).replace(constants.DEFAULT_IMAGE_TOKEN, "").strip()
        image_token = constants.DEFAULT_IMAGE_TOKEN
        if getattr(self.model.config, "mm_use_im_start_end", False):
            image_token = (
                constants.DEFAULT_IM_START_TOKEN + image_token + constants.DEFAULT_IM_END_TOKEN
            )
        conversation = self.runtime.conv_templates[self.conv_mode].copy()
        conversation.append_message(
            conversation.roles[0], (image_token + "\n") * len(native) + question
        )
        conversation.append_message(conversation.roles[1], None)
        device = self.model.get_input_embeddings().weight.device
        ids = (
            self.runtime.tokenizer_image_token(
                conversation.get_prompt(),
                self.tokenizer,
                constants.IMAGE_TOKEN_INDEX,
                return_tensors="pt",
            )
            .unsqueeze(0)
            .to(device)
        )
        pixels = native
        if (getattr(self, "deterministic_image_padding", False)
                and getattr(self.model.config, "image_aspect_ratio", None) == "pad"):
            # Official expand2square can randomly jitter nonsquare images by a
            # pixel. Fix this before source paired interventions; no RNG change.
            background = tuple(int(value * 255) for value in self.image_processor.image_mean)
            pixels = []
            for view in native:
                side = max(view.size)
                canvas = Image.new("RGB", (side, side), background)
                canvas.paste(view, ((side - view.width) // 2, (side - view.height) // 2))
                pixels.append(canvas)
        images = self.runtime.process_images(pixels, self.image_processor, self.model.config)
        tower = self.model.get_vision_tower()
        if isinstance(images, list):
            images = [image.to(device=tower.device, dtype=tower.dtype) for image in images]
        else:
            images = images.to(device=tower.device, dtype=tower.dtype)
        return {
            "inputs": ids,
            "attention_mask": self.torch.ones_like(ids),
            "images": images,
            "image_sizes": [view.size for view in native],
        }

    def new_answer_session(self, image, prompt):
        return LlavaMedAnswerSession(self, image, prompt)

    def _validate_context(self, inputs, max_new_tokens):
        """Do not let the official multimodal helper silently truncate evidence/prefix."""
        config = self.model.config
        tower = self.model.get_vision_tower()
        patches = int(getattr(tower, "num_patches", 1))
        if getattr(tower, "select_feature", "patch") == "cls_patch":
            patches += 1
        image_count = int((inputs["inputs"] == self.runtime.constants.IMAGE_TOKEN_INDEX).sum())
        expanded = int(inputs["inputs"].shape[1]) + image_count * (patches - 1)
        tokenizer_limit = getattr(config, "tokenizer_model_max_length", None)
        position_limit = getattr(config, "max_position_embeddings", None)
        if tokenizer_limit is not None and expanded > int(tokenizer_limit):
            raise ValueError(
                f"LLaVA-Med input expands to {expanded} tokens, exceeding tokenizer_model_max_length="
                f"{tokenizer_limit}. Reduce evidence/controller context; silent prefix truncation is forbidden."
            )
        if position_limit is not None and expanded + max_new_tokens > int(position_limit):
            raise ValueError("LLaVA-Med image, evidence and answer exceed max_position_embeddings")

    def generate(
        self, image, prompt, max_new_tokens=64, logits_processor=None, *, allowed_texts=None
    ):
        return self.generate_with_usage(
            image, prompt, max_new_tokens, logits_processor, allowed_texts=allowed_texts
        )["text"]

    def generate_with_usage(
        self, image, prompt, max_new_tokens=64, logits_processor=None, *, allowed_texts=None
    ):
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        inputs = self._inputs(image, prompt)
        self._validate_context(inputs, max_new_tokens)
        options = {}
        if logits_processor is not None:
            options["logits_processor"] = [logits_processor]
        if allowed_texts is not None:
            from .constrained import finite_choice_constraint

            options["prefix_allowed_tokens_fn"] = finite_choice_constraint(
                self.tokenizer,
                allowed_texts,
                prompt_length=None,
                eos_token_id=next(iter(_eos_ids(self.model, self.tokenizer)), None),
            )
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                return_dict_in_generate=True,
                output_scores=True,
                **options,
            )
        ids, _, _ = _generated_rows(output, self.model, self.tokenizer)[0]
        text = self.tokenizer.decode(ids, skip_special_tokens=True).strip()
        if allowed_texts is not None and text not in allowed_texts:
            raise RuntimeError(
                "Constrained LLaVA-Med action was truncated or did not match its whitelist"
            )
        return {
            "text": text,
            # Text-side token count includes IMAGE_TOKEN_INDEX once. It is not
            # advertised as the expanded number of CLIP patch embeddings.
            "input_tokens": int(inputs["inputs"].shape[1]),
            "output_tokens": len(ids),
            "token_ids": list(ids),
        }


class LlavaMedAnswerSession:
    """Re-prefill image + prompt + exact committed IDs after an evidence update."""

    def __init__(self, generalist, image, prompt):
        self.generalist = generalist
        self.inputs = generalist._inputs(image, prompt)

    def decode(self, tokens):
        return self.generalist.tokenizer.decode(tokens, skip_special_tokens=True)

    def propose(self, prefix, count, length):
        if count != 1:
            raise ValueError("LLaVA-Med capability generation supports greedy blocks (count=1)")
        if length < 1:
            raise ValueError("block length must be positive")
        torch, model = self.generalist.torch, self.generalist.model
        inputs = dict(self.inputs)
        if prefix:
            extra = torch.tensor(
                [prefix], device=inputs["inputs"].device, dtype=inputs["inputs"].dtype
            )
            inputs["inputs"] = torch.cat((inputs["inputs"], extra), dim=1)
            inputs["attention_mask"] = torch.cat(
                (inputs["attention_mask"], torch.ones_like(extra)), dim=1
            )
        self.generalist._validate_context(inputs, length)
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=length,
                do_sample=False,
                use_cache=True,
                return_dict_in_generate=True,
                output_scores=True,
            )
        ids, score, finished = _generated_rows(output, model, self.generalist.tokenizer)[0]
        if len(ids) > length:
            raise RuntimeError("LLaVA-Med exceeded its block token budget")
        return [Block(ids, self.decode(ids), score, finished)]
