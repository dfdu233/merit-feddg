from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .experts.base import load_rgb, null_image_like


class QwenLayerProbe:
    """Teacher-forced concept likelihood at selected Qwen-VL decoder layers.

    This produces a common concept-space interface. It deliberately does not
    learn a cross-model projection and does not inject specialist logits.
    """

    def __init__(
        self,
        model_id: str,
        layers: list[int],
        dtype: str = "bfloat16",
        device_map: str = "auto",
    ) -> None:
        try:
            import torch
            from qwen_vl_utils import process_vision_info
            from transformers import AutoProcessor
            try:
                from transformers import Qwen2_5_VLForConditionalGeneration as ModelClass
            except ImportError:
                from transformers import AutoModelForVision2Seq as ModelClass
        except ImportError as exc:
            raise RuntimeError("install merit-feddg[research] for Qwen layer probing") from exc

        self.torch = torch
        self.process_vision_info = process_vision_info
        self.layers = [int(layer) for layer in layers]
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        torch_dtype = getattr(torch, dtype)
        self.model = ModelClass.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
        ).eval()
        self.output_head = self.model.get_output_embeddings()
        self.final_norm = self._find_final_norm()

    def _find_final_norm(self):
        candidates = [
            getattr(getattr(self.model, "model", None), "norm", None),
            getattr(getattr(getattr(self.model, "model", None), "language_model", None), "norm", None),
        ]
        return next((item for item in candidates if item is not None), None)

    def _inputs(self, image: Image.Image, prompt: str, answer: str | None = None):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        prompt_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if answer is not None:
            prompt_text += answer
        image_inputs, video_inputs = self.process_vision_info(messages)
        inputs = self.processor(
            text=[prompt_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        return inputs.to(self.model.device)

    def _layer_scores(self, image: Image.Image, prompt: str, concept: str) -> np.ndarray:
        prompt_inputs = self._inputs(image, prompt, None)
        full_inputs = self._inputs(image, prompt, " " + concept)
        prompt_length = int(prompt_inputs["input_ids"].shape[1])
        full_ids = full_inputs["input_ids"]
        answer_ids = full_ids[:, prompt_length:]
        if answer_ids.numel() == 0:
            raise RuntimeError(f"candidate {concept!r} produced no answer tokens")

        with self.torch.inference_mode():
            output = self.model(**full_inputs, output_hidden_states=True, use_cache=False)
        hidden_states = output.hidden_states
        start = prompt_length - 1
        end = full_ids.shape[1] - 1
        scores = []
        for layer in self.layers:
            if layer >= len(hidden_states):
                raise IndexError(f"layer {layer} unavailable; model exposes {len(hidden_states)} states")
            hidden = hidden_states[layer][:, start:end, :]
            if self.final_norm is not None:
                hidden = self.final_norm(hidden)
            logits = self.output_head(hidden).float().log_softmax(dim=-1)
            values = logits.gather(-1, answer_ids.unsqueeze(-1)).squeeze(-1)
            scores.append(float(values.mean().detach().cpu()))
        return np.asarray(scores, dtype=float)

    def probe(
        self,
        image: str | Path | Image.Image,
        prompt: str,
        concepts: list[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        native = load_rgb(image)
        null = null_image_like(native)
        native_scores = np.stack(
            [self._layer_scores(native, prompt, concept) for concept in concepts], axis=1
        )
        null_scores = np.stack(
            [self._layer_scores(null, prompt, concept) for concept in concepts], axis=1
        )
        return null_scores[-1], native_scores - null_scores

    def generate(self, image: str | Path | Image.Image, prompt: str, max_new_tokens: int = 64) -> str:
        native = load_rgb(image)
        inputs = self._inputs(native, prompt, None)
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        trimmed = [output[len(source) :] for source, output in zip(inputs.input_ids, generated)]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
