from __future__ import annotations

import tempfile
from contextlib import contextmanager
from pathlib import Path
from types import MethodType

import numpy as np
from PIL import Image

from .base import ConceptExpert, load_rgb, null_image_like


@contextmanager
def _chexagent_transformers_compatibility():
    """Scope CheXagent's exact-version import guard to its dynamic import."""

    import transformers

    original = transformers.__version__
    transformers.__version__ = "4.40.0"
    try:
        yield
    finally:
        transformers.__version__ = original


def _patch_visual_hidden_state_fallback(model) -> None:
    """Keep CheXagent working when newer SigLIP omits ``hidden_states``."""

    visual = getattr(getattr(model, "model", None), "visual", None)
    if visual is None or getattr(visual, "_merit_hidden_state_fallback", False):
        return

    def forward(instance, pixels):
        output = instance.model(pixels, output_hidden_states=True)
        hidden_states = getattr(output, "hidden_states", None)
        features = hidden_states[-1] if hidden_states else output.last_hidden_state
        return instance.forward_resampler(features)

    visual.forward = MethodType(forward, visual)
    visual._merit_hidden_state_fallback = True


class CheXagentConceptExpert(ConceptExpert):
    """Image-minus-null phrase likelihood for CheXagent-2-3B."""

    def __init__(self, model_id: str, device_map: str = "auto") -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("install merit-feddg[research] to load CheXagent") from exc
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        with _chexagent_transformers_compatibility():
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                device_map=device_map,
            ).eval()
        _patch_visual_hidden_state_fallback(self.model)

    def _prompt_ids(self, image_path: str, prompt: str):
        query = self.tokenizer.from_list_format([{"image": image_path}, {"text": prompt}])
        conversation = [
            {"from": "system", "value": "You are a careful medical imaging assistant."},
            {"from": "human", "value": query},
        ]
        return self.tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_tensors="pt",
        )

    def _score(self, image_path: str, prompt: str, concept: str) -> float:
        prompt_ids = self._prompt_ids(image_path, prompt)
        answer_ids = self.tokenizer.encode(
            " " + concept, add_special_tokens=False, return_tensors="pt"
        )
        device = next(self.model.parameters()).device
        prompt_ids = prompt_ids.to(device)
        answer_ids = answer_ids.to(device)
        full = self.torch.cat([prompt_ids, answer_ids], dim=1)
        with self.torch.inference_mode():
            logits = self.model(input_ids=full).logits
            start = prompt_ids.shape[1] - 1
            selected = logits[:, start : start + answer_ids.shape[1], :]
            log_probs = selected.log_softmax(dim=-1)
            values = log_probs.gather(-1, answer_ids.unsqueeze(-1)).squeeze(-1)
        return float(values.sum().detach().cpu())

    def image_null_scores(
        self,
        image: str | Path | Image.Image,
        prompt: str,
        concepts: list[str],
    ) -> np.ndarray:
        native = load_rgb(image)
        null = null_image_like(native)
        with tempfile.TemporaryDirectory(prefix="merit-chex-") as directory:
            native_path = str(Path(directory) / "native.png")
            null_path = str(Path(directory) / "null.png")
            native.save(native_path)
            null.save(null_path)
            return np.asarray(
                [
                    self._score(native_path, prompt, item) - self._score(null_path, prompt, item)
                    for item in concepts
                ],
                dtype=float,
            )
