from types import SimpleNamespace

import pytest

from merit_feddg.experts.chexagent import (
    _chexagent_transformers_compatibility,
    _patch_visual_hidden_state_fallback,
)


def test_transformers_version_override_is_scoped():
    transformers = pytest.importorskip("transformers")

    original = transformers.__version__
    with _chexagent_transformers_compatibility():
        assert transformers.__version__ == "4.40.0"
    assert transformers.__version__ == original


def test_visual_forward_falls_back_to_last_hidden_state():
    class Visual:
        def __init__(self):
            self.model = lambda pixels, output_hidden_states: SimpleNamespace(
                hidden_states=None,
                last_hidden_state=pixels + 1,
            )

        def forward_resampler(self, features):
            return features * 2

    visual = Visual()
    model = SimpleNamespace(model=SimpleNamespace(visual=visual))
    _patch_visual_hidden_state_fallback(model)

    assert visual.forward(3) == 8
    assert visual._merit_hidden_state_fallback


def test_visual_forward_casts_pixels_to_patch_weight_dtype():
    torch = pytest.importorskip("torch")

    class Encoder:
        def __init__(self):
            self.embeddings = SimpleNamespace(
                patch_embedding=SimpleNamespace(
                    weight=torch.ones(1, dtype=torch.bfloat16)
                )
            )

        def __call__(self, pixels, output_hidden_states):
            assert output_hidden_states is True
            assert pixels.dtype == torch.bfloat16
            return SimpleNamespace(hidden_states=(pixels,), last_hidden_state=None)

    visual = SimpleNamespace(model=Encoder(), forward_resampler=lambda features: features)
    model = SimpleNamespace(model=SimpleNamespace(visual=visual))
    _patch_visual_hidden_state_fallback(model)
    result = visual.forward(torch.ones(1, dtype=torch.float32))
    assert result.dtype == torch.bfloat16
