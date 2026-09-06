"""Protocol and token accounting tests, not pretrained CheXagent accuracy tests."""

from types import SimpleNamespace

import pytest
from PIL import Image

import merit_feddg.experts.native_chexagent as module
from merit_feddg.capabilities import CapabilityRequest


def request(tmp_path, **changes):
    image = tmp_path / "cxr.png"
    Image.new("RGB", (8, 8), (100, 100, 100)).save(image)
    values = {"sample_id": "case", "image": str(image), "question": "What is visible?",
              "modality": "cxr", "task": "open_vqa", "domain": "target", "group_id": "patient",
              "capability": "generation", "scope": "cxr_description", "query": "Describe lungs."}
    return CapabilityRequest(**{**values, **changes})


def test_chexagent_generates_only_continuation_with_real_token_counts(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    prompts, loads = [], []

    class Model:
        def parameters(self):
            yield torch.tensor([1.])

        def to(self, **kwargs):
            return self

        def generate(self, input_ids, **kwargs):
            assert kwargs["do_sample"] is False
            return torch.cat([input_ids, torch.tensor([[8, 9, 2]])], dim=1)

    class Expert:
        def __init__(self, *args, **kwargs):
            loads.append((args, kwargs))
            self.torch, self.model = torch, Model()
            self.tokenizer = SimpleNamespace(decode=self.decode)

        def _prompt_ids(self, image, prompt):
            prompts.append(prompt)
            return torch.tensor([[1, 4, 5]])

        def decode(self, tokens, **kwargs):
            assert tokens.tolist() == [8, 9, 2]
            assert kwargs["skip_special_tokens"] is True
            return " Small pleural effusion. "

    monkeypatch.setattr(module, "CheXagentConceptExpert", Expert)
    expert = module.CheXagentCapabilityExpert(str(tmp_path), "cxr")
    assert not loads
    item = expert.infer(request(tmp_path, generated_prefix="UNVERIFIED PRIOR CLAIM")).items[0]
    assert item.payload["generated_text"] == "Small pleural effusion."
    assert item.payload["usage"] == {"input_tokens": 3, "output_tokens": 3}
    assert item.payload["observation_status"] == "unverified_specialist_output"
    assert "UNVERIFIED PRIOR CLAIM" not in prompts[0]
    assert item.provenance["candidate_scores_used"] is False
    assert item.confidence is None
    expert.infer(request(tmp_path))
    assert len(loads) == 1
    assert loads[0][1]["dtype"] == "bfloat16"


@pytest.mark.parametrize("changes,reason", [
    ({"modality": "pathology"}, "wrong_modality"),
    ({"capability": "classification"}, "wrong_capability"),
    ({"scope": "diagnosis"}, "wrong_scope"),
    ({"region": (0., 0., .5, .5)}, "unsupported_region"),
])
def test_chexagent_incompatible_request_never_loads(tmp_path, changes, reason):
    expert = module.CheXagentCapabilityExpert("missing/local/path", "cxr")
    result = expert.infer(request(tmp_path, **changes))
    assert result.reason == reason and expert.expert is None


def test_chexagent_never_downloads_missing_checkpoint(tmp_path):
    expert = module.CheXagentCapabilityExpert("StanfordAIMI/CheXagent-2-3b", "cxr")
    with pytest.raises(FileNotFoundError, match="never downloads"):
        expert.infer(request(tmp_path))
