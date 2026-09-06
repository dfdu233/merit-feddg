"""Optional means missing-checkpoint omission, not swallowing runtime defects."""

import pytest

from merit_feddg.capability_study import _filter_optional_experts
from merit_feddg.open_study import model_provenance


def test_missing_optional_checkpoint_is_recorded_without_touching_required_tool(tmp_path):
    specs = {
        "required": {"id": "needed/model", "checkpoint_path": str(tmp_path / "missing-required")},
        "chexagent": {
            "id": "StanfordAIMI/CheXagent-2-3b", "optional": True,
            "checkpoint_path": str(tmp_path / "missing-optional"),
            "capabilities": ["generation"], "scope": "cxr_description",
        },
    }
    active, excluded = _filter_optional_experts(specs, tmp_path)
    assert list(active) == ["required"]
    assert excluded["chexagent"]["reason"] == "optional_checkpoint_missing"
    assert excluded["chexagent"]["download_attempted"] is False
    assert list(specs) == ["required", "chexagent"]  # caller config remains unchanged
    with pytest.raises(FileNotFoundError, match="download the model first"):
        model_provenance(active["required"], tmp_path)


def test_present_optional_directory_is_not_filtered_even_if_empty(tmp_path):
    path = tmp_path / "chexagent"
    path.mkdir()
    spec = {"id": "local/chexagent", "checkpoint_path": str(path), "optional": True}
    active, excluded = _filter_optional_experts({"chexagent": spec}, tmp_path)
    assert active == {"chexagent": spec} and not excluded
    with pytest.raises(ValueError, match="empty local checkpoint"):
        model_provenance(active["chexagent"], tmp_path)


def test_present_optional_checkpoint_file_is_enabled(tmp_path):
    path = tmp_path / "local.pth"
    path.write_bytes(b"test-fixture")
    spec = {"id": "xrv/pspnet", "checkpoint_path": str(path), "optional": True}
    active, excluded = _filter_optional_experts({"anatomy": spec}, tmp_path)
    assert list(active) == ["anatomy"] and not excluded


def test_optional_hf_model_uses_artifact_directory_without_remote_check(tmp_path):
    spec = {"id": "owner/model", "optional": True}
    active, excluded = _filter_optional_experts({"tool": spec}, tmp_path)
    assert not active and list(excluded) == ["tool"]
    snapshot = tmp_path / "models" / "owner--model"
    snapshot.mkdir(parents=True)
    active, excluded = _filter_optional_experts({"tool": spec}, tmp_path)
    assert list(active) == ["tool"] and not excluded


@pytest.mark.parametrize("value", ["false", "true", 0, 1, None])
def test_optional_flag_must_be_boolean(tmp_path, value):
    with pytest.raises(TypeError, match="explicit boolean"):
        _filter_optional_experts({"tool": {"id": "model", "optional": value}}, tmp_path)
