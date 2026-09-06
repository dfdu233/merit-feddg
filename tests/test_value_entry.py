import random
from types import SimpleNamespace

import pytest
from PIL import Image
from test_capability_runtime import make_multiimage_backend

from merit_feddg.llava_run import experiment_config, parser


def test_value_profile_is_additive_and_uses_existing_models():
    old = experiment_config(parser().parse_args([]))
    new = experiment_config(parser().parse_args(["--study", "value"]))
    assert "capability_value" not in old
    assert new["experts"] == old["experts"]
    assert new["generalist"]["checkpoint_path"] == old["generalist"]["checkpoint_path"]
    assert new["generalist"]["deterministic_image_padding"] is True
    assert new["capability_value"]["generation"]["max_expert_calls"] == 2
    assert new["capability_value"]["collection"]["verify_block_none"] is True
    assert new["capability_value"]["quality"]["name"] == "token_f1"


def test_deterministic_multiview_padding_keeps_original_geometry_and_rng():
    backend, logs = make_multiimage_backend()
    backend.deterministic_image_padding = True
    backend.model.config.image_aspect_ratio = "pad"
    backend.image_processor = SimpleNamespace(image_mean=[0.5, 0.5, 0.5])
    original = Image.new("RGB", (7, 4), "red")
    overlay = original.copy()
    overlay.putpixel((2, 1), (0, 255, 0))
    before = random.getstate()
    first = backend._inputs([original, overlay], "question")
    captured = [image.copy() for image in logs["images"]]
    backend._inputs([original, overlay], "question")
    assert [v.tobytes() for v in captured] == [v.tobytes() for v in logs["images"]]
    assert random.getstate() == before
    assert first["image_sizes"] == [(7, 4), (7, 4)]
    assert captured[0].size == captured[1].size == (7, 7)
    assert captured[0].getpixel((0, 0)) == (127, 127, 127)
    assert captured[0].getpixel((0, 1)) == (255, 0, 0)
    assert captured[1].getpixel((2, 2)) == (0, 255, 0)
    assert original.size == (7, 4) and original.getpixel((2, 1)) == (255, 0, 0)


def test_legacy_padding_is_not_silently_changed():
    backend, logs = make_multiimage_backend()
    backend.model.config.image_aspect_ratio = "pad"
    backend._inputs(Image.new("RGB", (7, 4)), "question")
    assert logs["images"][0].size == (7, 4)


@pytest.mark.parametrize("count", [-1, True, 1.5])
def test_bad_pair_budget_fails_before_tool_execution(count):
    from merit_feddg.capability_value_study import collect_source_case

    runtime = SimpleNamespace(row={"role": "source"})
    with pytest.raises(ValueError, match="nonnegative integer"):
        collect_source_case(runtime, [], None, max_pair_first_tools=count)


def test_block_none_mismatch_stops_source_calibration():
    from merit_feddg.capability_value_study import collect_source_case

    runtime = SimpleNamespace(row={"role": "source", "id": "test"},
                              complete=lambda state: {"token_ids": [1]},
                              run=lambda mode: {"token_ids": [2]})
    with pytest.raises(RuntimeError, match="block-NONE differs"):
        collect_source_case(runtime, [], None, verify_block_none=True)
