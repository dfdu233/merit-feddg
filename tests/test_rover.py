import json

import numpy as np
import pytest

from merit_feddg.rover import decode_regions, region_boxes, region_distribution
from merit_feddg.rover_run import read_regions


def test_stable_specific_region_can_change_decision():
    base = np.log([.6, .4])
    region = np.log([.1, .9])
    out, trace = region_distribution(base, region, base, region)
    assert out.argmax() == 1
    assert trace["weight"] == pytest.approx(.5)
    assert np.exp(out).sum() == pytest.approx(1)


def test_control_indistinguishable_region_cannot_intervene():
    base, region = np.log([.8, .2]), np.log([.1, .9])
    out, trace = region_distribution(base, region, region, region)
    np.testing.assert_allclose(out, base)
    assert trace["weight"] == 0


def test_unstable_region_rejected():
    base, region = np.log([.9, .1]), np.log([.1, .9])
    out, trace = region_distribution(base, region, base, base)
    np.testing.assert_allclose(out, base)
    assert trace["weight"] == 0


def test_zero_weight_and_masked_vocabulary():
    base, region = np.array([0., -2., -np.inf]), np.array([-2., 0., -np.inf])
    out, _ = region_distribution(base, region, base, region, weight=0)
    assert out.argmax() == 0 and np.isneginf(out[-1])
    with pytest.raises(ValueError, match="support"):
        region_distribution(base, [0, 0, 0], base, base)


def test_control_geometry_preserves_size_and_expansion():
    boxes = region_boxes([.2, .2, .4, .5], (100, 100))
    r, c, e = [boxes[k] for k in ("region", "control", "expanded")]
    assert (r[2]-r[0], r[3]-r[1]) == (c[2]-c[0], c[3]-c[1])
    assert e[0] < r[0] and e[2] > r[2]
    with pytest.raises(ValueError, match="control"):
        region_boxes([0, 0, 1, 1], (100, 100))


@pytest.mark.parametrize("box", [[0, 0, 0, 1], [0, 0, 2, 1], [0, 0, float('nan'), 1]])
def test_invalid_boxes(box):
    with pytest.raises(ValueError):
        region_boxes(box, (100, 100))


def test_same_committed_prefix_and_eos():
    class Session:
        eos_ids = frozenset({2})
        def __init__(self, first):
            self.first, self.prefixes = first, []
        def next_scores(self, prefix):
            self.prefixes.append(tuple(prefix))
            return np.log(self.first if not prefix else [.01, .01, .98])
        def decode(self, tokens):
            return str(tokens)
    sessions = {"base": Session([.6, .39, .01]), "region": Session([.01, .98, .01]),
                "control": Session([.6, .39, .01]), "expanded": Session([.01, .98, .01])}
    result = decode_regions(sessions, "rover", max_tokens=4)
    assert result["token_ids"] == [1, 2]
    assert all(s.prefixes == [(), (1,)] for s in sessions.values())


def test_region_provenance_and_no_annotation_boxes(tmp_path):
    path = tmp_path / "regions.json"
    item = {"box": [.1, .1, .3, .3], "producer": "test-predictor", "checkpoint_sha256": "a"*64,
            "image_sha256": "pixels", "kind": "predicted_anatomy"}
    rows = [{"id": "case", "image_sha256": "pixels"}]
    path.write_text(json.dumps({"case": item}))
    assert read_regions(path, rows)["case"] == item
    item["kind"] = "ground_truth"
    path.write_text(json.dumps({"case": item}))
    with pytest.raises(ValueError, match="predicted"):
        read_regions(path, rows)
    item["kind"], item["image_sha256"] = "predicted_lesion", "wrong"
    path.write_text(json.dumps({"case": item}))
    with pytest.raises(ValueError, match="fingerprint"):
        read_regions(path, rows)


def test_runner_cache_resume_does_not_load_model(tmp_path, monkeypatch):
    from PIL import Image

    from merit_feddg import rover_run
    from merit_feddg.open_data import pixel_digest

    image_path = tmp_path / "image.png"
    Image.new("RGB", (100, 100)).save(image_path)
    digest = pixel_digest(image_path)
    row = {"id": "source-1", "image": str(image_path), "question": "Describe the image.",
           "modality": "cxr", "capability": "generation", "task": "vqa", "domain": "source-a",
           "domain_kind": "proxy", "role": "source", "group_id": "patient-1", "image_sha256": digest}
    manifest, regions, config = [tmp_path / name for name in ("source.jsonl", "regions.json", "c.yaml")]
    manifest.write_text(json.dumps(row)+"\n")
    regions.write_text(json.dumps({row["id"]: {"box": [.1, .1, .4, .5], "producer": "test",
        "checkpoint_sha256": "a"*64, "image_sha256": digest, "kind": "predicted_anatomy"}}))
    config.write_text("generalist:\n  backend: llava_med\n  id: test\n")

    class Session:
        def __init__(self):
            self.inputs = {}
        eos_ids = frozenset({1})
        def next_scores(self, prefix):
            return np.log([.1, .9])
        def decode(self, prefix):
            return "test"
    class Model:
        def new_answer_session(self, image, prompt):
            return Session()
        def _validate_context(self, inputs, count):
            pass
        def generate_with_usage(self, image, prompt, max_new_tokens):
            return {"token_ids": [1]}
    monkeypatch.setattr(rover_run, "generalist_provenance", lambda *a: {"test": True})
    monkeypatch.setattr(rover_run, "load_generalist", lambda *a: Model())
    output = tmp_path / "run"
    import sys
    monkeypatch.setattr(sys, "argv", ["rover", "--source-manifest", str(manifest),
        "--regions", str(regions), "--config", str(config), "--output", str(output)])
    rover_run.main()
    report = json.loads((output / "result.json").read_text())
    assert report["cases"][0]["production_parity"]
    assert report["target_generations"] == 0
    def forbidden(*args):
        raise AssertionError("cache resume must not load model")
    monkeypatch.setattr(rover_run, "load_generalist", forbidden)
    rover_run.main()
