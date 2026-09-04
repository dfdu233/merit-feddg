import numpy as np
from PIL import Image

from merit_feddg.io import load_yaml
from merit_feddg.methods import selected_route
from merit_feddg.prepare import _write_text_if_changed, image_bytes_are_valid, proxy_domain
from merit_feddg.runner import make_oracle_records, route_metrics
from merit_feddg.simulation import simulate_records
from merit_feddg.types import EvidenceRecord


def test_proxy_domain_is_stable_and_image_level():
    key = "01234567" + "0" * 56
    assert proxy_domain("example", key) == proxy_domain("example", key)
    assert proxy_domain("example", key).endswith(("source_a", "source_b", "target"))


def test_image_validation_rejects_arbitrary_bytes(tmp_path):
    path = tmp_path / "sample.png"
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(path)
    assert image_bytes_are_valid(path.read_bytes())
    assert not image_bytes_are_valid(b"not-an-image")


def test_unchanged_prepared_manifest_keeps_cache_timestamp(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    assert _write_text_if_changed(manifest, '{"id":"a"}\n') is True
    timestamp = manifest.stat().st_mtime_ns
    assert _write_text_if_changed(manifest, '{"id":"a"}\n') is False
    assert manifest.stat().st_mtime_ns == timestamp


def test_oracle_cache_routes_to_ground_truth_modality():
    records = simulate_records(load_yaml("configs/smoke.yaml"))[:3]
    converted = make_oracle_records(records)
    for record in converted:
        assert max(record.router_probs, key=record.router_probs.get) == record.modality


def test_multi_expert_route_is_selected_in_modality_space():
    record = EvidenceRecord(
        sample_id="path-1",
        domain="center-a",
        modality="pathology",
        candidates=["a", "b"],
        label=0,
        general_null_logits=np.zeros(2),
        general_visual_layers=np.zeros((2, 2)),
        expert_scores={
            "conch_a": np.asarray([1.0, 0.0]),
            "conch_b": np.asarray([0.5, 0.0]),
            "cxr_model": np.asarray([0.0, 1.0]),
        },
        broad_specialist_scores=np.zeros(2),
        # Pathology mass is split across two plugins.  A direct expert-ID
        # argmax would incorrectly select the single CXR plugin.
        router_probs={"conch_a": 0.30, "conch_b": 0.30, "cxr_model": 0.40},
        metadata={
            "modality_router_probs": {"pathology": 0.60, "cxr": 0.40},
            "expert_modalities": {
                "conch_a": ["pathology"],
                "conch_b": ["pathology"],
                "cxr_model": ["cxr"],
            },
            "expert_capabilities": {
                "conch_a": ["classification"],
                "conch_b": ["classification"],
                "cxr_model": ["classification"],
            },
        },
    )

    expert, confidence = selected_route(record)
    assert expert == "conch_a"
    assert confidence > 0.0
    assert route_metrics([record])["accuracy"] == 1.0

    make_oracle_records([record])
    assert np.allclose(
        [record.metadata["modality_router_probs"][name] for name in ("cxr", "pathology")],
        [0.02, 0.98],
    )
    assert np.allclose(
        [record.router_probs[name] for name in ("conch_a", "conch_b", "cxr_model")],
        [0.49, 0.49, 0.02],
    )
