from pathlib import Path

from merit_feddg.assets import download_profile
from merit_feddg.io import load_yaml
from merit_feddg.runner import compare_records
from merit_feddg.simulation import simulate_records


def test_smoke_profile_needs_no_network(tmp_path: Path):
    report = download_profile("smoke", tmp_path)
    assert report["models"] == []
    assert report["datasets"] == []


def test_full_comparison_and_negative_controls(tmp_path: Path):
    config = load_yaml("configs/smoke.yaml")
    report = compare_records(simulate_records(config), config, tmp_path)
    metrics = report["metrics"]
    assert set(config["evaluation"]["methods"]) == set(metrics)
    assert report["target_labels_used_during_fit"] is False
    assert metrics["merit"]["accuracy"] >= metrics["generalist"]["accuracy"]
    assert metrics["merit"]["accuracy"] >= metrics["wrong_route"]["accuracy"]
    assert (tmp_path / "comparison.md").exists()
