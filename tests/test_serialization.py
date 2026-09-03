from merit_feddg.io import load_records, load_yaml, save_records
from merit_feddg.simulation import simulate_records


def test_evidence_round_trip(tmp_path):
    config = load_yaml("configs/smoke.yaml")
    original = simulate_records(config)[:3]
    path = tmp_path / "evidence.jsonl"
    save_records(path, original)
    restored = load_records(path)
    assert [item.sample_id for item in restored] == [item.sample_id for item in original]
    assert restored[0].general_visual_layers.shape == original[0].general_visual_layers.shape
