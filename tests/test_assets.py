from pathlib import Path

from merit_feddg.assets import verify_assets
from merit_feddg.doctor import diagnostics
from merit_feddg.experts.conch import resolve_checkpoint_source


def test_smoke_assets_are_ready_without_downloads(tmp_path: Path):
    report = verify_assets("smoke", tmp_path)
    assert report["ready"] is True
    assert report["missing"] == []


def test_open_assets_report_missing_and_present_snapshots(tmp_path: Path):
    missing = verify_assets("open-small", tmp_path)
    assert missing["ready"] is False
    assert len(missing["missing"]) == 3

    for item in missing["missing"]:
        path = Path(item["path"])
        path.mkdir(parents=True)
        (path / "payload.bin").write_bytes(b"ready")

    present = verify_assets("open-small", tmp_path)
    assert present["ready"] is True
    assert len(present["present"]) == 3


def test_doctor_reports_local_runtime_without_secrets(tmp_path: Path):
    report = diagnostics(tmp_path / "artifacts")
    assert report["artifact_root"] == str((tmp_path / "artifacts").resolve())
    assert report["disk_gib"]["free"] > 0
    assert "version" in report["python"]
    assert "authenticated" in report["huggingface"]


def test_conch_local_snapshot_resolves_checkpoint_file(tmp_path: Path):
    checkpoint = tmp_path / "pytorch_model.bin"
    checkpoint.write_bytes(b"weights")
    assert resolve_checkpoint_source(str(tmp_path)) == str(checkpoint)
