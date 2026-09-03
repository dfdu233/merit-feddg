from pathlib import Path
from types import ModuleType

from merit_feddg.assets import (
    _asset_state,
    _requirements_satisfied,
    download_profile,
    verify_assets,
)
from merit_feddg.doctor import diagnostics
from merit_feddg.experts.conch import resolve_checkpoint_source


def test_smoke_assets_are_ready_without_downloads(tmp_path: Path):
    report = verify_assets("smoke", tmp_path)
    assert report["ready"] is True
    assert report["missing"] == []


def _fake_huggingface_hub(monkeypatch, calls: list[str]) -> None:
    module = ModuleType("huggingface_hub")

    def snapshot_download(*, repo_id, local_dir, **_kwargs):
        calls.append(repo_id)
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / "payload.bin").write_bytes(repo_id.encode())
        return str(local_dir)

    module.snapshot_download = snapshot_download
    monkeypatch.setitem(__import__("sys").modules, "huggingface_hub", module)


def test_open_assets_report_missing_and_present_snapshots(tmp_path: Path, monkeypatch):
    missing = verify_assets("open-small", tmp_path)
    assert missing["ready"] is False
    assert len(missing["missing"]) == 3

    calls = []
    _fake_huggingface_hub(monkeypatch, calls)
    download_profile("open-small", tmp_path)

    present = verify_assets("open-small", tmp_path)
    assert present["ready"] is True
    assert len(present["present"]) == 3
    assert len(calls) == 3


def test_completed_assets_are_reused_and_damaged_assets_resume(tmp_path: Path, monkeypatch):
    calls = []
    _fake_huggingface_hub(monkeypatch, calls)

    first = download_profile("open-small", tmp_path)
    assert len(first["downloaded"]) == 3
    assert len(calls) == 3

    second = download_profile("open-small", tmp_path)
    assert len(second["reused"]) == 3
    assert len(calls) == 3

    damaged = tmp_path / "models" / "microsoft--rad-dino" / "payload.bin"
    damaged.write_bytes(b"truncated")
    third = download_profile("open-small", tmp_path)
    assert len(third["resumed"]) == 1
    assert len(third["reused"]) == 2
    assert calls[-1] == "microsoft/rad-dino"


def test_force_download_refreshes_completed_assets(tmp_path: Path, monkeypatch):
    calls = []
    _fake_huggingface_hub(monkeypatch, calls)
    download_profile("open-small", tmp_path)

    refreshed = download_profile("open-small", tmp_path, force_download=True)
    assert len(refreshed["downloaded"]) == 3
    assert len(calls) == 6


def test_dataset_requirements_reject_partial_multifile_snapshot(tmp_path: Path):
    (tmp_path / "train.json").write_text("[]", encoding="utf-8")
    entry = {"id": "example/data", "required_files": ["train.json", "imgs.zip"]}
    assert _requirements_satisfied(tmp_path, entry, _asset_state(tmp_path, "dataset")) is False

    (tmp_path / "imgs.zip").write_bytes(b"zip-payload")
    assert _requirements_satisfied(tmp_path, entry, _asset_state(tmp_path, "dataset")) is True


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
