import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import merit_feddg.capability_setup as module

ASSET = {"id": "tiny-xrv", "filename": "test.pt", "bytes": 10}


class Response(io.BytesIO):
    def __init__(self, data, status=200, headers=None):
        super().__init__(data)
        self.status = status
        self.headers = headers or {"Content-Length": str(len(data))}


def location(tmp_path):
    return tmp_path / "models" / "xrv" / ASSET["filename"]


def partial_file(tmp_path, data):
    path = location(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".incomplete")
    partial.write_bytes(data)
    return partial


def test_full_download_and_hash_verified_reuse_never_contacts_network(tmp_path, monkeypatch):
    requests = []

    def get(request, timeout):
        requests.append(request)
        assert request.full_url == module._RELEASE + "test.pt"
        assert request.get_header("Range") is None
        return Response(b"0123456789")

    monkeypatch.setattr(module.urllib.request, "urlopen", get)
    first = module.ensure_xrv_weight(tmp_path, ASSET)
    assert first["status"] == "downloaded"
    assert first["upstream_sha256"] is None
    assert first["third_party_transport"] is False
    assert location(tmp_path).read_bytes() == b"0123456789"
    assert not list(location(tmp_path).parent.glob("*.incomplete"))
    assert not list(location(tmp_path).parent.glob("*.download.lock"))
    second = module.ensure_xrv_weight(tmp_path, ASSET)
    third = module.ensure_xrv_weight(tmp_path, ASSET, check_only=True)
    assert second["status"] == third["status"] == "reused_verified_local"
    assert first["sha256"] == second["sha256"]
    assert len(requests) == 1


def test_resume_uses_matching_content_range(tmp_path, monkeypatch):
    partial = partial_file(tmp_path, b"0123")

    def get(request, timeout):
        assert request.get_header("Range") == "bytes=4-"
        return Response(b"456789", 206, {"Content-Range": "bytes 4-9/10", "Content-Length": "6"})

    monkeypatch.setattr(module.urllib.request, "urlopen", get)
    report = module.ensure_xrv_weight(tmp_path, ASSET)
    assert report["status"] == "resumed"
    assert location(tmp_path).read_bytes() == b"0123456789"
    assert not partial.exists()


def test_interrupted_body_retries_from_preserved_bytes(tmp_path, monkeypatch):
    calls = []

    def get(request, timeout):
        calls.append(request.get_header("Range"))
        if len(calls) == 1:
            return Response(b"012", 200, {"Content-Length": "10"})
        return Response(b"3456789", 206, {"Content-Range": "bytes 3-9/10", "Content-Length": "7"})

    monkeypatch.setattr(module.urllib.request, "urlopen", get)
    module.ensure_xrv_weight(tmp_path, ASSET, attempts=2)
    assert calls == [None, "bytes=3-"]
    assert location(tmp_path).read_bytes() == b"0123456789"


@pytest.mark.parametrize(
    "status,headers",
    [
        (200, {"Content-Length": "10"}),
        (206, {"Content-Range": "bytes 0-9/10", "Content-Length": "10"}),
        (206, {"Content-Range": "bytes 4-9/11", "Content-Length": "6"}),
        (206, {"Content-Range": "bytes 4-9/10", "Content-Length": "5"}),
    ],
)
def test_bad_range_or_length_never_appends_to_existing_partial(
    tmp_path, monkeypatch, status, headers
):
    partial = partial_file(tmp_path, b"0123")
    monkeypatch.setattr(
        module.urllib.request, "urlopen", lambda *a, **kw: Response(b"bad-payload", status, headers)
    )
    with pytest.raises(RuntimeError, match="keep .incomplete"):
        module.ensure_xrv_weight(tmp_path, ASSET, attempts=1)
    assert partial.read_bytes() == b"0123"
    assert not location(tmp_path).exists()
    assert not list(partial.parent.glob("*.download.lock"))


def test_html_proxy_response_is_not_saved_as_weights(tmp_path, monkeypatch):
    monkeypatch.setattr(
        module.urllib.request,
        "urlopen",
        lambda *a, **kw: Response(b"badcontent", headers={"Content-Type": "text/html"}),
    )
    with pytest.raises(RuntimeError, match="page/error"):
        module.ensure_xrv_weight(tmp_path, ASSET, attempts=1)
    assert not location(tmp_path).exists()


def test_complete_partial_is_finalized_offline(tmp_path, monkeypatch):
    partial_file(tmp_path, b"0123456789")
    monkeypatch.setattr(
        module.urllib.request, "urlopen", lambda *a, **kw: pytest.fail("unneeded transfer")
    )
    report = module.ensure_xrv_weight(tmp_path, ASSET)
    assert report["status"] == "resumed"
    assert location(tmp_path).read_bytes() == b"0123456789"


def test_existing_hash_mismatch_preserves_file_and_does_not_redownload(tmp_path, monkeypatch):
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda *a, **kw: Response(b"0123456789"))
    module.ensure_xrv_weight(tmp_path, ASSET)
    path = location(tmp_path)
    path.write_bytes(b"changedxxx")
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda *a, **kw: pytest.fail("overwrite"))
    with pytest.raises(RuntimeError, match="SHA256 changed"):
        module.ensure_xrv_weight(tmp_path, ASSET)
    assert path.read_bytes() == b"changedxxx"


def test_untracked_existing_weight_requires_explicit_validation(tmp_path):
    path = location(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"0123456789")
    with pytest.raises(RuntimeError, match="no integrity manifest"):
        module.ensure_xrv_weight(tmp_path, ASSET)
    assert path.read_bytes() == b"0123456789"


def test_lock_prevents_shared_partial_writers_and_is_not_removed(tmp_path):
    partial = partial_file(tmp_path, b"0123")
    lock = partial.with_name("test.pt.download.lock")
    lock.write_text('{"pid":1234}')
    with pytest.raises(RuntimeError, match="Another download owns"):
        module.ensure_xrv_weight(tmp_path, ASSET)
    assert lock.exists() and partial.read_bytes() == b"0123"


def test_check_only_has_no_download_or_directory_writes(tmp_path, monkeypatch):
    target = tmp_path / "not-created"
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda *a, **kw: pytest.fail("network"))
    monkeypatch.setattr(module, "dependency_report", list)
    report = module.prepare_capabilities(target, mirror="cn", check_only=True)
    assert not report["ready_weights"]
    assert all(item["status"] == "missing" for item in report["weights"])
    assert not target.exists()


def test_explicit_proxy_is_recorded_but_cn_does_not_imply_it(tmp_path, monkeypatch):
    seen = []

    def get(request, timeout):
        seen.append(request.full_url)
        return Response(b"0123456789")

    monkeypatch.setattr(module.urllib.request, "urlopen", get)
    report = module.ensure_xrv_weight(tmp_path, ASSET, github_proxy="https://ghfast.top/")
    assert seen == ["https://ghfast.top/" + module._RELEASE + "test.pt"]
    assert report["third_party_transport"] is True and report["upstream_sha256"] is None
    assert module._transport_url(module._RELEASE + "test.pt") == module._RELEASE + "test.pt"


@pytest.mark.parametrize(
    "proxy", ["http://proxy.example", "https://secret@proxy.example", "https://x/?token=secret"]
)
def test_proxy_credentials_and_insecure_urls_are_rejected(proxy):
    with pytest.raises(ValueError):
        module._transport_url(module._RELEASE + "test.pt", proxy)


def test_only_missing_allowed_dependencies_are_installed_without_dependency_upgrade(monkeypatch):
    calls = []
    monkeypatch.setattr(module.subprocess, "run", lambda args, check: calls.append(args))
    report = [
        {"module": "torch", "role": "preserved_core", "present": True, "install_if_missing": None},
        {
            "module": "transformers",
            "role": "preserved_core",
            "present": True,
            "install_if_missing": None,
        },
        {
            "module": "torchxrayvision",
            "role": "xrv",
            "present": False,
            "install_if_missing": "torchxrayvision==1.5.4",
        },
        {
            "module": "open_clip",
            "role": "biomedclip",
            "present": True,
            "install_if_missing": "open_clip_torch==3.2.0",
        },
        {"module": "conch", "role": "conch", "present": False, "install_if_missing": None},
    ]
    assert module.install_missing_dependencies(report, mirror="cn") == ["torchxrayvision==1.5.4"]
    assert len(calls) == 1 and "--no-deps" in calls[0]
    assert "--upgrade" not in calls[0]
    assert calls[0][-1] == "torchxrayvision==1.5.4"
    assert "https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple" in calls[0]


def test_core_stack_is_never_installed_automatically(monkeypatch):
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **kw: pytest.fail("changed torch"))
    with pytest.raises(RuntimeError, match="existing torch"):
        module.install_missing_dependencies(
            [
                {
                    "module": "torch",
                    "role": "preserved_core",
                    "present": False,
                    "install_if_missing": None,
                }
            ]
        )


def test_old_openclip_is_preserved_and_has_explicit_upgrade_hint(monkeypatch):
    monkeypatch.setattr(module.importlib.util, "find_spec", lambda name: SimpleNamespace())
    monkeypatch.setattr(
        module.importlib.metadata,
        "version",
        lambda name: "2.24.0" if name == "open-clip-torch" else "1.0",
    )
    report = module.dependency_report()
    item = next(item for item in report if item["module"] == "open_clip")
    assert item["present"] and "preserved" in item["hint"]
    assert "--no-deps open_clip_torch==3.2.0" in item["hint"]


def test_cli_reports_missing_without_mutation(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(module, "dependency_report", list)
    result = module.main(["--artifacts", str(tmp_path / "a"), "--check-only"])
    report = json.loads(capsys.readouterr().out)
    assert result == 1 and report["check_only"] is True
    assert not (tmp_path / "a").exists()
    assert module.main(["--check-only", "--install-deps"]) == 1


def test_registry_uses_published_weights_and_sizes():
    assert [asset["bytes"] for asset in module.XRV_ASSETS] == [28382008, 272988989]
    assert all(Path(asset["filename"]).suffix in {".pt", ".pth"} for asset in module.XRV_ASSETS)
