"""Metadata parity for answer-blind TRAIN preparation (synthetic fixture)."""
import json
from types import SimpleNamespace

from test_request_parse_study import ROW, load_script, spec


def test_prepare_uses_prior_input_modality_but_not_answer(tmp_path, monkeypatch):
    runner = load_script("run_request_parse_study")
    manifest, incumbent, config = (tmp_path / x for x in ("manifest.jsonl", "inc.json", "config.yaml"))
    manifest.write_text(json.dumps(dict(ROW, modality="radiology", image_sha256="image-hash")) + "\n")
    incumbent.write_text(json.dumps({ROW["id"]: {"input_modality": "cxr", "text": "private answer"}}))
    config.write_text(json.dumps({"generalist": {"backend": "llava_med"},
                                  "experts": {k: spec() for k in runner.EXPERTS}}))
    monkeypatch.setattr(runner, "code_hashes", lambda: {"fixture": "no upstream dependencies"})
    args = SimpleNamespace(manifest=manifest, incumbent_json=incumbent, config=config,
                           output=tmp_path / "run", limit=12, max_new_tokens=512)
    runner.prepare(args)
    frozen = json.loads((args.output / "frozen.json").read_text())
    assert frozen["questions"][0]["modality"] == "cxr"
    assert "private answer" not in json.dumps(frozen)
    assert frozen["input_metadata_source"]["kind"] == "incumbent_input_modality"
    assert frozen["image_hashes"][ROW["id"]] == "image-hash"
    runner.run(SimpleNamespace(output=args.output, execute=False))
