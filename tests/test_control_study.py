import copy
import importlib
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from merit_feddg.capabilities import EvidenceItem
from merit_feddg.capability_runtime import NativeSession, NativeState, ValueGenerationConfig
from merit_feddg.control_study import ABLATIONS, ARMS, control_case, prompt_config
from merit_feddg.matched_evaluation import generation_prompt
from merit_feddg.spatial_evidence import SpatialEvidenceBridge, encode_soft_mask, spatial_packet


class FakeSession:
    def __init__(self, probe, prompt, packet=None):
        self.probe, self.prompt, self.tensor_packet = probe, prompt, packet

    def decode(self, tokens):
        return " ".join(map(str, tokens))

    def next_scores(self, prefix):
        import torch
        if self.tensor_packet is not None:
            self.probe.tensor_bridge(torch.arange(64.).reshape(1, 64, 1), self.tensor_packet)
        # EOS at second token makes all arms comparable without expensive models.
        if prefix:
            return np.array([-1., -1., 4.])
        return np.array([.2, 0., -4.]) if self.tensor_packet is None else np.array([0., 1., -4.])

    def propose(self, prefix, count, length):
        tokens = list(prefix)
        for _ in range(length):
            tokens.append(int(np.argmax(self.next_scores(tuple(tokens)))))
            if tokens[-1] == 2:
                break
        return [SimpleNamespace(tokens=tuple(tokens), text=self.decode(tokens), finished=True)]


class FakeProbe:
    def __init__(self):
        self.tokenizer = SimpleNamespace(eos_token_id=2, decode=lambda t, **k: " ".join(map(str, t)))
        self.processor = SimpleNamespace(tokenizer=self.tokenizer)
        self.tensor_bridge = SpatialEvidenceBridge(1, grid_size=8)

    def context_token_budget(self, image, prompt, reserve):
        return {"fits": True, "input_tokens": len(prompt), "context_limit": 100000}

    def new_answer_session(self, image, prompt):
        return FakeSession(self, prompt)

    def tensor_packet(self, items, image, **kwargs):
        return spatial_packet(items, (8, 8), grid_size=8, **kwargs)

    def new_tensor_answer_session(self, image, prompt, items, **kwargs):
        return FakeSession(self, prompt, self.tensor_packet(items, image, **kwargs))


def source(tmp_path, *, masks=True):
    path = tmp_path / "image.png"
    Image.new("RGB", (8, 8)).save(path)
    row = {"id": "x", "image": str(path), "question": "What is visible?", "answer_type": "closed"}
    config = {"prompt_contract": "anchor-ce-v1"}
    cfg = ValueGenerationConfig(evidence_style="semantic", token_budgeted_evidence=True,
                                compact_native=True, visual_views=0, uncertainty_from_probe=False)
    values = np.zeros((8, 8)); values[1:3, 1:3] = 1
    items = (EvidenceItem("seg", "expert", "segmentation", "anatomy", {"structures": [{
        "label": "lung", "soft_mask": encode_soft_mask(values), "mask_coordinate_system": "original_image"}]}),
        EvidenceItem("cls", "classifier", "classification", "anatomy", {"findings": [{"finding": "lung"}]}))
    items = items if masks else items[1:]
    probe = FakeProbe()
    session = NativeSession(probe, str(path), generation_prompt(row, config), row["question"], cfg)
    session.context(NativeState(items=items))
    old = {"generation_config": asdict(cfg), "evidence": [asdict(i) for i in items],
           "trace": [{"event": "decode", "evidence_transport": copy.deepcopy(session.last_transport)}]}
    options = {"prompt_contract": "uniform", "fixed_strength": .5, "max_strength": 1.,
               "kl_budget": .05, "ablations": True}
    return probe, row, old, {"config": config}, options


def test_real_packet_transport_and_all_arms_with_fake_model(tmp_path):
    inputs = source(tmp_path)
    old = copy.deepcopy(inputs[2])
    result = control_case(*inputs)
    assert set(result["arms"]) == {*ARMS, *ABLATIONS}
    assert result["zero_parity"] and result["controls_available"]
    assert result["retained_other_sources"] == ["classifier"]
    assert result["arms"]["cres"]["score_calls"] == 8
    # Here true and null geometry produce the SAME fake-model score effect;
    # CRES cancels it rather than pretending to have useful geometric evidence.
    assert result["arms"]["cres"]["token_ids"] == result["arms"]["deletion"]["token_ids"]
    assert result["arms"]["native_fixed"]["token_ids"] != result["arms"]["deletion"]["token_ids"]
    assert inputs[2] == old


def test_nonspatial_cases_remain_in_denominator_without_fake_guidance(tmp_path):
    result = control_case(*source(tmp_path, masks=False))
    assert not result["applicable"] and not result["controls_available"]
    assert result["arms"]["cres"]["token_ids"] == result["arms"]["compact"]["token_ids"]
    assert result["arms"]["cres"]["guidance_applied"] is False


def test_changed_question_and_labels_are_rejected(tmp_path):
    probe, row, old, protocol, options = source(tmp_path)
    with pytest.raises(ValueError, match="delivery mismatch"):
        control_case(probe, {**row, "question": "Different question?"}, old, protocol, options)
    with pytest.raises(ValueError, match="contains labels"):
        control_case(probe, {**row, "answer": "yes"}, old, protocol, options)
    old["trace"] = []
    with pytest.raises(ValueError, match="binding audit"):
        control_case(probe, row, old, protocol, options)


def test_uniform_prompt_does_not_use_ce_oe_to_choose_output_contract():
    config = prompt_config({"prompt_contract": "anchor-ce-v1"}, "uniform")
    prompts = [generation_prompt({"question": "Q", "answer_type": kind}, config) for kind in ("closed", "open")]
    assert prompts[0] == prompts[1] and "Yes or No" not in prompts[0]


def test_offline_evaluator_refuses_incomplete_before_importing_scorer(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    mod = importlib.import_module("evaluate_control_evidence")
    for name in mod.SCORERS:
        file = tmp_path / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("# not importable scorer")
    (tmp_path / "frozen.json").write_text(json.dumps({"rows": [], "arms": []}))
    monkeypatch.setattr("sys.argv", ["evaluate", "--run", str(tmp_path), "--anchor-root", str(tmp_path), "--freeze-scorer"])
    mod.main()
    (tmp_path / "complete.json").write_text(json.dumps({"full_manifest_complete": False}))
    monkeypatch.setattr("sys.argv", ["evaluate", "--run", str(tmp_path), "--anchor-root", str(tmp_path)])
    with pytest.raises(ValueError, match="complete full manifest"):
        mod.main()


def test_runner_preflight_live_fake_backend_and_resume(tmp_path, monkeypatch):
    import hashlib

    import torch

    from merit_feddg import generalist_factory
    from merit_feddg.open_study import fingerprint

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    mod = importlib.import_module("run_control_evidence")
    probe, row, historical, protocol, options = source(tmp_path)
    row["image_sha256"] = hashlib.sha256(Path(row["image"]).read_bytes()).hexdigest()
    protocol.update(n=1, identity="source", shards_complete=True)
    protocol["config"]["generalist"] = {"backend": "llava_med", "id": "fake"}
    (tmp_path / "protocol.json").write_text(json.dumps(protocol))
    (tmp_path / "routing.json").write_text(json.dumps({"x": {"group_id": row["image_sha256"]}}))
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(row) + "\n")
    cache = tmp_path / "case-cache" / "compact_rows" / f"{fingerprint('x')}.json"
    cache.parent.mkdir(parents=True)
    cache.write_text(json.dumps({"identity": "source", "output": historical}))
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"schema": "cres-v1", **options}))
    argv = ["run", "--source-run", str(tmp_path), "--manifest", str(manifest),
            "--config", str(cfg), "--output", str(tmp_path / "output")]
    monkeypatch.setattr("sys.argv", [*argv, "--check-only"])
    mod.main()
    assert not (tmp_path / "output").exists()
    probe.model = torch.nn.Linear(1, 1)
    monkeypatch.setattr(generalist_factory, "generalist_provenance", lambda *a: {"fake": True})
    monkeypatch.setattr(generalist_factory, "load_generalist", lambda *a: probe)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a: "FAKE CPU backend")
    original_ones = torch.ones
    monkeypatch.setattr(torch, "ones", lambda *a, **k: original_ones(*a))
    monkeypatch.setattr("sys.argv", argv)
    mod.main()
    roots = list((tmp_path / "output").iterdir())
    assert len(roots) == 1
    output = roots[0] / "cases" / f"{fingerprint('x')}.json"
    before = output.read_bytes()
    assert json.loads((roots[0] / "complete.json").read_text())["n"] == 1
    mod.main()
    assert output.read_bytes() == before
    damaged = json.loads(before)
    damaged["arms"]["cres"]["token_ids"] = []
    output.write_text(json.dumps(damaged))
    with pytest.raises(ValueError, match="invalid/empty"):
        mod.main()
