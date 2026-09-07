import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from merit_feddg import applicability_run as runner
from merit_feddg.open_data import pixel_digest


def test_collect_evaluate_and_cached_resume(tmp_path, monkeypatch):
    rows = []
    for i in range(8):
        path = tmp_path / f"{i}.png"
        Image.new("RGB", (16, 16), (i, 0, 0)).save(path)
        rows.append({"id": str(i), "image": str(path), "question": "What finding?",
            "modality": "cxr", "task": "vqa", "capability": "generation", "role": "source",
            "domain": f"d{i//2}", "domain_kind": "independent", "group_id": str(i),
            "image_sha256": pixel_digest(path)})
    source, refs, config = [tmp_path / p for p in ("source.jsonl", "refs.json", "config.yaml")]
    source.write_text("\n".join(json.dumps(r) for r in rows))
    refs.write_text(json.dumps({r["id"]: ["finding"] for r in rows}))
    config.write_text("""generalist:
  id: fake
  backend: llava_med
experts:
  A:
    id: fake-expert
applicability:
  admission:
    neighbors: 2
    min_per_domain: 2
    min_domains: 2
  generation:
    max_new_tokens: 4
""")
    loads = []
    monkeypatch.setattr(runner, "generalist_provenance", lambda *a: {"fake": True})
    monkeypatch.setattr(runner, "model_provenance", lambda *a: {"fake": True})
    monkeypatch.setattr(runner, "load_generalist", lambda *a: loads.append(1) or object())

    class Pool:
        def __init__(self, *args, **kwargs):
            pass
        def domain_embedding(self, expert, image):
            return np.array([1., 0.])
        def reset_case(self):
            pass
        def clear(self):
            pass
    class Runtime:
        def __init__(self, session, pool, row, specs, decoder):
            self.row = row
        def descriptors(self, state):
            return [{"expert": "A", "capability": "classification", "scope": "findings"}]
        def run(self, mode, applicability=None):
            if applicability:
                assert applicability.decide(self.row, self.descriptors(None)[0])["allowed"]
            return {"text": "finding" if mode != "generalist" else "unknown", "expert_calls": 0,
                    "seconds": 0., "trace": [], "token_ids": [1]}
        def execute(self, state, descriptor):
            return SimpleNamespace(), {"reason": "ok", "adopted": True}
        def complete(self, state):
            return {"text": "finding", "token_ids": [2]}
    monkeypatch.setattr(runner, "CapabilityPool", Pool)
    monkeypatch.setattr(runner, "CapabilityRuntime", Runtime)
    output = tmp_path / "out"
    common = ["applicability", "--source-manifest", str(source), "--references", str(refs),
              "--config", str(config), "--output", str(output)]
    monkeypatch.setattr(sys, "argv", common+["--stage", "collect"])
    runner.main()
    memory = json.loads((output / "memory.json").read_text())
    assert memory["metric"]["name"] == "paired_generation_harm"
    assert sum(len(v) for v in memory["scopes"].values()) == 8
    monkeypatch.setattr(sys, "argv", common+["--stage", "evaluate", "--memory",
                        str(output / "memory.json"), "--limit", "2"])
    runner.main()
    assert len(json.loads((output / "result.json").read_text())["results"]) == 6
    old_loads = len(loads)
    runner.main()
    assert len(loads) == old_loads
    refs.write_text(json.dumps({r["id"]: ["changed"] for r in rows}))
    with pytest.raises(ValueError, match="source/retrieval"):
        runner.main()
