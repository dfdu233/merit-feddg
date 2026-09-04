from __future__ import annotations

import gc
import json
import weakref
from itertools import pairwise
from pathlib import Path

import numpy as np

import merit_feddg.extract as extraction


def _write_model_marker(root: Path, model_id: str, fingerprint: str) -> None:
    directory = root / "models" / model_id.replace("/", "--")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / ".merit-download-complete.json").write_text(
        json.dumps(
            {
                "schema": 2,
                "id": model_id,
                "kind": "model",
                "revision": "fixed",
                "files": 1,
                "payload_files": 1,
                "bytes": 10,
                "fingerprint": fingerprint,
            }
        ),
        encoding="utf-8",
    )


def test_candidate_only_stage_skips_legacy_null_and_hidden_layer_probe(monkeypatch) -> None:
    class CandidateOnlyGeneralist:
        def __init__(self, *_args, **_kwargs):
            pass

        def candidate_log_likelihoods(self, _image, _prompt, candidates):
            assert candidates == ["one", "two", "three"]
            return np.asarray([-3.0, -1.0, -2.0])

        def probe(self, *_args, **_kwargs):
            raise AssertionError("legacy layer/null probe must not run")

    monkeypatch.setattr(extraction, "QwenLayerProbe", CandidateOnlyGeneralist)
    rows = [
        {
            "id": "case",
            "image": "unused.png",
            "prompt": "Choose one.",
            "candidates": ["one", "two", "three"],
        }
    ]
    state = {"case": {}}
    extraction._extract_generalist_stage(
        rows,
        state,
        {"id": "fake", "layers": [-1], "candidate_scores_only": True},
        None,
    )
    np.testing.assert_allclose(state["case"]["general_null_logits"], np.zeros(3))
    np.testing.assert_allclose(state["case"]["general_visual_layers"], [[-3.0, -1.0, -2.0]])


def test_cache_reuse_requires_exact_manifest_config_and_model_provenance(tmp_path: Path) -> None:
    image = tmp_path / "image.bin"
    image.write_bytes(b"original-pixels")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "id": "one",
                "image": str(image),
                "domain": "center-a",
                "modality": "pathology",
                "prompt": "Choose.",
                "candidates": ["a", "b", "c"],
                "label": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config = {
        "generalist": {"id": "org/general", "adapter": "qwen", "revision": "fixed"},
        "broad_specialist": {
            "id": "org/router",
            "adapter": "clip",
            "revision": "fixed",
        },
        "experts": {
            "path_cls": {
                "id": "org/expert",
                "adapter": "conch",
                "revision": "fixed",
            }
        },
    }
    for index, model_id in enumerate(("org/general", "org/router", "org/expert")):
        _write_model_marker(tmp_path, model_id, f"snapshot-{index}")
    cache = tmp_path / "cache.jsonl"
    record = extraction.EvidenceRecord(
        sample_id="one",
        domain="center-a",
        modality="pathology",
        candidates=["a", "b", "c"],
        label=0,
        general_null_logits=np.zeros(3),
        general_visual_layers=np.zeros((1, 3)),
        expert_scores={"path_cls": np.zeros(3)},
        broad_specialist_scores=np.zeros(3),
        router_probs={"path_cls": 1.0},
    )
    cache.write_text(json.dumps(record.to_json()) + "\n", encoding="utf-8")
    provenance = extraction.extraction_provenance(manifest, config, tmp_path)
    cache.with_suffix(".report.json").write_text(
        json.dumps(
            {
                "records": 1,
                "sample_ids": ["one"],
                "cache_sha256": extraction._file_sha256(cache),
                "extraction_provenance": provenance,
            }
        ),
        encoding="utf-8",
    )

    assert extraction.verify_extraction_cache(cache, manifest, config, tmp_path)["ready"]

    changed = {**config, "generalist": {**config["generalist"], "layers": [3, -1]}}
    report = extraction.verify_extraction_cache(cache, manifest, changed, tmp_path)
    assert not report["ready"]
    assert "extraction-provenance-mismatch" in report["reasons"]

    _write_model_marker(tmp_path, "org/expert", "different-snapshot")
    assert not extraction.verify_extraction_cache(cache, manifest, config, tmp_path)["ready"]

    _write_model_marker(tmp_path, "org/expert", "snapshot-2")
    image.write_bytes(b"replaced-pixels")
    changed_image = extraction.verify_extraction_cache(cache, manifest, config, tmp_path)
    assert not changed_image["ready"]
    assert "extraction-provenance-mismatch" in changed_image["reasons"]

    image.write_bytes(b"original-pixels")
    cache.write_text('{"same-line-count":', encoding="utf-8")
    malformed = extraction.verify_extraction_cache(cache, manifest, config, tmp_path)
    assert not malformed["ready"]
    assert "missing-or-invalid-cache-record" in malformed["reasons"]


def test_sequential_extraction_drops_stage_references_before_release(
    monkeypatch, tmp_path: Path
) -> None:
    events: list[tuple[str, object]] = []
    references: list[tuple[str, weakref.ReferenceType]] = []
    routed_modalities: list[tuple[str, ...]] = []

    class TrackedModel:
        def __init__(self, name: str) -> None:
            self.name = name
            references.append((name, weakref.ref(self)))
            events.append(("init", name))

        def __del__(self) -> None:
            events.append(("del", self.name))

    class FakeGeneralist(TrackedModel):
        def __init__(self, _model_id: str, **_kwargs) -> None:
            super().__init__("generalist")

        def probe(self, _image: str, _prompt: str, candidates: list[str]):
            count = len(candidates)
            return np.zeros(count), np.zeros((2, count))

    class FakeConceptModel(TrackedModel):
        def score_and_domain_embedding(self, _image: str, _prompt: str, candidates: list[str]):
            return np.zeros(len(candidates)), np.zeros(3)

        def route(self, _image: str, modalities: list[str]) -> dict[str, float]:
            routed_modalities.append(tuple(modalities))
            return {name: 1.0 / len(modalities) for name in modalities}

    def fake_factory(spec: dict, _artifact_root):
        return FakeConceptModel(str(spec["id"]))

    def observe_release(_model=None) -> None:
        gc.collect()
        alive = tuple(name for name, reference in references if reference() is not None)
        events.append(("release", alive))

    monkeypatch.setattr(extraction, "QwenLayerProbe", FakeGeneralist)
    monkeypatch.setattr(extraction, "_expert_from_spec", fake_factory)
    monkeypatch.setattr(extraction, "_release", observe_release)

    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "id": "sample",
                "image": "unused.png",
                "domain": "center-target",
                "modality": "alpha",
                "prompt": "Choose a class.",
                "candidates": ["a", "b"],
                "label": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config = {
        "generalist": {"id": "general", "layers": [-1]},
        "broad_specialist": {"id": "broad", "adapter": "contrastive_biomedclip"},
        "router": {"adapter": "biomedclip_router", "abstain_entropy": 1.1},
        "experts": {
            "alpha_classifier": {
                "id": "expert-alpha-classifier",
                "adapter": "fake",
                "modalities": ["alpha"],
                "capabilities": ["classification"],
            },
            "alpha_segmenter": {
                "id": "expert-alpha-segmenter",
                "adapter": "fake",
                "modalities": ["alpha"],
                "capabilities": ["segmentation"],
            },
            "zeta_classifier": {
                "id": "expert-zeta-classifier",
                "adapter": "fake",
                "modalities": ["zeta"],
                "capabilities": ["classification"],
            },
        },
    }

    records = extraction.extract_manifest(manifest, config, tmp_path / "cache.jsonl")

    assert len(records) == 1
    assert [value for event, value in events if event == "init"] == [
        "generalist",
        "broad",
        "expert-alpha-classifier",
        "expert-alpha-segmenter",
        "expert-zeta-classifier",
    ]
    releases = [value for event, value in events if event == "release"]
    assert releases == [()] * (2 + len(config["experts"]))
    assert all(reference() is None for _, reference in references)
    assert routed_modalities == [("alpha", "zeta")]
    assert set(records[0].expert_scores) == {
        "alpha_classifier",
        "alpha_segmenter",
        "zeta_classifier",
    }
    assert set(records[0].router_probs) == set(records[0].expert_scores)
    assert records[0].router_probs == {
        "alpha_classifier": 0.25,
        "alpha_segmenter": 0.25,
        "zeta_classifier": 0.5,
    }
    assert records[0].metadata["modality_router_probs"] == {"alpha": 0.5, "zeta": 0.5}
    assert records[0].metadata["expert_capabilities"]["alpha_segmenter"] == ["segmentation"]

    for current, following in pairwise(events):
        assert not (current[0] == "release" and following[0] == "del")
