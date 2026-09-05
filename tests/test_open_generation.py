from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from merit_feddg.block_decode import Block, BlockConfig, QwenBlockSession, decode_blocks
from merit_feddg.contribution import answer_metrics, qualify_contribution
from merit_feddg.med_defer import NativeEvidence
from merit_feddg.open_data import audit_open_split, pixel_digest, read_manifest
from merit_feddg.open_study import choose_expert, run_open_study


def test_shuffled_image_control_uses_same_task_but_not_same_image():
    from merit_feddg.open_study import shuffled_image_donors

    rows = [
        {
            "id": f"s{i}",
            "image_sha256": str(i),
            "modality": "pathology",
            "capability": "classification",
            "task": "open_vqa",
        }
        for i in range(4)
    ]
    donors = shuffled_image_donors(rows)
    assert donors == shuffled_image_donors(list(reversed(rows)))
    assert len({r["id"] for r in donors.values()}) == 4
    assert all(key != donor["id"] for key, donor in donors.items())
    assert shuffled_image_donors(rows[:1]) == {}


def test_explicit_local_checkpoint_requires_no_hf_marker_and_tracks_changes(tmp_path):
    from merit_feddg.open_study import model_provenance

    path = tmp_path / "specialist.pt"
    path.write_bytes(b"test fixture only")
    spec = {"id": "local/my-expert", "checkpoint_path": str(path)}
    before = model_provenance(spec, tmp_path)
    path.write_bytes(b"modified test fixture weights")
    after = model_provenance(spec, tmp_path)
    assert before["file_stats"] != after["file_stats"]
    assert after["snapshot"] == "custom-local"


class Session:
    def __init__(self):
        self.prefixes = []

    def decode(self, tokens):
        return " ".join(
            {1: "wrong", 2: "correct", 3: "detail", 4: "other", 9: ""}[t] for t in tokens
        ).strip()

    def propose(self, prefix, count, length):
        self.prefixes.append(prefix)
        if not prefix:
            return [Block((1,), "wrong", -0.1), Block((2,), "correct", -0.2)][:count]
        return [Block((3,), "detail", -0.1, True), Block((4,), "other", -0.2, True)][:count]


def evidence(claim, prefix):
    return NativeEvidence(
        "specialist",
        "classification",
        {},
        1,
        generated_text="native evidence",
        provenance={"candidate_support": {"b0": 0.0, "b1": 1.0}, "score_semantics": "probability"},
    )


def test_expert_changes_first_high_confidence_block_before_commit():
    session, seen = Session(), []

    def inspect(claim, prefix):
        seen.append((claim, prefix))
        return evidence(claim, prefix)

    out = decode_blocks(
        session,
        question="What tissue?",
        modality="pathology",
        capability="classification",
        config=BlockConfig(max_calls=2),
        evidence=inspect,
        expert_id="specialist",
    )
    assert out["text"] == "correct other"
    assert session.prefixes == [(), (2,)]  # Exact chosen token IDs, not decoded/re-tokenized.
    assert seen[0][1] == ""
    assert seen[1][1] == "correct"
    assert "correct other" in seen[1][0].propositions[1].proposition
    assert all("yes" not in p.answer for c, _ in seen for p in c.propositions)


def test_none_and_budget_and_reversed_support():
    kwargs = {"question": "q", "modality": "pathology", "capability": "classification"}
    out = decode_blocks(
        Session(),
        **kwargs,
        config=BlockConfig(max_calls=0),
        evidence=lambda *_: pytest.fail("expert must not load"),
    )
    assert out["text"] == "wrong detail"
    assert out["expert_calls"] == 0
    reverse = decode_blocks(
        Session(),
        **kwargs,
        config=BlockConfig(),
        evidence=evidence,
        expert_id="specialist",
        reverse_scores=True,
    )
    assert reverse["text"] == "wrong detail"


@pytest.mark.parametrize("capability", ["segmentation", "detection", "retrieval", "generation"])
def test_native_heterogeneous_payloads_reach_same_decoder(capability):
    def infer(claim, prefix):
        kwargs = {"masks": (np.ones((2, 2)),)} if capability == "segmentation" else {}
        if capability == "detection":
            kwargs["boxes"] = ((0.0, 0.0, 1.0, 1.0),)
        if capability in {"retrieval", "generation"}:
            kwargs["generated_text"] = "image-grounded specialist observation"
        return NativeEvidence(
            "native",
            capability,
            {},
            1.0,
            **kwargs,
            provenance={
                "candidate_support": {"b0": 0.0, "b1": 1.0},
                "score_semantics": "probability",
                "references": ["source-case"],
            },
        )

    out = decode_blocks(
        Session(),
        question="q",
        modality="pathology",
        capability=capability,
        config=BlockConfig(),
        evidence=infer,
        expert_id="native",
    )
    assert out["text"] == "correct other"


def test_mask_without_semantic_mapping_cannot_override_language():
    def infer(claim, prefix):
        return NativeEvidence("seg", "segmentation", {}, 1.0, masks=(np.ones((2, 2)),))

    out = decode_blocks(
        Session(),
        question="q",
        modality="pathology",
        capability="segmentation",
        config=BlockConfig(),
        evidence=infer,
        expert_id="seg",
    )
    assert out["text"] == "wrong detail"
    assert all(row["expert"] is None for row in out["trace"])


def test_plausibility_and_finite_validation():
    out = decode_blocks(
        Session(),
        question="q",
        modality="pathology",
        capability="classification",
        config=BlockConfig(plausibility_gap=0.01),
        evidence=evidence,
        expert_id="specialist",
    )
    assert out["text"] == "wrong detail"
    with pytest.raises(ValueError):
        BlockConfig(strength=float("nan"))


def test_empty_eos_continuation_is_supported_and_terminates():
    class EosSession(Session):
        def propose(self, prefix, count, length):
            return [Block((9,), "", -0.1, True), Block((2,), "correct", -0.2, True)]

    out = decode_blocks(
        EosSession(),
        question="q",
        modality="pathology",
        capability="classification",
        config=BlockConfig(),
        evidence=evidence,
        expert_id="specialist",
    )
    assert out["text"] == "correct"
    assert len(out["trace"]) == 1
    assert out["trace"][0]["evidence"]["candidate_support"][1]["support"] == 1.0


def test_continuous_gain_qualification_rejects_bad_source_domain():
    rows = [
        {"role": "source", "domain": domain, "base_f1": 0.2, "guided_f1": 0.45}
        for domain in ("hospital-a", "hospital-b")
        for _ in range(8)
    ]
    card = qualify_contribution(rows)
    assert card["qualified"] and card["robust_gain"] == pytest.approx(0.25)
    rows[-1]["guided_f1"] = 0.0
    rows[-8:] = [{**r, "guided_f1": 0.0} for r in rows[-8:]]
    assert not qualify_contribution(rows)["qualified"]
    with pytest.raises(ValueError, match="target"):
        qualify_contribution([{**rows[0], "role": "target"}])
    assert not qualify_contribution(rows[:8])["qualified"]
    assert not qualify_contribution([])["qualified"]


def test_f1_is_continuous_and_not_a_hallucination_estimator():
    assert answer_metrics("normal tissue", ["normal"]) == {"token_f1": 2 / 3, "exact_match": 0.0}
    assert answer_metrics("no tumor", ["tumor"])["token_f1"] > 0  # Important metric limitation.
    with pytest.raises(ValueError):
        answer_metrics("anything", [])


def make_row(tmp_path, sample_id, role, color):
    path = tmp_path / f"{sample_id}.png"
    Image.new("RGB", (4, 4), color).save(path)
    return {
        "id": sample_id,
        "image": str(path),
        "image_sha256": pixel_digest(path),
        "question": "What tissue?",
        "modality": "pathology",
        "capability": "classification",
        "task": "open_vqa",
        "domain": f"{role}-domain-{color % 2}",
        "domain_kind": "proxy",
        "role": role,
        "group_id": sample_id,
    }


def test_manifest_rejects_label_fields_pixel_mutation_and_group_leak(tmp_path):
    source = make_row(tmp_path, "s", "source", 0)
    target = make_row(tmp_path, "t", "target", 3)
    path = tmp_path / "source.jsonl"
    path.write_text(json.dumps(source))
    assert read_manifest(path, "source")[0]["id"] == "s"
    path.write_text(json.dumps({**source, "answer": "tumor"}))
    with pytest.raises(ValueError, match="fields"):
        read_manifest(path, "source")
    with pytest.raises(ValueError, match="group_id"):
        audit_open_split([source], [{**target, "group_id": "s"}])
    path.write_text(json.dumps(source))
    Image.new("RGB", (4, 4), 7).save(source["image"])
    with pytest.raises(ValueError, match="content changed"):
        read_manifest(path, "source")


def test_multiple_same_modality_experts_fail_closed_without_task_card():
    row = {"modality": "pathology", "capability": "segmentation", "task": "open_vqa"}
    specs = {
        n: {"modalities": ["pathology"], "capabilities": ["segmentation"], "tasks": ["open_vqa"]}
        for n in ("a", "b")
    }
    assert choose_expert(specs, {}, row) is None
    assert (
        choose_expert(
            specs,
            {"a|pathology|classification|open_vqa": {"qualified": True, "robust_gain": 0.5}},
            row,
        )
        is None
    )
    assert (
        choose_expert(
            specs,
            {
                f"{n}|pathology|segmentation|open_vqa": {"qualified": True, "robust_gain": gain}
                for n, gain in (("a", 0.1), ("b", 0.2))
            },
            row,
        )
        == "b"
    )


def test_full_study_target_reference_changes_do_not_change_generations_or_source_cards(
    tmp_path, monkeypatch
):
    import merit_feddg.open_study as study
    from merit_feddg import generalist
    from merit_feddg.io import save_yaml

    source = [make_row(tmp_path, f"s{i}", "source", i) for i in range(4)]
    target = [make_row(tmp_path, "t0", "target", 12)]
    for name, rows in (("source", source), ("target", target)):
        (tmp_path / f"{name}.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    refs = {r["id"]: ["correct other"] for r in source + target}
    ref_path = tmp_path / "references.json"
    ref_path.write_text(json.dumps(refs))
    cfg = {
        "generalist": {"id": "tiny"},
        "prompt_suffix": "Be concise.",
        "decoding": as_config(),
        "qualification": {"min_per_domain": 2},
        "experts": {
            "specialist": {
                "id": "tiny-expert",
                "modalities": ["pathology"],
                "tasks": ["open_vqa"],
                "capabilities": ["classification"],
            }
        },
    }
    config_path = tmp_path / "config.yaml"
    save_yaml(config_path, cfg)
    monkeypatch.setattr(study, "model_provenance", lambda *_: {"test": "mock"})
    monkeypatch.setattr(study, "_extraction_runtime_provenance", lambda: {"test": "mock"})
    monkeypatch.setattr(
        generalist,
        "QwenLayerProbe",
        lambda *a, **k: SimpleNamespace(
            torch=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
            generate=lambda *a, **k: "wrong detail",
        ),
    )
    counters = []

    def new_session(*_):
        counters.append(1)
        return Session()

    class Pool:
        def __init__(self, *_):
            self.models = {}

        def reset_case(self):
            pass

        def evidence_function(self, *_):
            return evidence

    monkeypatch.setattr(study, "QwenBlockSession", new_session)
    monkeypatch.setattr(study, "OpenExpertPool", Pool)
    args = (
        tmp_path / "source.jsonl",
        tmp_path / "target.jsonl",
        ref_path,
        config_path,
        tmp_path,
        tmp_path / "out",
    )
    first = run_open_study(*args)
    count = len(counters)
    assert first["results"]["robust"]["token_f1"] == 1.0
    refs["t0"] = ["entirely different target reference"]
    ref_path.write_text(json.dumps(refs))
    second = run_open_study(*args)
    assert (
        len(counters) == count
    )  # Reference evaluation changes; inference cache remains identical.
    assert first["qualification"] == second["qualification"]
    assert second["results"]["robust"]["token_f1"] == 0.0
    cfg["decoding"]["strength"] = 0.1
    save_yaml(config_path, cfg)
    changed = run_open_study(*args)
    assert changed["run_dir"] != first["run_dir"]
    assert len(counters) > count


def as_config():
    from dataclasses import asdict

    return asdict(BlockConfig(candidates=2, block_tokens=1, max_new_tokens=2, max_calls=2))


def test_qwen_session_real_transformers_cpu_generation_optional():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    if not transformers.utils.is_torch_available():
        pytest.skip("installed torch is below the Transformers minimum version")
    model = transformers.GPT2LMHeadModel(
        transformers.GPT2Config(
            vocab_size=32,
            n_positions=32,
            n_embd=16,
            n_layer=1,
            n_head=2,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
    ).eval()
    processor = SimpleNamespace(
        tokenizer=SimpleNamespace(decode=lambda tokens, **kwargs: " ".join(map(str, tokens)))
    )
    probe = SimpleNamespace(
        torch=torch,
        model=model,
        processor=processor,
        _inputs=lambda *_: {
            "input_ids": torch.tensor([[1, 4]]),
            "attention_mask": torch.ones((1, 2), dtype=torch.long),
        },
    )
    session = QwenBlockSession(probe, Image.new("RGB", (2, 2)), "question")
    first = session.propose((), 3, 3)
    assert 1 <= len(first) <= 3
    assert all(np.isfinite(b.log_probability) and len(b.tokens) <= 3 for b in first)
    if not first[0].finished:
        assert session.propose(first[0].tokens, 3, 2)


def test_tiny_qwen_vl_image_beams_and_committed_prefix_cpu():
    """Real image-bearing Qwen architecture, random tiny weights, no downloads."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    if not transformers.utils.is_torch_available():
        pytest.skip("installed torch is below the Transformers minimum version")
    torch.manual_seed(11)
    config = transformers.Qwen2_5_VLConfig(
        text_config={
            "vocab_size": 64,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "max_position_embeddings": 64,
            "rope_scaling": {"rope_type": "default", "mrope_section": [1, 1, 2]},
        },
        vision_config={
            "depth": 1,
            "hidden_size": 32,
            "out_hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 4,
            "patch_size": 2,
            "spatial_merge_size": 1,
            "temporal_patch_size": 1,
            "window_size": 4,
            "fullatt_block_indexes": [0],
        },
        image_token_id=60,
        video_token_id=61,
        vision_start_token_id=63,
        vision_end_token_id=62,
        eos_token_id=2,
        pad_token_id=0,
    )
    model = transformers.Qwen2_5_VLForConditionalGeneration(config).eval()
    model.generation_config.suppress_tokens = [0, 2, 60, 61, 62, 63]
    processor = SimpleNamespace(
        tokenizer=SimpleNamespace(decode=lambda tokens, **kwargs: " ".join(map(str, tokens)))
    )
    input_ids = torch.tensor([[1, 63, 60, 60, 60, 60, 62, 10]])
    pixels = torch.zeros((4, 12))
    probe = SimpleNamespace(
        torch=torch,
        model=model,
        processor=processor,
        _inputs=lambda *_: {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "pixel_values": pixels,
            "image_grid_thw": torch.tensor([[1, 2, 2]]),
        },
    )
    session = QwenBlockSession(probe, Image.new("RGB", (4, 4)), "question")
    first = session.propose((), 3, 3)
    second = session.propose(first[0].tokens, 3, 2)
    assert len(first) == len(second) == 3
    assert all(len(b.tokens) == 2 and np.isfinite(b.log_probability) for b in second)
    assert session.inputs["input_ids"].shape[1] == 8  # Caller inputs never mutated.
