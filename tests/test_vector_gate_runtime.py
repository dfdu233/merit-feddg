import pytest
from test_capability_runtime import Probe, spec
from test_capability_runtime import setup as setup  # noqa: PLC0414 -- pytest fixture

from merit_feddg.capability_runtime import NativeState
from merit_feddg.tensor_evidence import TensorContract, compile_tensor_evidence


@pytest.mark.parametrize("accepted", [False, True])
def test_runtime_adopts_only_gated_items_and_keeps_prefix(setup, monkeypatch, accepted):
    build, _, _ = setup
    contract = TensorContract(("tissue",), ("scope",), ({
        "expert": "A", "scope": "native-classification", "label": "tissue",
        "concept": "tissue", "scope_id": "scope"},), grid_size=2)

    def packet(self, items, image):
        return compile_tensor_evidence(items, (32, 24), contract)

    monkeypatch.setattr(Probe, "tensor_bridge", object(), raising=False)
    monkeypatch.setattr(Probe, "tensor_packet", packet, raising=False)
    monkeypatch.setattr(Probe, "new_tensor_answer_session", lambda *a: None, raising=False)
    runtime, _, pool = build(specs={"A": spec()}, evidence_style="tensor", visual_views=0,
                                    vector_gate="visual_contrast")
    # The fixture's classifier output is made native-tensor compatible.
    original = pool.infer

    def infer(expert, request):
        from dataclasses import replace

        result = original(expert, request)
        item = replace(result.items[0], payload={"findings": [{"finding": "tissue", "score": 0.8}],
                                                "score_semantics": "probability"})
        return replace(result, items=(item,))

    pool.infer = infer
    seen = []

    def decide(state, proposed):
        seen.append((state.prefix, proposed))
        return {"accepted": accepted, "reason": "test_decision"}

    runtime.session.assess_vector_evidence = decide
    state = NativeState(prefix=(9, 10))
    result, trace = runtime.execute(state, runtime.descriptors(state)[0])
    assert result.prefix == state.prefix and not state.items
    assert len(result.history) == 1 and len(pool.calls) == 1
    assert bool(result.items) == trace["adopted"] == accepted
    assert trace["vector_gate"]["accepted"] == accepted and seen[0][0] == state.prefix
    assert "native_evidence" in trace  # Raw rejected evidence stays auditable.


def test_vector_protocol_refuses_a_text_configuration():
    from merit_feddg.capability_runtime import ValueGenerationConfig
    from merit_feddg.matched_evaluation import experiment_arms

    with pytest.raises(ValueError, match="vector protocol"):
        experiment_arms(ValueGenerationConfig(), "vector")
