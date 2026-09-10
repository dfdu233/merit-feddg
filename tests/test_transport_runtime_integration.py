from test_capability_runtime import Probe, spec
from test_capability_runtime import setup as setup  # noqa: PLC0414 -- expose pytest fixture


def budget(self, image, prompt, reserve):
    tokens = 50 + 800 * prompt.count('"expert_id"')
    return {"fits": tokens + reserve <= 950, "input_tokens": tokens,
            "reserved_tokens": reserve, "context_limit": 950,
            "remaining_tokens": 950 - tokens - reserve}


def test_decode_reports_actual_presented_records_not_historical_adoption(setup, monkeypatch):
    monkeypatch.setattr(Probe, "context_token_budget", budget, raising=False)
    build, _, _ = setup
    runtime, _, pool = build(specs={"A": spec(), "B": spec()}, visual_views=0,
                             token_budgeted_evidence=True, evidence_order="acquisition",
                             max_evidence_chars=10000)
    result = runtime.run("all_evidence")
    assert len(pool.calls) == 2  # No invented pre-call estimate for unknown packet sizes.
    assert result["presented_evidence_count"] == 1
    events = [e for e in result["trace"] if e["event"] == "tool"]
    assert events[0]["adopted"] and not events[1]["adopted"]
    assert events[1]["packing_preview"]["omitted"][0]["reason"] == "token_budget"
    decode = next(e for e in result["trace"] if e["event"] == "decode")
    assert decode["evidence_transport"]["presented"] == [{"expert_id": "A", "evidence_id": "A:native"}]


def test_declared_minimum_packet_budget_avoids_an_unusable_call(setup, monkeypatch):
    monkeypatch.setattr(Probe, "context_token_budget", budget, raising=False)
    build, _, _ = setup
    runtime, _, pool = build(specs={"A": spec(), "B": spec(minimum_evidence_tokens=800)},
                             visual_views=0, token_budgeted_evidence=True,
                             evidence_order="acquisition", max_evidence_chars=10000)
    result = runtime.run("all_evidence")
    assert len(pool.calls) == 1
    assert any(e.get("reason") == "declared_packet_cannot_fit" for e in result["trace"])
