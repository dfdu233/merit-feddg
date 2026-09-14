import importlib
from pathlib import Path
from types import SimpleNamespace


def test_retrieval_uses_actual_items_and_shared_prefix_decoder(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    mod = importlib.import_module('run_channel_soft_canary')
    block = SimpleNamespace(text='candidate', tokens=(1,))
    base = SimpleNamespace(propose=lambda *a, **k: [block])

    class Session:
        def __init__(self, *a):
            self.last_transport = {'presented': []}
            self._session = base

        def propose(self, state, *a):
            self.last_transport = {'presented': [
                {'expert_id': e.expert_id, 'evidence_id': e.evidence_id} for e in state.items]}
            return block

        def context(self, *a):
            return 'image', 'prompt'

    monkeypatch.setattr(mod, 'NativeSession', Session)
    calls = []

    def decode(a, b, **kwargs):
        calls.append(kwargs['alpha'])
        return {'text': 'candidate', 'token_ids': [1]}

    monkeypatch.setattr(mod, 'decode', decode)
    probe = SimpleNamespace(new_answer_session=lambda *a: base,
                            tokenizer=SimpleNamespace(eos_token_id=2))
    evidence = mod.EvidenceItem('id', 'kb', 'retrieval', 'general_knowledge',
                               {'knowledge': [{'text': 'sourced text'}]})
    kb = SimpleNamespace(infer=lambda *a: SimpleNamespace(items=(evidence,)))
    old = {'evidence': [], 'token_ids': [1], 'generation_config': {
        'evidence_style': 'semantic', 'token_budgeted_evidence': True, 'visual_views': 0}}
    row = {'id': 'case', 'image': 'image', 'question': 'question', 'answer_type': 'open'}
    result = mod.case(probe, row, old, {'config': {'prompt_contract': 'anchor-ce-v1'}}, 'retrieval', kb)
    assert result['status'] == 'real_candidate'
    assert result['selected_evidence'] == ['id']
    assert result['native_class_to_token_logits'] is False
    assert calls == [0, .5]
