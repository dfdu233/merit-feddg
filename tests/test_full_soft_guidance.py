import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_full_control_drift_is_explicit_and_fallback_uses_current_not_historical(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    full=importlib.import_module('run_soft_guidance_full')
    block=SimpleNamespace(text='current',tokens=(2,))
    class Session:
        def __init__(self,*a): self.last_transport={'presented':[]}
        def propose(self,*a): return block
    class EmptyPacket:
        rejected=()
        def __len__(self): return 0
    monkeypatch.setattr(full,'NativeSession',Session)
    probe=SimpleNamespace(tokenizer=SimpleNamespace(eos_token_id=2),
        tensor_packet=lambda *a,**k:EmptyPacket(),
        new_answer_session=lambda *a:SimpleNamespace(propose=lambda *a,**k:[block]))
    old={'text':'old','token_ids':[1],'seconds':1.,'evidence':[],
         'generation_config':{'evidence_style':'semantic','token_budgeted_evidence':True,'visual_views':0}}
    row={'id':'case','image':'unused','question':'question','answer_type':'closed'}
    protocol={'config':{'prompt_contract':'anchor-ce-v1'}}
    result=full.full_case(probe,row,old,old,protocol,.5)
    assert result['historical_compact_token_parity'] is False
    assert result['applicable'] is False and result['new_expert_calls']==0
    assert set(result['arms'])==set(full.ARMS)
    for arm in ('native_spatial_soft','text_soft','other_text_only'):
        assert result['arms'][arm]['text']=='current'
        assert result['arms'][arm]['guidance_applied'] is False
    old['trace']=[{'event':'decode','evidence_transport':{'presented':[{'expert':'changed'}]}}]
    with pytest.raises(RuntimeError,match='delivery changed'):
        full.full_case(probe,row,old,old,protocol,.5)


def test_full_evaluator_refuses_incomplete_before_loading_references(monkeypatch,tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    evaluator=importlib.import_module('evaluate_soft_guidance_full')
    monkeypatch.setattr(evaluator,'scorer_hashes',lambda:{'frozen':'hash'})
    (tmp_path/'frozen.json').write_text('{}')
    (tmp_path/'scorer-frozen.json').write_text(json.dumps({'frozen':'hash'}))
    (tmp_path/'complete.json').write_text(json.dumps({'full_dataset_complete':False}))
    monkeypatch.setattr('sys.argv',['evaluate','--run',str(tmp_path)])
    with pytest.raises(RuntimeError,match='full completion required'):
        evaluator.main()
    assert not (tmp_path/'evaluation.json').exists()


def test_full_paired_counts_keep_continuous_open_scores(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    paired=importlib.import_module('evaluate_soft_guidance_full').paired
    result=paired({'a':1.,'b':.25},{'a':.5,'b':.75},['a','b'])
    assert result=={'n':2,'mean_delta':0.,'score_improvements':1,'score_harms':1}
