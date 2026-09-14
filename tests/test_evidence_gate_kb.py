import copy
import json
from pathlib import Path

import pytest
from PIL import Image

from merit_feddg.evidence_html import render
from merit_feddg.evidence_use import filter_evidence, purpose_prompt, restore_contracts
from merit_feddg.experts.native_small_medical import MedMNISTExpert, UKANExpert, load_bundle
from merit_feddg.plug_observe import available_actions, observation, run_case
from merit_feddg.text_kb import TextKnowledgeExpert, build, query


def spec():
    return {'e':{'id':'model', 'scope':'anatomy', 'description':'Anatomy classification',
        'modalities':['ct'], 'tasks':['open_vqa'], 'capabilities':['classification'],
        'requires_region':False, 'request_contract':{'mode':'named_concepts',
                                                  'concept_aliases':{'liver':['hepatic']}}}}


def case():
    return {'id':'sensitive-case', 'image':'unused', 'question':'Where is the liver?',
            'modality':'ct', 'task':'open_vqa'}


def obs(name='a', **changes):
    item = {'expert_id':'e','evidence_id':name,'scope':'anatomy','capability':'classification',
            'payload':{'catalog':[{'concept':'liver','score':.2}]}, 'confidence':None,
            'summary':'private clinical observation', 'provenance':{}}
    return observation(item | changes, 'sensitive-case')


def test_existing_scope_restored_without_changing_old_default():
    c = case() | {'question':'What is the heart size?'}
    assert available_actions(spec(),c,[],set(),(20,20),2)[0]
    assert not available_actions(spec(),c,[],set(),(20,20),2,True)[0]


def test_restore_only_missing_contract_not_models_or_conflicting_policy():
    specs = spec()
    contract = specs['e'].pop('request_contract')
    overrides = {'e': {'request_contract': contract}}
    assert restore_contracts(specs, overrides) == ['e']
    assert specs['e']['id'] == 'model'
    assert restore_contracts(specs, overrides) == []
    overrides['e']['request_contract'] = {'mode': 'free_query'}
    with pytest.raises(ValueError, match='no automatic policy override'):
        restore_contracts(specs, overrides)
    with pytest.raises(ValueError, match='request_contract only'):
        restore_contracts(specs, {'e': {'id': 'replacement'}})


@pytest.mark.parametrize('use,admit',[('ANSWER',True),('AUXILIARY',False),('IRRELEVANT',False),('UNKNOWN',False),('garbage',False)])
def test_inherited_and_new_evidence_same_gate(use, admit):
    items = [obs('old'), obs('new')]
    before = copy.deepcopy(items)
    kept,audit = filter_evidence(case(),items,spec(),lambda _:use)
    assert len(kept)==(2 if admit else 0)
    assert items==before and len(audit)==2
    assert all(a['confidence'] is None and a['medical_correctness']=='not_assessed' for a in audit)


def test_conflict_not_automatic_rejection_and_parent_dependency():
    parent = obs(payload={'negation':True, 'disagrees_with_generalist':True})
    kept,_=filter_evidence(case(),[parent],spec(),lambda _:'ANSWER')
    assert kept==[parent]
    child=observation(obs()['artifact'],'sensitive-case',parent=parent['ref'])
    kept,audit=filter_evidence(case(),[parent,child],spec(),lambda o:'AUXILIARY' if o==parent else 'ANSWER')
    assert kept==[] and audit[-1]['reason']=='parent_not_answer_evidence'
    assert 'not its medical correctness' in purpose_prompt('q',parent)


def test_real_runner_filters_answer_prompt_and_audits_rejected(tmp_path):
    im=tmp_path/'image.png'; Image.new('RGB',(20,20)).save(im)
    prompts=[]
    def generate(image,prompt,tokens,allowed):
        prompts.append(prompt)
        return {'text':'AUXILIARY' if allowed else 'candidate','token_ids':[1]}
    out=run_case(case=case()|{'image':str(im)},specs=spec(),seed_items=[obs()['artifact']],
        incumbent={'text':'baseline','token_ids':[2]}, generate=generate,
        measure=lambda *a:{'fits':True}, invoke=lambda *a:[],output_dir=tmp_path/'crops',
        mode='read', evidence_gate='purpose')
    assert out['text']=='baseline' and out['token_ids']==[2]
    assert len(prompts)==1  # use judgement ran, final answer did NOT run
    assert out['agent_workflow']['delivery']['reason']=='no_answer_evidence_after_gate'
    assert len(out['agent_workflow']['observations'])==1
    assert out['agent_workflow']['candidate'] is None


def test_kb_real_seed_query_no_overwrite_no_patient_fact(tmp_path):
    seed=Path(__file__).parents[1]/'assets/knowledge/imaging_seed.json'
    db=tmp_path/'kb.sqlite'
    assert build(seed,db)['documents']==6
    result=query(db,'MRI magnetic fields')
    assert result and all(not r['current_patient_fact'] and r['source_url'] for r in result)
    with pytest.raises(FileExistsError): build(seed,db)
    assert query(db,'"; DROP TABLE docs; --')==[]
    assert query(db,'MRI magnetic fields')==result  # index survives SQL-like query text
    from merit_feddg.capabilities import CapabilityRequest
    r=CapabilityRequest('x','no-image','MRI magnetic fields','mri','open_vqa','source','group','retrieval',scope='general_knowledge')
    assert TextKnowledgeExpert(str(db)).infer(r).items[0].payload['knowledge']


@pytest.mark.parametrize('cls',[MedMNISTExpert,UKANExpert])
def test_no_random_init_when_weights_missing(cls,tmp_path):
    with pytest.raises(FileNotFoundError,match='no random'): cls(str(tmp_path/'missing.json'))


def test_bundle_digest_rejects_before_model_loading(tmp_path):
    weight=tmp_path/'w';weight.write_bytes(b'not a checkpoint')
    b=tmp_path/'bundle.json';b.write_text(json.dumps({'kind':'ukan-busi',
        'source_url':'https://github.com/CUHK-AIM-Group/U-KAN','checkpoint':str(weight),
        'checkpoint_sha256':'wrong'}))
    with pytest.raises(ValueError,match='digest'): load_bundle(b,'ukan-busi')


def test_html_default_no_patient_text_paths_images_or_scripts():
    out={'incumbent':{'patient-secret':{'text':'<script>private</script>', 'token_ids':[1],
         'image':'/patient/private.png','agent_workflow':{'observations':[obs()]}}}}
    page=render(out)
    assert 'patient-secret' not in page and 'private' not in page and '<script>' not in page
    assert '<img' not in page and 'scope_hash' in page
    assert '&lt;script&gt;private' in render(out,True)
