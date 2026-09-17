import sys
from pathlib import Path
from dataclasses import replace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from native_quilt_factory import NativeQuiltExpert
from merit_feddg.capabilities import CapabilityRequest,validate_result


def request():
    return CapabilityRequest(sample_id='case',image='image.png',question='What is visible?',
        modality='pathology',task='open_vqa',domain='test',group_id='image',
        capability='generation',scope='histology_question_observation',query='Describe visible observations.')


def test_native_contract_and_request_template():
    expert=NativeQuiltExpert('local-model','quilt_pathology','histology_question_observation','out','gpu')
    class Backend:
        def generate_with_usage(self,image,prompt,max_new_tokens):
            assert 'requested_observation' in prompt and 'Describe visible observations.' in prompt
            assert image=='image.png' and max_new_tokens==64
            return {'text':'Observation.','input_tokens':30,'output_tokens':4}
    expert.probe=Backend()
    r=request();result=validate_result(expert.infer(r),'quilt_pathology',r)
    item=result.items[0]
    assert item.payload=={'generated_text':'Observation.','usage':{'input_tokens':30,'output_tokens':4},
                          'observation_status':'unverified_specialist_output','empty_generation':False}
    assert item.provenance['adapter']=='native_quilt_vqa'
    assert item.confidence is None


def test_wrong_scope_does_not_call_backend():
    expert=NativeQuiltExpert('local-model','quilt_pathology','histology_question_observation','out','gpu')
    expert.probe=None
    result=expert.infer(replace(request(),scope='other'))
    assert not result.items and result.reason=='wrong_scope'
