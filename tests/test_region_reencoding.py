"""Synthetic image/tensor tests; no pretrained weights or clinical claims."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from merit_feddg.region_reencoding import (
    ARMS,
    VIEW_ARMS,
    _overlap_axis,
    align_features,
    factorial_contrasts,
    fuse_features,
    pixel_hash,
    projected_view,
    region_plan,
    region_views,
)


def mask():
    value = np.zeros((4, 4))
    value[1:3, 0:1] = .8
    return value.reshape(1, -1)


def plan():
    value, audit = region_plan(mask(), (80, 80), 32)
    assert audit['reason'] == 'available'
    return value


@pytest.mark.parametrize('shape', [(0,), (1, 3), (2, 6), (2, 4, 4)])
def test_invalid_region_shape(shape):
    with pytest.raises(ValueError):
        region_plan(np.zeros(shape), (80, 80), 32)


@pytest.mark.parametrize('number', [-.1, 1.1, float('nan'), float('inf')])
def test_invalid_masks(number):
    value = mask()
    value[0, 0] = number
    with pytest.raises(ValueError):
        region_plan(value, (80,80), 32)


@pytest.mark.parametrize('size', [(1,80), (True,80), (80.,80), (80,0)])
def test_bad_source_dimensions(size):
    with pytest.raises(ValueError):
        region_plan(mask(), size, 32)


def test_no_resolution_advantage_is_unavailable():
    value, audit = region_plan(mask(), (32,32),32)
    assert value is None and audit['reason']=='no_native_resolution_advantage'


def test_empty_and_full_frame_masks_never_invent_roi():
    for values in (np.zeros((0,16)), np.zeros((1,16)), np.ones((1,16))):
        value, audit = region_plan(values,(80,80),32)
        assert value is None and audit['reason']=='no_nondegenerate_matched_region'


def test_selection_is_stored_order_not_labels_or_scores():
    values = np.concatenate([np.ones((1,16)), mask(), mask()])
    before = values.copy()
    value, _ = region_plan(values,(80,80),32)
    assert value['region_index']==1
    np.testing.assert_array_equal(values,before)


def test_controls_keep_area_shape_and_orientation():
    m = mask().reshape(4,4)
    m[1,0],m[2,0]=.6,.9
    p,_=region_plan(m.reshape(1,-1),(80,80),32)
    a,b=p['real_box'],p['displaced_box']
    assert (a[2]-a[0],a[3]-a[1])==(b[2]-b[0],b[3]-b[1])
    ra,rb=np.asarray(p['real_support']),np.asarray(p['displaced_support'])
    np.testing.assert_array_equal(np.sort(ra.ravel()),np.sort(rb.ravel()))
    np.testing.assert_array_equal(ra[ra>0],rb[rb>0])
    assert a!=b


def test_padding_cells_are_not_read_or_modified():
    values=np.zeros((4,4))
    values[0,0]=1.
    values[1,0]=.8
    p,_=region_plan(values.reshape(1,-1),(80,40),16)
    assert p is not None
    for where in ('real','displaced'):
        support=np.asarray(p[where+'_support'])
        assert not support[0].any() and not support[3].any()
        assert p[where+'_box'][1]>=20 and p[where+'_box'][3]<=60


def test_matched_views_retain_original_and_exact_shape():
    array=np.zeros((80,80,3),dtype=np.uint8)
    array[::2,:,0]=255
    array[:,::2,1]=255
    image=Image.fromarray(array)
    before=pixel_hash(image)
    views,audit=region_views(image,plan(),(123,117,104))
    assert set(views)==set(VIEW_ARMS)
    assert len({v.size for v in views.values()})==1
    assert pixel_hash(image)==before
    assert audit['real_mean_abs_pixel_difference']>0
    assert audit['displaced_mean_abs_pixel_difference']>0
    box=plan()['real_box']
    np.testing.assert_array_equal(np.asarray(views['native_real']),array[box[1]:box[3],box[0]:box[2]])


def test_constant_pixels_do_not_fabricate_detail():
    image=Image.new('RGB',(80,80),(12,34,56))
    _,audit=region_views(image,plan(),(12,34,56))
    assert audit['real_mean_abs_pixel_difference']==0
    assert audit['displaced_mean_abs_pixel_difference']==0


def test_plan_source_mismatch_fails():
    with pytest.raises(ValueError):
        region_views(Image.new('RGB',(79,80)),plan(),(0,0,0))


@pytest.mark.parametrize('background', [(0,0),(-1,0,0),(1.5,0,0),(True,0,0)])
def test_bad_background(background):
    with pytest.raises(ValueError):
        region_views(Image.new('RGB',(80,80)),plan(),background)


@pytest.mark.parametrize('begin,end,pad,local_side', [(0,20,10,40),(20,60,0,40),(0,80,0,80)])
def test_overlap_is_nonnegative_partition(begin,end,pad,local_side):
    weights=_overlap_axis(4,80,begin,end,local_side,pad)
    assert (weights>=0).all()
    sums=weights.sum(1)
    assert np.all(np.isclose(sums,0)|np.isclose(sums,1))


def test_full_frame_alignment_exact_identity():
    local=torch.arange(48,dtype=torch.float32).reshape(1,16,3)
    aligned=align_features(local,[0,0,80,80],side=80,grid=4)
    assert torch.equal(aligned,local)


def test_rectangular_crop_padding_is_inverted():
    # Crop 20x40 -> square40, horizontal padding10 on both sides.
    # Local columns 0/3 are padding; they must not leak into the original ROI.
    local=torch.zeros((1,16,1)).reshape(4,4,1)
    local[:,0]=100
    local[:,3]=100
    local[:,1:3]=2
    out=align_features(local.reshape(1,16,1),[0,20,20,60],side=80,grid=4).reshape(4,4)
    assert out[1,0]==2 and out[2,0]==2
    assert out.sum()==4


@pytest.mark.parametrize('box', [[-1,0,20,20],[20,0,0,20],[0,0,81,20],[0.,0,20,20]])
def test_invalid_alignment_boxes(box):
    with pytest.raises(ValueError):
        align_features(torch.ones(1,16,3),box,side=80,grid=4)


def test_original_shape_and_outside_support_are_exact():
    original=torch.randn(1,16,7,dtype=torch.float16)
    local=torch.randn_like(original)
    m=torch.tensor(mask().reshape(-1))
    before=original.clone()
    out=fuse_features(original,local,m)
    assert out.shape==original.shape and out.dtype==original.dtype
    assert torch.equal(out[:,m==0],original[:,m==0])
    assert torch.equal(original,before)
    assert torch.equal(fuse_features(original,local,torch.zeros(16)),original)


def test_convex_fusion_does_not_extrapolate_features():
    original=torch.zeros(1,16,1)
    local=torch.ones_like(original)*10
    out=fuse_features(original,local,np.ones((4,4)))
    assert torch.equal(out,torch.ones_like(original)*5)


@pytest.mark.parametrize('bad', [np.zeros(15), np.full(16,-1),np.full(16,np.nan)])
def test_bad_support_fails(bad):
    with pytest.raises(ValueError):
        fuse_features(torch.zeros(1,16,1),torch.zeros(1,16,1),bad)


def test_hook_is_scoped_and_zero_support_is_exact():
    module=torch.nn.Identity()
    original=torch.randn(1,16,3)
    with projected_view(module,torch.ones_like(original),np.zeros(16)) as audit:
        out=module(original)
    assert torch.equal(out,original)
    assert audit['projector_calls']==1 and audit['max_abs_feature_delta']==0
    assert not module._forward_hooks
    assert not hasattr(module,'_merit_reencoding_active')


def test_hook_cleanup_on_exception_and_nesting():
    module=torch.nn.Identity()
    value=torch.ones(1,16,3)
    with (
        pytest.raises(RuntimeError,match='nested'),
        projected_view(module,value,np.ones(16)),
        projected_view(module,value,np.ones(16)),
    ):
        pass
    assert not module._forward_hooks and not hasattr(module,'_merit_reencoding_active')
    with pytest.raises(ValueError), projected_view(module,value,np.ones(16)):
        module(torch.zeros(1,15,3))
    assert not module._forward_hooks


def test_factorial_is_not_deletion_gain_disguised_as_detail():
    values=dict(zip(VIEW_ARMS,[.6,.6,.6,.6]))
    assert set(factorial_contrasts(values).values())=={0}
    values=dict(zip(VIEW_ARMS,[.8,.6,.65,.6]))
    result=factorial_contrasts(values)
    assert result['detail_location_interaction']==pytest.approx(.15)


def test_missing_factorial_cell_rejected():
    with pytest.raises(ValueError):
        factorial_contrasts({'native_real':1.})


def test_arms_and_cli_have_bounded_defaults():
    assert len(ARMS)==7 and ARMS[:3]==('compact','deletion','identity')
    root=Path(__file__).resolve().parents[1]
    result=subprocess.run([sys.executable,str(root/'scripts/run_region_reencoding.py'),'--help'],
        capture_output=True,text=True,check=True)
    assert '--max-cases' in result.stdout and '--gpu-uuid' in result.stdout
    result=subprocess.run([sys.executable,str(root/'scripts/evaluate_region_reencoding.py'),'--help'],
        capture_output=True,text=True,check=True)
    assert '--partial-diagnostic' in result.stdout


def test_live_adapter_path_with_explicit_fake_backend(monkeypatch, tmp_path):
    """Exercise actual adapter control flow, not a medical model/processor test."""
    import dataclasses
    import hashlib
    import json
    import types

    from merit_feddg.region_reencoding import reencoding_case

    @dataclasses.dataclass
    class Item:
        evidence_id: str
        expert_id: str
        capability: str
        scope: str
        payload: dict
        provenance: dict = dataclasses.field(default_factory=dict)

    @dataclasses.dataclass
    class Config:
        semantic_spatial: bool = False
        vector_gate: str = 'off'
        evidence_style: str = 'semantic'
        compact_native: bool = True
        max_new_tokens: int = 64
        block_tokens: int = 64
        behavior_probe: str = 'off'
        uncertainty_from_probe: bool = False

    @dataclasses.dataclass
    class State:
        items: tuple = ()

    def transport(items, prompt):
        entries = [{'expert_id':x.expert_id,'evidence_id':x.evidence_id} for x in items]
        return {'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),
                'evidence_sha256':hashlib.sha256(json.dumps(entries).encode()).hexdigest(),
                'presented':entries,'omitted':[], 'context':{}}

    class Native:
        def __init__(self, probe, image, prompt, question, cfg):
            self.probe,self.image,self.prompt=probe,image,prompt
        def context(self,state):
            self.last_transport=transport(state.items,self.prompt)
            return self.image,self.prompt
        def propose(self,state,budget):
            image,prompt=self.context(state)
            return self.probe.new_answer_session(image,prompt).propose((),count=1,length=budget)[0]
        def decode(self,tokens):
            return 'stable'

    def module(name, **attrs):
        value=types.ModuleType(name)
        value.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules,name,value)
    module('merit_feddg.capabilities',EvidenceItem=Item)
    module('merit_feddg.capability_runtime',NativeSession=Native,NativeState=State,ValueGenerationConfig=Config)
    def validate(record,names):
        assert set(record['arms'])==set(names)
        assert all(v['text'] and v['token_ids'] for v in record['arms'].values())
    module('merit_feddg.control_study',prompt_config=lambda c,k:{**c,'suffix':' uniform'},
           reuse=lambda v,r:{**v,'seconds':0,'reason':r},validate_outputs=validate)
    module('merit_feddg.matched_evaluation',generation_prompt=lambda r,c:r['question']+c.get('suffix',''))

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mm_projector=torch.nn.Linear(3,3,bias=False)
            with torch.no_grad():
                self.mm_projector.weight.copy_(torch.eye(3))
            self.mm_projector.requires_grad_(False)
            self.config=types.SimpleNamespace(image_aspect_ratio='pad',hidden_size=3)
            self.tower=types.SimpleNamespace(select_feature='patch',num_patches=16,
                                            config=types.SimpleNamespace(image_size=32))
            self.encodes=0
        def get_model(self):
            return self
        def get_vision_tower(self):
            return self.tower
        def get_input_embeddings(self):
            return self.mm_projector
        def encode_images(self,pixels):
            self.encodes+=1
            features=torch.nn.functional.adaptive_avg_pool2d(pixels,(4,4)).flatten(2).transpose(1,2)
            return self.mm_projector(features)

    class Probe:
        deterministic_image_padding=True
        image_processor=types.SimpleNamespace(image_mean=[0.,0.,0.])
        def __init__(self):
            self.model=Model().eval()
            self.prompts=[]
        def _inputs(self,image,prompt):
            if isinstance(image,(str,Path)):
                image=Image.open(image)
            square=Image.new('RGB',(max(image.size),)*2)
            square.paste(image,((square.width-image.width)//2,(square.height-image.height)//2))
            array=np.asarray(square.resize((32,32)),dtype=np.float32).copy()
            return {'images':torch.from_numpy(array).permute(2,0,1)[None]}
        def new_answer_session(self,image,prompt):
            self.prompts.append((str(image),prompt))
            def propose(prefix,count,length):
                self.model.encode_images(self._inputs(image,prompt)['images'])
                return [types.SimpleNamespace(tokens=(2,1),text='stable')]
            return types.SimpleNamespace(propose=propose)
        def tensor_packet(self,items,image,**kw):
            return types.SimpleNamespace(regions=mask(),sources=[('expert','E0')],labels=['structure'])

    path=tmp_path/'image.png'
    Image.fromarray(np.arange(80*80*3,dtype=np.uint8).reshape(80,80,3)).save(path)
    item=Item('E0','expert','segmentation','native',{})
    hist={'generation_config':{},'evidence':[dataclasses.asdict(item)],
          'trace':[{'event':'decode','evidence_transport':transport([item],'q')} ]}
    probe=Probe()
    with torch.inference_mode():
        result=reencoding_case(probe,{'id':'x','image':str(path),'question':'q'},hist,{'config':{}})
    assert result['applicable']
    assert set(result['arms'])==set(ARMS)
    assert probe.model.encodes==11  # seven original prefills + four local encodes
    assert len(probe.prompts)==7 and {p[0] for p in probe.prompts}=={str(path)}
    assert {p[1] for p in probe.prompts}=={'q uniform'}
    assert result['identity_audit']['max_abs_feature_delta']==0
    for name in VIEW_ARMS:
        assert result['arms'][name]['extra_vision_encodes']==1
        assert result['arms'][name]['per_token_auxiliary_branches']==0
        assert result['arms'][name]['hook_audit']['projector_calls']==1
    assert not probe.model.mm_projector._forward_hooks


def test_train_split_checked_before_legacy_normalization(monkeypatch,tmp_path):
    import json
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    from run_region_reencoding import train_rows
    path=tmp_path/'manifest.jsonl'
    path.write_text(json.dumps({'id':'x','official_split':'train'})+'\n')
    expected=[{'id':'x','task':'open_vqa'}]
    assert train_rows(path,lambda _:expected)==expected
    for split in ('test','validate',None):
        path.write_text(json.dumps({'id':'x','official_split':split})+'\n')
        with pytest.raises(ValueError,match='TRAIN'):
            train_rows(path,lambda _:expected)
    path.write_text(json.dumps({'id':'x','official_split':'train','mask':'target'})+'\n')
    with pytest.raises(ValueError,match='target'):
        train_rows(path,lambda _:expected)


def test_report_cannot_use_vqa_protocol(monkeypatch,tmp_path):
    import json
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    from run_region_reencoding import train_rows
    path=tmp_path/'manifest.jsonl'
    path.write_text(json.dumps({'id':'x','official_split':'train'})+'\n')
    with pytest.raises(ValueError,match='VQA-only'):
        train_rows(path,lambda _:[{'id':'x','task':'report_generation'}])
