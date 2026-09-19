import copy
import importlib.util
from pathlib import Path

import numpy as np

from merit_feddg.spatial_evidence import encode_soft_mask

spec = importlib.util.spec_from_file_location('probe',Path(__file__).parents[1]/'scripts/observation_blind_probe.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def packet():
    mask = np.zeros((20,30), dtype=np.float32)
    mask[4:8,6:10] = 1
    return [{'expert_id':'biomedparse_objects','evidence_id':'native',
        'payload':{'structures':[{'label':'left lung','mask_coordinate_system':'original_image',
                                  'soft_mask':encode_soft_mask(mask)}]}}]


def test_delivery_and_inheritance():
    evidence = packet()
    old = copy.deepcopy(evidence)
    a = probe.observation('What color is the lung?', evidence,(30,20))
    assert a['available'] and a['entries'][0]['existence_confirmed'] is False
    assert evidence == old
    assert a == probe.observation('What color is the lung?',old,(30,20))


def test_unrelated_and_laterality():
    assert not probe.observation('Where is the liver?',packet(),(30,20))['available']
    assert not probe.observation('Where is the right lung?',packet(),(30,20))['available']
    assert probe.observation('Where is the left lung?',packet(),(30,20))['available']


def test_no_fake_roi():
    data=packet()
    data[0]['payload']['structures'][0]['soft_mask']=encode_soft_mask(np.zeros((20,30)))
    assert not probe.observation('lung',data,(30,20))['available']


def test_full_queue_keeps_unavailable_and_language_denominator(tmp_path, monkeypatch):
    base, out = tmp_path/'base', tmp_path/'new'
    monkeypatch.setattr(probe, 'BASE', base)
    monkeypatch.setattr(probe, 'OUT', out)
    monkeypatch.setattr(probe, 'FULL_QUEUE', True)
    image = tmp_path/'synthetic.png'
    probe.Image.new('RGB', (30,20)).save(image)
    rows = [dict(id=str(i), image=str(image), question=q, q_lang=lang)
            for i,(q,lang) in enumerate([('lung','en'),('liver','en'),('lung','zh')])]
    probe.write(base/'protocol.json', {'identity':'synthetic', 'rows':rows})
    probe.write(base/'complete.json', {'identity':'synthetic'})
    for row in rows:
        probe.write(base/'cases'/(row['id']+'.json'),
                    {'identity':'synthetic','complete':True,'compact_raw':{'evidence':packet()}})
    probe.prepare()
    result = probe.read(out/'protocol.json')
    assert [r['id'] for r in result['rows']] == ['0','1','2']
    assert [r['observation']['available'] for r in result['rows']] == [True,False,False]
    assert result['rows'][2]['observation']['reason'] == 'unsupported_language'
    assert result['inference_reads_references'] is False
