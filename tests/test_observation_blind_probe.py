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
