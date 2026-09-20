"""Regression for the observed TEST cache export changing array ordinals."""
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path

import pytest

from merit_feddg.capabilities import EvidenceItem
from merit_feddg.compact_evidence import compact_records, compact_prompt


def test_export_preserves_native_array_ordinals_and_prompt(tmp_path):
    path = Path(__file__).resolve().parents[1]/'scripts/prepare_algorithm_test.py'
    spec = importlib.util.spec_from_file_location('prepare_algorithm_test',path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    array = dict(encoding='float32-zlib-base64',shape=[1],data='opaque-native-audit')
    item = EvidenceItem('packet','expert','segmentation','image',dict(z_mask=array,a_mask=array))
    original = dict(evidence=[asdict(item)])
    def render(value):
        return compact_prompt('Question',compact_records([EvidenceItem(**x) for x in value['evidence']]),columns=False)
    expected = render(original)
    # Generic canonical JSON is semantically equal, but changes array ordinals.
    reordered = json.loads(json.dumps(original,sort_keys=True))
    assert reordered == original
    assert render(reordered) != expected
    target = tmp_path/'case.json'
    module.write_case(target,original)
    assert render(json.loads(target.read_text())) == expected
    with pytest.raises(FileExistsError):
        module.write_case(target,reordered)
    assert render(json.loads(target.read_text())) == expected
