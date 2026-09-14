import pytest

from merit_feddg.experts.native_chexagent import (
    _legacy_generation_cache_mode,
    _legacy_transformers_cache_api,
)
from merit_feddg.llava_generalist import _install_generation_kwarg_compat


def test_generation_cache_override_restored_on_exception():
    class Base:
        @classmethod
        def _supports_default_dynamic_cache(cls):
            return True
    class Model(Base):
        pass
    with pytest.raises(ValueError), _legacy_generation_cache_mode(Model()):
        assert not Model._supports_default_dynamic_cache()
        raise ValueError('test cleanup')
    assert Model._supports_default_dynamic_cache()
    assert '_supports_default_dynamic_cache' not in Model.__dict__


def test_legacy_cache_names_are_scoped():
    transformers = pytest.importorskip('transformers')
    del transformers
    from transformers.cache_utils import Cache
    names=('seen_tokens','get_max_length','get_usable_length')
    before={n:hasattr(Cache,n) for n in names}
    with _legacy_transformers_cache_api():
        assert all(hasattr(Cache,n) for n in names)
    assert {n:hasattr(Cache,n) for n in names}==before


def test_llava_old_forward_keeps_inputs_and_drops_only_cache_position():
    class Model:
        def forward(self, input_ids, images=None):
            return input_ids, images
    model=Model()
    assert _install_generation_kwarg_compat(model)=='drop_redundant_cache_position'
    assert model.forward([1],images='original',cache_position=[0])==([1],'original')
