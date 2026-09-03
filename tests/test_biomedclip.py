import numpy as np
import pytest

from merit_feddg.experts.biomedclip import MODALITY_PROMPTS, route_probabilities


def test_small_router_covers_every_configured_modality():
    assert {"cxr", "pathology", "oct"} <= set(MODALITY_PROMPTS)


def test_router_probabilities_are_normalized_and_ordered():
    route = route_probabilities(np.asarray([1.0, 4.0, 2.0]), ["cxr", "pathology", "oct"])
    assert sum(route.values()) == pytest.approx(1.0)
    assert max(route, key=route.get) == "pathology"
