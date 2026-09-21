from types import SimpleNamespace

import numpy as np
import pytest

from merit_feddg.experts.plip import PlipConceptExpert


def test_overlength_transaction_is_rejected_before_model_scoring():
    expert = PlipConceptExpert.__new__(PlipConceptExpert)
    options = {}

    def processor(**kwargs):
        options.update(kwargs)
        return {"input_ids": np.zeros((2, 106), dtype=int)}

    expert.processor = processor
    expert.model = SimpleNamespace(
        config=SimpleNamespace(text_config=SimpleNamespace(max_position_embeddings=77))
    )
    with pytest.raises(ValueError, match="refusing to truncate transaction evidence"):
        expert._text_embeddings(["incumbent", "candidate"])
    assert options["truncation"] is False
