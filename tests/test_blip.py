import pytest

torch = pytest.importorskip("torch")

from merit_feddg.experts.blip import _continuation_labels


def test_continuation_labels_align_full_sequence_and_mask_prompt():
    prompt = torch.tensor([[1, 2, 3]])
    continuation = torch.tensor([[4, 5]])

    full, labels = _continuation_labels(prompt, continuation)

    assert full.tolist() == [[1, 2, 3, 4, 5]]
    assert labels.tolist() == [[-100, -100, -100, 4, 5]]
