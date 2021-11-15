import pytest
import torch
from torchaudio_augmentations.utils import (
    add_audio_batch_dimension,
    remove_audio_batch_dimension,
    tensor_has_valid_audio_batch_dimension,
)


@pytest.mark.parametrize(
    "tensor,expected_value",
    [
        (torch.zeros(1), False),
        (torch.zeros(1, 48000), False),
        (torch.zeros(16, 48000), False),
        (torch.zeros(1, 1, 48000), True),
        (torch.zeros(16, 1, 48000), True),
    ],
)
def test_tensor_has_valid_audio_batch_dimension(tensor, expected_value):

    assert tensor_has_valid_audio_batch_dimension(tensor) == expected_value


def test_add_audio_batch_dimension():
    tensor = torch.ones(1, 48000)
    expected_tensor = torch.ones(1, 1, 48000)

    tensor = add_audio_batch_dimension(tensor)
    assert torch.eq(tensor, expected_tensor).all()
    assert tensor_has_valid_audio_batch_dimension(tensor) == True

    tensor = torch.ones(48000)
    expected_tensor = torch.ones(1, 48000)

    tensor = add_audio_batch_dimension(tensor)
    assert torch.eq(tensor, expected_tensor).all()
    assert tensor_has_valid_audio_batch_dimension(tensor) == False


def test_remove_audio_batch_dimension():
    tensor = torch.ones(1, 1, 48000)
    expected_tensor = torch.ones(1, 48000)

    tensor = remove_audio_batch_dimension(tensor)
    assert torch.eq(tensor, expected_tensor).all()

    tensor = torch.ones(1, 48000)
    expected_tensor = torch.ones(48000)

    tensor = remove_audio_batch_dimension(tensor)
    assert torch.eq(tensor, expected_tensor).all()
    assert tensor_has_valid_audio_batch_dimension(tensor) == False
