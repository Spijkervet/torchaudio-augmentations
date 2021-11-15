import torch


def tensor_has_valid_audio_batch_dimension(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 3:
        return True
    return False


def add_audio_batch_dimension(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.unsqueeze(dim=0)


def remove_audio_batch_dimension(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.squeeze(dim=0)
