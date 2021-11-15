import random
import torch


class Reverse(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, audio):
        return torch.flip(audio, dims=[-1])
