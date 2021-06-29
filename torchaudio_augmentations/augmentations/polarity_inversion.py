import torch
import random


class PolarityInversion(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, audio):
        audio = torch.neg(audio)
        return audio
