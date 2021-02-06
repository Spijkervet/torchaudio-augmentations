import torch
import random
from torchaudio.transforms import Vol


class Gain(torch.nn.Module):
    def __init__(self, min_gain=-20, max_gain=-1):
        super().__init__()
        self.min_gain = min_gain
        self.max_gain = max_gain

    def forward(self, audio):
        gain = torch.randint(self.min_gain, self.max_gain, size=(1,)).item()
        audio = Vol(gain, gain_type="db")(audio)
        return audio
