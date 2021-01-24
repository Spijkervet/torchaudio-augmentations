import random
import torch
from torchaudio.transforms import Vol


class Gain:
    def __init__(self, sr, p=0.5):
        self.p = p

    def __call__(self, audio):
        if random.random() < self.p:
            gain = random.randint(-20, -1)  # input was normalized to max(x)
            audio = torch.from_numpy(audio)
            audio = Vol(gain, gain_type="db")(audio)  # takes Tensor
            audio = audio.numpy()
        return audio
