import random
import torch


class PolarityInversion:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, audio):
        if random.random() < self.p:
            audio = torch.neg(audio)
        return audio
