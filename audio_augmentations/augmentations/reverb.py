import random
import augment
import torch.nn as nn

class Reverb(nn.Module):
    def __init__(self, sr, reverberance_min=0, reverberance_max=100, dumping_factor_min=0, dumping_factor_max=100, room_size_min=0, room_size_max=100, p=0.5):
        self.sr = sr
        self.reverberance_min = reverberance_min
        self.reverberance_max = reverberance_max
        self.dumping_factor_min = dumping_factor_min
        self.dumping_factor_max = dumping_factor_max
        self.room_size_min = room_size_min
        self.room_size_max = room_size_max
        self.p = p
        self.src_info = {"rate": self.sr}
        self.target_info = {
            "channels": 1,
            "rate": self.sr,
        }

    def forward(self, audio):
        self.effect_chain = (
            augment.EffectChain()
            .reverb(self.reverberance, self.dumping_factor, self.room_size)
            .channels(1)
        )

        if random.random() < self.p:
            audio = self.effect_chain.apply(
                audio, src_info=self.src_info, target_info=self.target_info
            )

        return audio
