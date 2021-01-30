import random
import numpy as np
import torch


class Delay:
    def __init__(self, sr, min_delay=200, max_delay=500, delay_interval=50, p=0.5):
        self.sr = sr
        self.p = p
        self.factor = 0.2  # volume factor of delay signal
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.delay_interval = delay_interval

    def calc_offset(self, ms):
        return int(ms * (self.sr / 1000))

    def __call__(self, audio):
        if random.random() < self.p:
            ms = random.choice(
                np.arange(self.min_delay, self.max_delay, self.delay_interval)
            )

            offset = self.calc_offset(ms)
            beginning = torch.zeros(audio.shape[0], offset)
            end = audio[:, :-offset]
            delayed_signal = torch.cat((beginning, end), dim=1)
            delayed_signal = delayed_signal * self.factor
            audio = (audio + delayed_signal) / 2
        return audio
