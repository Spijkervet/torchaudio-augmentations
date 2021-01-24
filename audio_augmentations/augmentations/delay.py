import random
import numpy as np
import torch

class Delay:
    def __init__(self, sr, p=0.5):
        self.sr = sr
        self.p = p
        self.factor = 0.2  # volume factor of delay signal

    def calc_offset(self, ms):
        return int(ms * (self.sr / 1000))

    def __call__(self, audio):
        if random.random() < self.p:
            # delay between 200 - 500ms with 50ms intervals
            mss = np.arange(200, 500, 50)
            ms = random.choice(mss)

            # calculate delay
            offset = self.calc_offset(ms)
            beginning = torch.zeros(audio.shape[0], offset)
            end = audio[:, :-offset]
            delayed_signal = torch.cat((beginning, end), dim=1)
            delayed_signal = delayed_signal * self.factor
            audio = (audio + delayed_signal) / 2
        return audio
