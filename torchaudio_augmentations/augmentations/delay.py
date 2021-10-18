import random
import numpy as np
import torch


class Delay(torch.nn.Module):
    def __init__(
        self,
        sample_rate,
        volume_factor=0.5,
        min_delay=200,
        max_delay=500,
        delay_interval=50,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.volume_factor = volume_factor
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.delay_interval = delay_interval

    def calc_offset(self, ms):
        return int(ms * (self.sample_rate / 1000))

    def forward(self, audio):
        ms = random.choice(
            np.arange(self.min_delay, self.max_delay, self.delay_interval)
        )

        offset = self.calc_offset(ms)
        beginning = torch.zeros(audio.shape[0], offset).to(audio.device)
        end = audio[:, :-offset]
        delayed_signal = torch.cat((beginning, end), dim=1)
        delayed_signal = delayed_signal * self.volume_factor
        audio = (audio + delayed_signal) / 2
        return audio
