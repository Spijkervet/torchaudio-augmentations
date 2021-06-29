import random
import torch
from julius.filters import highpass_filter
from julius.lowpass import lowpass_filter


class HighLowPass(torch.nn.Module):
    def __init__(
        self,
        sample_rate: int,
        lowpass_freq_low: int = 2200,
        lowpass_freq_high: int = 4000,
        highpass_freq_low: int = 200,
        highpass_freq_high: int = 1200,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.lowpass_freq_low = lowpass_freq_low
        self.lowpass_freq_high = lowpass_freq_high
        self.highpass_freq_low = highpass_freq_low
        self.highpass_freq_high = highpass_freq_high

    def forward(self, audio):
        highlowband = random.randint(0, 1)
        if highlowband == 0:
            highpass_freq = random.randint(
                self.highpass_freq_low, self.highpass_freq_high
            )
            cutoff = highpass_freq / self.sample_rate
            audio = highpass_filter(audio, cutoff=cutoff)
        elif highlowband == 1:
            lowpass_freq = random.randint(self.lowpass_freq_low, self.lowpass_freq_high)
            cutoff = lowpass_freq / self.sample_rate
            audio = lowpass_filter(audio, cutoff=cutoff)

        return audio
