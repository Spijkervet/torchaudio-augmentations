import random
import torch
from julius.filters import highpass_filter, lowpass_filter


class FrequencyFilter(torch.nn.Module):
    def __init__(
        self,
        sample_rate: int,
        freq_low: float,
        freq_high: float,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.freq_low = freq_low
        self.freq_high = freq_high

    def cutoff_frequency(self, frequency: float) -> float:
        return frequency / self.sample_rate

    def sample_uniform_frequency(self):
        return random.uniform(self.freq_low, self.freq_high)


class HighPassFilter(FrequencyFilter):
    def __init__(
        self,
        sample_rate: int,
        freq_low: float = 200,
        freq_high: float = 1200,
    ):
        super().__init__(sample_rate, freq_low, freq_high)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        frequency = self.sample_uniform_frequency()
        cutoff = self.cutoff_frequency(frequency)
        audio = highpass_filter(audio, cutoff=cutoff)
        return audio


class LowPassFilter(FrequencyFilter):
    def __init__(
        self,
        sample_rate: int,
        freq_low: float = 2200,
        freq_high: float = 4000,
    ):
        super().__init__(sample_rate, freq_low, freq_high)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        frequency = self.sample_uniform_frequency()
        cutoff = self.cutoff_frequency(frequency)
        audio = lowpass_filter(audio, cutoff=cutoff)
        return audio
