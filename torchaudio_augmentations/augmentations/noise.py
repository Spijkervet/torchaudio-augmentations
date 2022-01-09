import random
import numpy as np
import torch


class Noise(torch.nn.Module):
    def __init__(self, min_snr=-4, max_snr=-2):
        """
        :param min_snr: Minimum signal-to-noise ratio in dB.
        :param max_snr: Maximum signal-to-noise ratio in dB.
        """
        super().__init__()
        self.min_snr = min_snr
        self.max_snr = max_snr

    def forward(self, audio):
        std = torch.std(audio)
        snr = random.uniform(self.min_snr, self.max_snr)
        noise_std = 10**(snr/20) * std

        noise = np.random.normal(0.0, noise_std, size=audio.shape).astype(np.float32)

        return audio + noise
