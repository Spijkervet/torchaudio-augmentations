import random
import numpy as np
import torch


class Noise:
    def __init__(self, snr=1, p=0.8):
        self.snr = snr
        self.p = p

    def __call__(self, audio):
        if random.random() < self.p:
            audio = audio.numpy()
            RMS_s = np.sqrt(np.mean(audio ** 2))
            RMS_n = np.sqrt(RMS_s ** 2 / (pow(10, self.snr / 20)))
            noise = np.random.normal(0, RMS_n, audio.shape[1]).astype("float32")
            audio = audio + noise
            audio = np.clip(audio, -1, 1)
            audio = torch.from_numpy(audio)
        return audio
