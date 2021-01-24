import random
import numpy as np

class Noise:
    def __init__(self, p=0.8):
        self.p = p
        self.snr = 80

    def __call__(self, audio):
        if random.random() < self.p:
            RMS_s = np.sqrt(np.mean(audio ** 2))
            RMS_n = np.sqrt(RMS_s ** 2 / (pow(10, self.snr / 20)))
            noise = np.random.normal(0, RMS_n, audio.shape[0]).astype("float32")
            audio = audio + noise
            audio = np.clip(audio, -1, 1)
        return audio

