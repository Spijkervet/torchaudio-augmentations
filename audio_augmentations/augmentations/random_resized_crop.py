import random

class RandomResizedCrop:
    def __init__(self, n_samples):
        self.n_samples = n_samples

    def __call__(self, audio):
        max_samples = audio.shape[0]
        start_idx = random.randint(0, max_samples - self.n_samples)  # * 2))
        audio = audio[start_idx : start_idx + self.n_samples]
        return audio
