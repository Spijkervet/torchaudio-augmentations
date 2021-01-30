import random
from torchaudio.transforms import Vol


class Gain:
    def __init__(self, min_gain=-20, max_gain=-1, p=0.5):
        self.p = p
        self.min_gain = min_gain
        self.max_gain = max_gain

    def __call__(self, audio):
        if random.random() < self.p:
            gain = random.randint(self.min_gain, self.max_gain)  # input was normalized to max(x)
            audio = Vol(gain, gain_type="db")(audio)  # takes Tensor
        return audio
