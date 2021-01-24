import random
import augment

class Reverb:
    def __init__(self, sr, p=0.5):
        self.sr = sr
        self.p = p
        self.reverberance = lambda: random.randint(1, 100) # 0 - 100
        self.dumping_factor = lambda: random.randint(1, 100) # 0 - 100
        self.room_size = lambda: random.randint(1, 100) # 0 - 100
        self.effect_chain = (
            augment.EffectChain().reverb(self.reverberance, self.dumping_factor, self.room_size).channels(1)
        )
        self.src_info = {"rate": self.sr}
        self.target_info = {
            "channels": 1,
            "rate": self.sr,
        }

    def __call__(self, audio):
        if random.random() < self.p:
            audio = torch.from_numpy(audio)
            y = self.effect_chain.apply(
                audio, src_info=self.src_info, target_info=self.target_info
            )
            audio = y.numpy()


        return audio