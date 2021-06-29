import augment
import torch


class Reverb(torch.nn.Module):
    def __init__(
        self,
        sample_rate,
        reverberance_min=0,
        reverberance_max=100,
        dumping_factor_min=0,
        dumping_factor_max=100,
        room_size_min=0,
        room_size_max=100,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.reverberance_min = reverberance_min
        self.reverberance_max = reverberance_max
        self.dumping_factor_min = dumping_factor_min
        self.dumping_factor_max = dumping_factor_max
        self.room_size_min = room_size_min
        self.room_size_max = room_size_max
        self.src_info = {"rate": self.sample_rate}
        self.target_info = {
            "channels": 1,
            "rate": self.sample_rate,
        }

    def forward(self, audio):
        reverberance = torch.randint(
            self.reverberance_min, self.reverberance_max, size=(1,)
        ).item()
        dumping_factor = torch.randint(
            self.dumping_factor_min, self.dumping_factor_max, size=(1,)
        ).item()
        room_size = torch.randint(
            self.room_size_min, self.room_size_max, size=(1,)
        ).item()

        num_channels = audio.shape[0]
        effect_chain = (
            augment.EffectChain()
            .reverb(reverberance, dumping_factor, room_size)
            .channels(num_channels)
        )

        audio = effect_chain.apply(
            audio, src_info=self.src_info, target_info=self.target_info
        )

        return audio
