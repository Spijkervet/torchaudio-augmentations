import numpy as np
import torch


def generate_waveform(
    sample_rate: int,
    num_samples: int,
    num_channels: int,
    frequency: int = 440,
) -> torch.Tensor:

    # Dividing x legnth value into three parts:- 1/10, 1/2, 4/10.
    attack_length = num_samples // 10
    decay_length = num_samples // 2
    sustain_length = num_samples - (attack_length + decay_length)
    sustain_value = 0.1  # Release amplitude between 0 and  1

    # Setting array size and length.
    attack = np.linspace(0, 1, num=attack_length)
    decay = np.linspace(1, sustain_value, num=decay_length)
    sustain = np.ones(sustain_length) * sustain_value
    attack_decay_sustain = np.concatenate((attack, decay, sustain))

    wavedata = np.sin(2 * np.pi * np.arange(num_samples) * frequency / sample_rate)

    wavedata = wavedata * attack_decay_sustain

    if num_channels == 2:
        wavedata = np.array([wavedata, wavedata * 0.9])
    return torch.from_numpy(wavedata.astype(np.float32)).reshape(num_channels, -1)
