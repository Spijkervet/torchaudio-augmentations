import torch
import torchaudio
import numpy as np
from torchaudio_augmentations import (
    Compose,
    RandomResizedCrop,
    RandomApply,
    PolarityInversion,
    Noise,
    Gain,
    HighLowPass,
    Delay,
    PitchShift,
    Reverb,
)

sr = 22050


def sine(num_samples, sr):
    freq = 440
    sine = np.sin(2 * np.pi * np.arange(num_samples) * freq / sr).astype(np.float32)
    return torch.from_numpy(sine).reshape(1, -1)


def test_readme_example():
    num_samples = sr * 5
    audio = sine(num_samples, sr)
    transforms = [
        RandomResizedCrop(n_samples=num_samples),
        RandomApply([PolarityInversion()], p=0.8),
        RandomApply([Noise(min_snr=0.3, max_snr=0.5)], p=0.3),
        RandomApply([Gain()], p=0.2),
        RandomApply([HighLowPass(sample_rate=sr)], p=0.8),
        RandomApply([Delay(sample_rate=sr)], p=0.5),
        RandomApply([PitchShift(n_samples=num_samples, sample_rate=sr)], p=0.4),
        RandomApply([Reverb(sample_rate=sr)], p=0.3),
    ]

    transform = Compose(transforms=transforms)
    transformed_audio = transform(audio)
    assert transformed_audio.shape[0] == 1
