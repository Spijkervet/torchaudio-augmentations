import numpy as np
import torch
import torchaudio
import pytest

from torchaudio_augmentations import (
    Compose,
    RandomResizedCrop,
    PolarityInversion,
    Noise,
    Gain,
    HighLowPass,
    Delay,
    PitchShift,
    Reverb,
    Reverse,
)
from .utils import generate_waveform


sample_rate = 22050
num_samples = sample_rate * 5


@pytest.mark.parametrize("num_channels", [1, 2])
def test_random_resized_crop(num_channels):
    num_samples = 22050 * 5
    audio = generate_waveform(sample_rate, num_samples, num_channels)
    transform = Compose([RandomResizedCrop(num_samples)])

    audio = transform(audio)
    assert audio.shape[0] == num_channels
    assert audio.shape[1] == num_samples


@pytest.mark.parametrize("num_channels", [1, 2])
def test_polarity(num_channels):
    audio = generate_waveform(sample_rate, num_samples,
                              num_channels=num_channels)
    transform = Compose([PolarityInversion()],)

    t_audio = transform(audio)
    assert (t_audio == torch.neg(audio)).all()
    assert t_audio.shape == audio.shape


@pytest.mark.parametrize("num_channels", [1, 2])
def test_filter(num_channels):
    audio = generate_waveform(sample_rate, num_samples, num_channels)
    transform = Compose([HighLowPass(sample_rate=sample_rate)],)
    t_audio = transform(audio)
    # torchaudio.save("tests/filter.wav", t_audio, sample_rate=sample_rate)
    assert t_audio.shape == audio.shape


@pytest.mark.parametrize("num_channels", [1, 2])
def test_delay(num_channels):
    audio = generate_waveform(sample_rate, num_samples, num_channels)
    transform = Compose([Delay(sample_rate=sample_rate)],)

    t_audio = transform(audio)
    # torchaudio.save("tests/delay.wav", t_audio, sample_rate=sample_rate)
    assert t_audio.shape == audio.shape


@pytest.mark.parametrize("num_channels", [1, 2])
def test_gain(num_channels):
    audio = generate_waveform(sample_rate, num_samples, num_channels)
    transform = Compose([Gain()],)

    t_audio = transform(audio)
    # torchaudio.save("tests/gain.wav", t_audio, sample_rate=sample_rate)
    assert t_audio.shape == audio.shape


@pytest.mark.parametrize("num_channels", [1, 2])
def test_noise(num_channels):
    audio = generate_waveform(sample_rate, num_samples, num_channels)
    transform = Compose([Noise(min_snr=0.5, max_snr=1)],)

    t_audio = transform(audio)
    # torchaudio.save("tests/noise.wav", t_audio, sample_rate=sample_rate)
    assert t_audio.shape == audio.shape


@pytest.mark.parametrize("num_channels", [1, 2])
def test_pitch(num_channels):
    audio = generate_waveform(sample_rate, num_samples, num_channels)
    transform = Compose(
        [PitchShift(n_samples=num_samples, sample_rate=sample_rate)],)

    t_audio = transform(audio)
    # torchaudio.save("tests/pitch.wav", t_audio, sample_rate=sample_rate)
    assert t_audio.shape == audio.shape


@pytest.mark.parametrize("num_channels", [1, 2])
def test_reverb(num_channels):
    audio = generate_waveform(sample_rate, num_samples, num_channels)
    transform = Compose([Reverb(sample_rate=sample_rate)],)

    t_audio = transform(audio)
    # torchaudio.save("tests/reverb.wav", t_audio, sample_rate=sample_rate)
    assert t_audio.shape == audio.shape


@pytest.mark.parametrize("num_channels", [1, 2])
def test_reverse(num_channels):
    stereo_audio = generate_waveform(sample_rate, num_samples, num_channels)
    transform = Compose([Reverse()],)

    t_audio = transform(stereo_audio)
    # torchaudio.save("tests/reverse.wav", t_audio, sample_rate=sample_rate)

    reversed_single_channel = torch.flip(t_audio, [1])[0]
    assert torch.equal(reversed_single_channel, stereo_audio[0]) == True

    reversed_stereo_channel = torch.flip(t_audio, [0])[0]
    assert torch.equal(reversed_stereo_channel, stereo_audio[0]) == False

    assert t_audio.shape == stereo_audio.shape

    mono_audio = stereo_audio.mean(dim=0)
    assert mono_audio.shape[0] == stereo_audio.shape[1]
