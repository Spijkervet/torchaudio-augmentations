import librosa
import torch
import pytest
import numpy as np

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


@pytest.mark.parametrize(
    ["batch_size", "num_channels"],
    [
        (1, 1),
        (4, 1),
        (16, 1),
        (1, 2),
        (4, 2),
        (16, 2),
    ],
)
def test_random_resized_crop_batched(batch_size, num_channels):

    num_samples = 22050 * 5
    audio = generate_waveform(sample_rate, num_samples, num_channels)
    audio = audio.repeat(batch_size, 1, 1)

    transform = Compose([RandomResizedCrop(num_samples)])

    audio = transform(audio)
    assert audio.shape[0] == batch_size
    assert audio.shape[1] == num_channels
    assert audio.shape[2] == num_samples


@pytest.mark.parametrize("num_channels", [1, 2])
def test_polarity(num_channels):
    audio = generate_waveform(sample_rate, num_samples, num_channels=num_channels)
    transform = Compose(
        [PolarityInversion()],
    )

    t_audio = transform(audio)
    assert torch.eq(t_audio, torch.neg(audio)).all()
    assert t_audio.shape == audio.shape

    audio = torch.Tensor([5, 6, -1, 3])
    expected_audio = torch.Tensor([-5, -6, 1, -3])
    t_audio = transform(audio)
    assert torch.eq(expected_audio, t_audio).all()

    mono_channel_audio = torch.Tensor([[5, 6, -1, 3]])
    expected_audio = torch.Tensor([[-5, -6, 1, -3]])
    t_audio = transform(mono_channel_audio)
    assert torch.eq(expected_audio, t_audio).all()

    stereo_channel_audio = mono_channel_audio.repeat(2, 1)
    expected_audio = expected_audio.repeat(2, 1)
    t_audio = transform(stereo_channel_audio)
    assert torch.eq(expected_audio, t_audio).all()

    batched_stereo_channel_audio = stereo_channel_audio.repeat(16, 1, 1)
    expected_audio = expected_audio.repeat(16, 1, 1)
    t_audio = transform(batched_stereo_channel_audio)
    assert torch.eq(expected_audio, t_audio).all()


@pytest.mark.parametrize("num_channels", [1, 2])
def test_filter(num_channels):
    audio = generate_waveform(sample_rate, num_samples, num_channels)
    transform = Compose(
        [HighLowPass(sample_rate=sample_rate)],
    )
    t_audio = transform(audio)
    # torchaudio.save("tests/filter.wav", t_audio, sample_rate=sample_rate)
    assert t_audio.shape == audio.shape


@pytest.mark.parametrize("num_channels", [1, 2])
def test_delay(num_channels):
    audio = generate_waveform(sample_rate, num_samples, num_channels)
    transform = Compose(
        [Delay(sample_rate=sample_rate)],
    )

    t_audio = transform(audio)
    # torchaudio.save("tests/delay.wav", t_audio, sample_rate=sample_rate)
    assert t_audio.shape == audio.shape


@pytest.mark.parametrize("num_channels", [1, 2])
def test_gain(num_channels):
    audio = generate_waveform(sample_rate, num_samples, num_channels)
    transform = Compose(
        [Gain()],
    )

    t_audio = transform(audio)
    # torchaudio.save("tests/gain.wav", t_audio, sample_rate=sample_rate)
    assert t_audio.shape == audio.shape


@pytest.mark.parametrize("num_channels", [1, 2])
def test_noise(num_channels):
    audio = generate_waveform(sample_rate, num_samples, num_channels)
    transform = Compose(
        [Noise(min_snr=0.5, max_snr=1)],
    )

    t_audio = transform(audio)
    # torchaudio.save("tests/noise.wav", t_audio, sample_rate=sample_rate)
    assert t_audio.shape == audio.shape


@pytest.mark.parametrize("num_channels", [1, 2])
def test_pitch(num_channels):
    audio = generate_waveform(sample_rate, num_samples, num_channels)
    transform = Compose(
        [PitchShift(n_samples=num_samples, sample_rate=sample_rate)],
    )

    t_audio = transform(audio)
    # torchaudio.save("tests/pitch.wav", audio, sample_rate=sample_rate)
    # torchaudio.save("tests/t_pitch.wav", t_audio, sample_rate=sample_rate)
    assert t_audio.shape == audio.shape


def test_pitch_shift_fast_ratios():
    ps = PitchShift(
        n_samples=num_samples,
        sample_rate=sample_rate,
        pitch_shift_min=-5,
        pitch_shift_max=5,
    )
    assert len(ps.fast_shifts) == 20


def test_pitch_shift_no_fast_ratios():
    with pytest.raises(ValueError):
        _ = PitchShift(
            n_samples=num_samples,
            sample_rate=sample_rate,
            pitch_shift_min=4,
            pitch_shift_max=4,
        )


def test_pitch_shift_transform_with_pitch_detection():
    """To check semi-tone values, check: http://www.homepages.ucl.ac.uk/~sslyjjt/speech/semitone.html"""

    source_frequency = 440
    max_semitone_shift = 4
    expected_frequency_shift = 554

    num_channels = 1
    audio = generate_waveform(
        sample_rate, num_samples, num_channels, frequency=source_frequency
    )
    pitch_shift = PitchShift(
        n_samples=num_samples,
        sample_rate=sample_rate,
        pitch_shift_min=max_semitone_shift,
        pitch_shift_max=max_semitone_shift + 1,
    )

    t_audio = pitch_shift(audio)
    librosa_audio = t_audio[0].numpy()
    f0_hz, _, _ = librosa.pyin(librosa_audio, fmin=10, fmax=1000)

    # remove nan values:
    f0_hz = f0_hz[~np.isnan(f0_hz)]

    detected_f0_hz = np.max(f0_hz)

    # the detected frequency vs. expected frequency should not be smaller than 20Hz.
    assert abs(detected_f0_hz - expected_frequency_shift) < 20


@pytest.mark.parametrize("num_channels", [1, 2])
def test_reverb(num_channels):
    audio = generate_waveform(sample_rate, num_samples, num_channels)
    transform = Compose(
        [Reverb(sample_rate=sample_rate)],
    )

    t_audio = transform(audio)
    # torchaudio.save("tests/reverb.wav", t_audio, sample_rate=sample_rate)
    assert t_audio.shape == audio.shape


@pytest.mark.parametrize("num_channels", [1, 2])
def test_reverse(num_channels):
    stereo_audio = generate_waveform(sample_rate, num_samples, num_channels)
    transform = Compose(
        [Reverse()],
    )

    t_audio = transform(stereo_audio)
    # torchaudio.save("tests/reverse.wav", t_audio, sample_rate=sample_rate)

    reversed_single_channel = torch.flip(t_audio, [1])[0]
    assert torch.equal(reversed_single_channel, stereo_audio[0]) == True

    reversed_stereo_channel = torch.flip(t_audio, [0])[0]
    assert torch.equal(reversed_stereo_channel, stereo_audio[0]) == False

    assert t_audio.shape == stereo_audio.shape

    mono_audio = stereo_audio.mean(dim=0)
    assert mono_audio.shape[0] == stereo_audio.shape[1]
