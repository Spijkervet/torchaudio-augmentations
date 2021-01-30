import torch
import torchaudio
import numpy as np
from audio_augmentations import *

sr = 22050


def random_waveform(num_samples, sr):
    freq = 440
    sine = np.sin(2 * np.pi * np.arange(num_samples) * freq / sr).astype(np.float32)
    return torch.from_numpy(sine).reshape(1, -1)


def test_polarity():
    num_samples = sr * 5
    audio = random_waveform(num_samples, sr)
    transform = Compose(
        [PolarityInversion(p=1)],
    )

    audios = transform(audio)
    assert (audios == torch.neg(audio)).all()

    assert audios.shape[1] == audio.shape[1]


def test_filter():
    audio, sr = torchaudio.load("tests/classical.00002.wav")
    num_samples = sr * 5
    transform = Compose(
        [HighLowPass(sr=sr, p=1)],
    )
    audios = transform(audio)
    torchaudio.save("tests/filter.wav", audios, sample_rate=sr)
    assert audios.shape[1] == audio.shape[1]


def test_delay():
    num_samples = sr * 5
    audio = random_waveform(num_samples, sr)
    transform = Compose(
        [Delay(sr, p=1)],
    )

    audios = transform(audio)
    torchaudio.save("tests/delay.wav", audios, sample_rate=sr)
    assert audios.shape[1] == audio.shape[1]

def test_gain():
    num_samples = sr * 5
    audio = random_waveform(num_samples, sr)
    transform = Compose(
        [Gain(p=1)],
    )

    audios = transform(audio)

    torchaudio.save("tests/gain.wav", audios, sample_rate=sr)
    assert audios.shape[1] == audio.shape[1]


def test_noise():
    audio, sr = torchaudio.load("tests/classical.00002.wav")
    num_samples = sr * 5
    transform = Compose(
        [Noise(snr=1, p=1)],
    )

    audios = transform(audio)

    torchaudio.save("tests/noise.wav", audios, sample_rate=sr)
    assert audios.shape[1] == audio.shape[1]


def test_pitch():
    num_samples = sr * 5
    audio = random_waveform(num_samples, sr)
    transform = Compose(
        [PitchShift(audio_length=num_samples, sr=sr, p=1)],
    )

    audios = transform(audio)

    torchaudio.save("tests/pitch.wav", audios, sample_rate=sr)
    assert audios.shape[1] == audio.shape[1]


def test_reverb():
    num_samples = sr * 5
    audio = random_waveform(num_samples, sr)
    transform = Compose(
        [Reverb(sr, p=1)],
    )

    audios = transform(audio)

    torchaudio.save("tests/reverb.wav", audios, sample_rate=sr)
    assert audios.shape[1] == audio.shape[1]

def test_reverse():
    audio, sr = torchaudio.load("tests/classical.00002.wav")
    num_samples = sr * 5
    transform = Compose(
        [Reverse(p=1)],
    )

    audios = transform(audio)
    torchaudio.save("tests/reverse.wav", audios, sample_rate=sr)
    assert audios.shape[1] == audio.shape[1]
