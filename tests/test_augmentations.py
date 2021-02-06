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
        [PolarityInversion()],
    )

    audios = transform(audio)
    assert (audios == torch.neg(audio)).all()

    assert audios.shape[1] == audio.shape[1]


def test_filter():
    audio, sr = torchaudio.load("tests/classical.00002.wav")
    num_samples = sr * 5
    transform = Compose(
        [HighLowPass(sr=sr)],
    )
    audios = transform(audio)
    torchaudio.save("tests/filter.wav", audios, sample_rate=sr)
    assert audios.shape[1] == audio.shape[1]


def test_delay():
    audio, sr = torchaudio.load("tests/classical.00002.wav")
    num_samples = sr * 5
    transform = Compose(
        [Delay(sr)],
    )

    audios = transform(audio)
    torchaudio.save("tests/delay.wav", audios, sample_rate=sr)
    assert audios.shape[1] == audio.shape[1]

def test_gain():
    audio, sr = torchaudio.load("tests/classical.00002.wav")
    num_samples = sr * 5
    transform = Compose(
        [Gain()],
    )

    audios = transform(audio)

    torchaudio.save("tests/gain.wav", audios, sample_rate=sr)
    assert audios.shape[1] == audio.shape[1]


def test_noise():
    audio, sr = torchaudio.load("tests/classical.00002.wav")
    num_samples = sr * 5
    transform = Compose(
        [Noise(min_snr=0.5, max_snr=1)],
    )

    audios = transform(audio)

    torchaudio.save("tests/noise.wav", audios, sample_rate=sr)
    assert audios.shape[1] == audio.shape[1]


def test_pitch():
    audio, sr = torchaudio.load("tests/classical.00002.wav")
    num_samples = sr * 5
    transform = Compose(
        [PitchShift(audio_length=num_samples, sr=sr)],
    )

    audios = transform(audio)

    torchaudio.save("tests/pitch.wav", audios, sample_rate=sr)
    assert audios.shape[1] == audio.shape[1]


def test_reverb():
    audio, sr = torchaudio.load("tests/classical.00002.wav")
    num_samples = sr * 5
    transform = Compose(
        [Reverb(sr)],
    )

    audios = transform(audio)

    torchaudio.save("tests/reverb.wav", audios, sample_rate=sr)
    assert audios.shape[1] == audio.shape[1]

def test_reverse():
    audio, sr = torchaudio.load("tests/classical.00002.wav")
    num_samples = sr * 5
    transform = Compose(
        [Reverse()],
    )

    audios = transform(audio)
    torchaudio.save("tests/reverse.wav", audios, sample_rate=sr)
    assert audios.shape[1] == audio.shape[1]
