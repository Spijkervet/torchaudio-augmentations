import numpy as np
import torch
import torchaudio
import unittest

from torchaudio_augmentations import *


class TestShapes(unittest.TestCase):

    def setUp(self):
        self.sr = 22050
        self.num_samples = self.sr * 5

    def stereo_waveform(self, num_samples):

        # Dividing x legnth value into three parts:- 1/10, 1/2, 4/10.
        attack_length = num_samples // 10
        decay_length = num_samples // 2
        sustain_length = num_samples - (attack_length + decay_length)
        sustain_value = 0.1   # Release amplitude between 0 and  1

        # Setting array size and length.
        attack = np.linspace(0, 1, num=attack_length)
        decay = np.linspace(1, sustain_value, num=decay_length)
        sustain = np.ones(sustain_length) * sustain_value
        attack_decay_sustain = np.concatenate((attack, decay, sustain))

        freq = 440
        wavedata = np.sin(2 * np.pi * np.arange(num_samples)
                          * freq / self.sr)

        wavedata = wavedata * attack_decay_sustain
        wavedata = np.array([wavedata, wavedata * 0.9]).astype(np.float32)
        return torch.from_numpy(wavedata).reshape(2, -1)

    def test_polarity(self):
        audio = self.stereo_waveform(self.num_samples)
        transform = Compose(
            [PolarityInversion()],
        )

        t_audio = transform(audio)
        assert (t_audio == torch.neg(audio)).all()

        assert t_audio.shape == audio.shape

    def test_filter(self):
        audio = self.stereo_waveform(self.num_samples)
        transform = Compose(
            [HighLowPass(sample_rate=self.sr)],
        )
        t_audio = transform(audio)
        # torchaudio.save("tests/filter.wav", t_audio, sample_rate=self.sr)
        assert t_audio.shape == audio.shape

    def test_delay(self):
        audio = self.stereo_waveform(self.num_samples)
        transform = Compose(
            [Delay(self.sr)],
        )

        t_audio = transform(audio)
        # torchaudio.save("tests/delay.wav", t_audio, sample_rate=self.sr)
        assert t_audio.shape == audio.shape

    def test_gain(self):
        audio = self.stereo_waveform(self.num_samples)
        transform = Compose(
            [Gain()],
        )

        t_audio = transform(audio)
        # torchaudio.save("tests/gain.wav", t_audio, sample_rate=self.sr)
        assert t_audio.shape == audio.shape

    def test_noise(self):
        audio = self.stereo_waveform(self.num_samples)
        transform = Compose(
            [Noise(min_snr=0.5, max_snr=1)],
        )

        t_audio = transform(audio)
        # torchaudio.save("tests/noise.wav", t_audio, sample_rate=self.sr)
        assert t_audio.shape == audio.shape

    def test_pitch(self):
        audio = self.stereo_waveform(self.num_samples)
        transform = Compose(
            [PitchShift(n_samples=self.num_samples, sample_rate=self.sr)],
        )

        t_audio = transform(audio)
        # torchaudio.save("tests/pitch.wav", t_audio, sample_rate=sr)
        assert t_audio.shape == audio.shape

    def test_reverb(self):
        audio = self.stereo_waveform(self.num_samples)
        transform = Compose(
            [Reverb(self.sr)],
        )

        t_audio = transform(audio)
        # torchaudio.save("tests/reverb.wav", t_audio, sample_rate=sr)
        assert t_audio.shape == audio.shape

    def test_reverse(self):
        stereo_audio = self.stereo_waveform(self.num_samples)
        transform = Compose(
            [Reverse()],
        )

        t_audio = transform(stereo_audio)
        # torchaudio.save("tests/reverse.wav", t_audio, sample_rate=self.sr)

        reversed_single_channel = torch.flip(t_audio, [1])[0]
        assert torch.equal(reversed_single_channel, stereo_audio[0]) == True

        reversed_stereo_channel = torch.flip(t_audio, [0])[0]
        assert torch.equal(reversed_stereo_channel, stereo_audio[0]) == False

        assert t_audio.shape == stereo_audio.shape

        mono_audio = stereo_audio.mean(dim=0)
        assert mono_audio.shape[0] == stereo_audio.shape[1]
