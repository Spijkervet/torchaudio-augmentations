import random
import torch
from fractions import Fraction
from typing import Optional
from torch_pitch_shift import get_fast_shifts, pitch_shift, semitones_to_ratio
from torchaudio_augmentations.utils import (
    add_audio_batch_dimension,
    remove_audio_batch_dimension,
    tensor_has_valid_audio_batch_dimension,
)


class PitchShift(torch.nn.Module):
    def __init__(
        self,
        n_samples,
        sample_rate,
        pitch_shift_min: int = -7.0,
        pitch_shift_max: int = 7.0,
        bins_per_octave: Optional[int] = 12,
    ):
        self.n_samples = n_samples
        self.sample_rate = sample_rate
        self.pitch_shift_min = pitch_shift_min
        self.pitch_shift_max = pitch_shift_max
        self.bins_per_octave = bins_per_octave

        self._fast_shifts = get_fast_shifts(
            sample_rate,
            lambda x: x >= semitones_to_ratio(self.pitch_shift_min)
            and x <= semitones_to_ratio(self.pitch_shift_max)
            and x != 1,
        )

        if len(self._fast_shifts) == 0:
            raise ValueError(
                f"Could not compute any fast pitch-shift ratios for the given sample rate and pitch shift range: {self.pitch_shift_min} - {self.pitch_shift_max} (semitones)"
            )

    @property
    def fast_shifts(self):
        return self._fast_shifts

    def draw_sample_uniform_from_fast_shifts(self) -> Fraction:
        return random.choice(self.fast_shifts)

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        is_batched = False
        if not tensor_has_valid_audio_batch_dimension(audio):
            audio = add_audio_batch_dimension(audio)
            is_batched = True

        fast_shift = self.draw_sample_uniform_from_fast_shifts()
        y = pitch_shift(
            input=audio,
            shift=fast_shift,
            sample_rate=self.sample_rate,
            bins_per_octave=self.bins_per_octave,
        )

        if is_batched:
            y = remove_audio_batch_dimension(y)
        return y
