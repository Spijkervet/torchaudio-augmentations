# Audio Augmentations

Audio augmentations library for PyTorch for audio in the time-domain, with support for stochastic data augmentations as used often in self-supervised / contrastive learning.


## Usage
We can define several audio augmentations, which will be applied sequentially to a raw audio waveform:
```
from audio_augmentations import *

audio, sr = torchaudio.load("tests/classical.00002.wav")

num_samples = sr * 5
transforms = [
    RandomResizedCrop(n_samples=num_samples),
    RandomApply([PolarityInversion()], p=0.8),
    RandomApply([Noise(min_snr=0.3, max_snr=0.5)], p=0.3),
    RandomApply([Gain()], p=0.2),
    RandomApply([HighLowPass(sample_rate=sr)], p=0.8),
    RandomApply([Delay(sample_rate=sr)], p=0.5),
    RandomApply([PitchShift(
        n_samples=num_samples,
        sample_rate=sr
    )], p=0.4),
    RandomApply([Reverb(sample_rate=sr)], p=0.3)
]
```

We can return either one or many versions of the same audio example:
```
transform = Compose(transforms=transforms)
transformed_audio =  transform(audio)
>> transformed_audio.shape[0] = 1
```

```
audio = torchaudio.load("testing/classical.00002.wav")
transform = ComposeMany(transforms=transforms, num_augmented_samples=4)
transformed_audio = transform(audio)
>> transformed_audio.shape[0] = 4
```

Similar to the `torchvision.datasets` interface, an instance of the `Compose` or `ComposeMany` class can be supplied to a torchaudio dataloaders that accept `transform=`.


## Optional
Install WavAugment for reverberation / pitch shifting:
```
pip install git+https://github.com/facebookresearch/WavAugment
```