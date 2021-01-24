# Audio Augmentations

Audio augmentations library for PyTorch for audio in the time-domain, with support for stochastic data augmentations as used often in self-supervised / contrastive learning.


## Usage
We can define several audio augmentations, which will be applied sequentially to a raw audio waveform:
```
transforms = [
    RandomResizedCrop(n_samples=audio_length),
    PolarityInversion(p=0.8),
    # Noise(p=0.1),
    Gain(p=0.3),
    HighLowPass(p=0.8, sr=sample_rate),
    Delay(p=0.4, sr=sample_rate),
    PitchShift(
        audio_length=audio_length,
        p=0.6,
        sr=sample_rate,
    )
    Reverb(p=0.6, sr=sample_rate)
]
```

We can return either one or many versions of the same audio example:
```
audio = torchaudio.load("testing/test.wav")
transform = Compose(transforms=transforms)
transformed_audio =  transform(audio)
>> transformed_audio.shape[0] = 1
```

```
audio = torchaudio.load("testing/test.wav")
transform = ComposeMany(transforms=transforms, num_augmented_samples=4)
transformed_audio = transform(audio)
>> transformed_audio.shape[0] = 4
```

Similar to the `torchvision.datasets` interface, an instance of the `Compose` or `ComposeMany` class can be supplied to a torchaudio dataloaders that accept `transform=`.

