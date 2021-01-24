import numpy as np
from audio_augmentations import Compose, ComposeMany
from audio_augmentations import RandomResizedCrop


def test_compose_many():
    num_augmented_samples = 10

    num_samples = 22050 * 5
    audio = np.arange(num_samples)

    transform = ComposeMany(
        [
            RandomResizedCrop(num_samples),
        ],
        num_augmented_samples=num_augmented_samples,
    )

    audios = transform(audio)
    assert audios.shape[0] == num_augmented_samples


def test_random_resized_crop():
    num_samples = 22050 * 5
    audio = np.arange(num_samples)

    transform = Compose([RandomResizedCrop(num_samples)])

    audio = transform(audio)
    assert audio.shape[1] == num_samples
