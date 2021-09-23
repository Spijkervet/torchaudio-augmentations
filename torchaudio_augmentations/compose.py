import torch


class Compose:
    """Data augmentation module that transforms any given data example with a chain of audio augmentations."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        x = self.transform(x)
        return x

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "\t{0}".format(t)
        format_string += "\n)"
        return format_string

    def transform(self, x):
        for t in self.transforms:
            x = t(x)
        return x


class ComposeMany(Compose):
    """
    Data augmentation module that transforms any given data example randomly
    resulting in N correlated views of the same example
    """

    def __init__(self, transforms, num_augmented_samples):
        self.transforms = transforms
        self.num_augmented_samples = num_augmented_samples

    def __call__(self, x):
        samples = []
        for _ in range(self.num_augmented_samples):
            samples.append(self.transform(x).unsqueeze(dim=0).clone())
        return torch.cat(samples, dim=0)
