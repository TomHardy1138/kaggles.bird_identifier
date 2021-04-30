from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
import numpy as np


def build_transforms(train=True):
    return Compose([
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
        TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
        PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
        Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
    ])


if __name__ == '__main__':
    SAMPLE_RATE = 16000
    # Generate 2 seconds of dummy audio for the sake of example
    samples = np.random.uniform(low=-0.2, high=0.2, size=(32000,)).astype(np.float32)
    augment = build_transforms()
    # Augment/transform/perturb the audio data
    augmented_samples = augment(samples=samples, sample_rate=SAMPLE_RATE)
