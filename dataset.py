import random

import librosa
import numpy as np
import os.path as Path
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchaudio


class BirdClefDataset(Dataset):
    def __init__(self, path_meta="train.csv", sr=32000, crop=15):
        self.crop = crop
        self.window = ''
        self.window_stride = 0.012
        self.window_size = 0.03
        self.meta = pd.read_csv(path_meta)
        self.num_classes = len(self.meta["primary_label"].unique())
        self.Melbins = 80
        self.amplitude = torchaudio.transforms.AmplitudeToDB()

    def read_ogg(self, path):
        data, sample_rate = torchaudio.load(path)
        data = data
        audio_len = data.size(1) / sample_rate
        window_length = int(sample_rate * (self.window_size + 1e-8))
        hop_length = int(sample_rate * (self.window_stride + 1e-8))
        if 0 < self.crop < audio_len:
            start = random.randint(0, int(sample_rate * (audio_len - self.crop)))
            data = data[:, start: start + int(self.crop * sample_rate)]
        spect = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=1024,
                                                     hop_length=hop_length, win_length=window_length)(data)
        spect = self.amplitude(spect)
        return spect[0, :self.Melbins]

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        # wave_path = "train_short_audio/" + row["path"]
        wave_path = Path.join("train_short_audio", row["path"])
        spectr = self.read_ogg(wave_path)
        # TODO: Add augs here
        label = np.zeros(self.num_classes, dtype=np.float32) + 0.003  # Label smoothing
        label[row["label_id"]] = 0.995
        return spectr, torch.tensor(label)


def collate_fn(batch):

    batch = sorted(batch, key=lambda sample: sample[0].size(1), reverse=True)
    longest_sample = max(batch, key=lambda sample: sample[0].size(1))[0]
    freq_size = longest_sample.size(0)
    minibatch_size = len(batch)
    max_seqlength = longest_sample.size(1)
    inputs = torch.zeros(minibatch_size, 1, freq_size, max_seqlength)
    targets = []
    for x in range(minibatch_size):
        tensor, target = batch[x]
        seq_length = tensor.size(1)
        inputs[x][0].narrow(1, 0, seq_length).copy_(tensor)
        targets.append(target)
    return inputs, torch.stack(targets) #input_percentages
