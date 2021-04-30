import librosa
import numpy as np
import os.path as Path
import pandas as pd
import torch
from torch.utils.data import Dataset


class BirdClefDataset(Dataset):
    def __init__(self, path_meta="train.csv", sr=32000):
        self.window = ''
        self.window_stride = 0.012
        self.window_size = 0.03
        self.meta = pd.read_csv(path_meta)
        self.sr = sr
        self.num_classes = len(self.meta["primary_label"].unique())
        self.SPEC_HEIGHT = 80
        self.n_fft = int(self.sr * (self.window_size + 1e-8))
        self.hop_length = int(self.sr * (self.window_stride + 1e-8))

    def read_ogg(self, path):
        data, sample_rate = librosa.load(path)

        D = librosa.stft(data, n_fft=self.n_fft, hop_length=self.hop_length,
                         win_length=self.n_fft, window='hamming')
        # spect, phase = librosa.magphase(D)
        # 3x faster
        spect = np.abs(D)

        return spect[:self.SPEC_HEIGHT]
        # spectr = librosa.feature.melspectrogram(
        #     y=data,
        #     sr=sample_rate,
        #     n_fft=512,
        #     hop_length=int(sample_rate * 5 / (self.SPEC_WIDTH - 1)),
        #     n_mels=self.SPEC_HEIGHT,
        #     fmin=self.FMIN,
        #     fmax=self.FMAX
        # )
        # return librosa.power_to_db(spectr)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        # wave_path = "train_short_audio/" + row["path"]
        wave_path = Path.join("train_short_audio", row["path"])
        spectr = self.read_ogg(wave_path)
        label = np.zeros(self.num_classes, dtype=np.float32) + 0.0025  # Label smoothing
        label[row["label_id"]] = 0.995
        return torch.tensor(spectr), torch.tensor(label)


def collate_fn(batch):
    # print(type(batch[0][0]))
    # print(batch[0][0].shape)
    batch = sorted(batch, key=lambda sample: sample[0].size(1), reverse=True)
    longest_sample = max(batch, key=lambda sample: sample[0].size(1))[0]
    freq_size = longest_sample.size(0)
    minibatch_size = len(batch)
    max_seqlength = longest_sample.size(1)
    inputs = torch.zeros(minibatch_size, 1, freq_size, max_seqlength)
    input_percentages = torch.FloatTensor(minibatch_size)
    targets = []
    for x in range(minibatch_size):
        tensor, target = batch[x]
        seq_length = tensor.size(1)
        inputs[x][0].narrow(1, 0, seq_length).copy_(tensor)
        input_percentages[x] = seq_length / float(max_seqlength)
        targets.append(target)
    return inputs, targets, input_percentages
