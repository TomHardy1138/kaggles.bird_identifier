import librosa
import numpy as np
import os.path as Path
import pandas as pd
from torch.utils.data import Dataset


class BirdClefDataset(Dataset):
    def __init__(self, path_meta="train.csv", sr=32000):
        self.meta = pd.read_csv(path_meta)
        self.sr = sr
        self.num_classes = len(self.meta["primary_label"].unique())
        self.SPEC_HEIGHT = 64
        self.SPEC_WIDTH = 256
        self.FMIN = 200
        self.FMAX = 12500

    def read_ogg(self, path):
        data, sample_rate = librosa.load(path)
        spectr = librosa.feature.melspectrogram(
            y=data,
            sr=sample_rate,
            n_fft=1024,
            hop_length=int(sample_rate * 5 / (self.SPEC_WIDTH - 1)),
            n_mels=self.SPEC_HEIGHT,
            fmin=self.FMIN,
            fmax=self.FMAX
        )
        return librosa.power_to_db(spectr)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        # wave_path = "train_short_audio/" + row["path"]
        wave_path = Path.join("train_short_audio", row["path"])
        spectr = self.read_ogg(wave_path)
        label = np.zeros(self.num_classes, dtype=np.float32) + 0.0025  # Label smoothing
        label[row["label_id"]] = 0.995
        return spectr, label
