import os

import pandas as pd
import torch
from torch.utils.data import Dataset


class DateInfoPredictDataset(Dataset):
    def __init__(self, root, normalize=False, window_size=30):
        self.dataset = os.path.join(root, "date_info.csv")
        self.normalize = normalize
        self.window_size = window_size

        self.inputs, self.targets = self._get_data()

    def _get_data(self):
        dataset = pd.read_csv(self.dataset).drop("date", axis=1).to_numpy()
        self.min_features = dataset.min(axis=0)
        self.max_features = dataset.max(axis=0)

        if self.normalize:
            dataset = self.scale_data(dataset, self.min_features, self.max_features)

        return self._windowing(dataset)

    def _windowing(self, array):
        X, y = [], []

        for i in range(len(array) - self.window_size * 2):
            X.append(torch.tensor(array[i : i + self.window_size], dtype=torch.float32))
            y.append(
                torch.tensor(
                    array[i + self.window_size : i + self.window_size * 2],
                    dtype=torch.float32,
                )
            )

        return X, y

    @staticmethod
    def scale_data(data, min_features, max_features):
        return (data - min_features) / (max_features - min_features + 1e-6)

    @staticmethod
    def reverse_scale_data(data, min_features, max_features):
        return data * (max_features - min_features) + min_features

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]
