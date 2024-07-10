import os

import pandas as pd
import torch
from torch.utils.data import Dataset


class PreferredProductPredictDataset(Dataset):
    def __init__(self, root, normalize=False):
        self.dataset = os.path.join(root, "order_info.csv")
        self.normalize = normalize

        self.X, self.y = self._get_data()

    def _get_data(self):
        dataset = pd.read_csv(self.dataset)
        dataset.gender = dataset.gender.apply(lambda x: 1 if x == "MALE" else 0).astype(
            "int"
        )

        targets = dataset[["gender", "birth_year"]]
        labels = dataset[["product_id"]]

        self.min_features = targets.min(axis=0)
        self.max_features = targets.max(axis=0)

        if self.normalize:
            targets = self.scale_data(targets, self.min_features, self.max_features)

        X = targets.to_numpy()
        y = labels.to_numpy() - 1
        return torch.tensor(X).float(), torch.tensor(y).squeeze()

    @staticmethod
    def scale_data(data, min_features, max_features):
        return (data - min_features) / (max_features - min_features + 1e-6)

    @staticmethod
    def reverse_scale_data(data, min_features, max_features):
        return data * (max_features - min_features) + min_features

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
