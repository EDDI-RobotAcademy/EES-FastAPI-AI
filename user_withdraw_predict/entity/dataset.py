import os

import pandas as pd
import torch
from torch.utils.data import Dataset


class UserWithdrawPredictDataset(Dataset):
    # WITHDRAW_REASONS = {
    #     "SERVICE_DISSATISFACTION": 1,
    #     "LOW_USAGE": 2,
    #     "OTHER_SERVICE": 3,
    #     "PRIVACY_CONCERN": 4,
    #     "OTHER": 5,
    # }

    def __init__(self, root, normalize=False):
        self.dataset = os.path.join(root, "user_info.csv")
        self.normalize = normalize

        self.X, self.y = self._get_data()

    def _get_data(self):
        dataset = pd.read_csv(self.dataset)
        dataset.last_login_to_withdraw = dataset.last_login_to_withdraw.fillna(0)
        dataset.gender = dataset.gender.apply(lambda x: 1 if x == "MALE" else 0).astype(
            "int"
        )
        dataset.withdraw = dataset.withdraw.astype("int")
        # dataset.withdraw_reason = dataset.withdraw_reason.fillna(0)
        # dataset.withdraw_reason = dataset.withdraw_reason.apply(
        #     lambda x: self.WITHDRAW_REASONS[x] if x in self.WITHDRAW_REASONS else 0
        # ).astype('int')
        dataset.drop("withdraw_reason", axis=1, inplace=True)
        self.min_features = dataset.min(axis=0)
        self.max_features = dataset.max(axis=0)

        if self.normalize:
            dataset = self.scale_data(dataset, self.min_features, self.max_features)

        X = dataset.drop("withdraw", axis=1).to_numpy()
        y = dataset.withdraw.to_numpy()

        return torch.tensor(X).float(), torch.tensor(y).unsqueeze(1).float()

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
