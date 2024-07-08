import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from total_user_predict.repository.total_user_predict_repository_impl import (
    TotalUserPredictRepositoryImpl,
)
from total_user_predict.service.total_user_predict_service import (
    TotalUserPredictService,
)


class TotalUserPredictServiceImpl(TotalUserPredictService):
    DATASET_ROOT = "assets/dataset"
    MODEL_ROOT = "assets/model"
    MODEL_NAME = "total_user_predict_model.pt"

    DEVICE = "cpu"
    WINDOW_SIZE = 30
    VAL_RATIO = 0.2
    BATCH_SIZE = 64
    EPOCHS = 100
    IN_FEATURES = 5
    OUT_FEATURES = 5
    HIDDEN_SIZE = 32

    def __init__(self):
        self.total_user_predict_repository = TotalUserPredictRepositoryImpl()

    def train_total_user(self):
        dataset = self.total_user_predict_repository.load_data(
            self.DATASET_ROOT, window_size=self.WINDOW_SIZE
        )
        model = self.total_user_predict_repository.load_model(
            self.IN_FEATURES, self.OUT_FEATURES, self.HIDDEN_SIZE
        )

        train_data, val_data = random_split(
            dataset,
            [
                int(len(dataset) * (1 - self.VAL_RATIO)),
                len(dataset) - int(len(dataset) * (1 - self.VAL_RATIO)),
            ],
        )

        train_loader = DataLoader(train_data, batch_size=self.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=self.BATCH_SIZE, shuffle=False)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        trainer = self.total_user_predict_repository.load_trainer(
            model=model,
            train_dataset_loader=train_loader,
            val_dataset_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            epochs=self.EPOCHS,
            model_path=self.MODEL_ROOT,
            model_name=self.MODEL_NAME,
            device=self.DEVICE,
        )

        self.total_user_predict_repository.train_model(trainer)

    def predict_total_user(self, n_days_after):
        result = []

        model = self.total_user_predict_repository.load_model(
            self.IN_FEATURES,
            self.OUT_FEATURES,
            self.HIDDEN_SIZE,
            os.path.join(self.MODEL_ROOT, self.MODEL_NAME),
        )
        dataset = self.total_user_predict_repository.load_data(
            self.DATASET_ROOT, window_size=self.WINDOW_SIZE
        )

        num_iter = n_days_after // self.WINDOW_SIZE
        num_iter = num_iter + 1 if n_days_after % self.WINDOW_SIZE else num_iter

        last_n_days = dataset[-1][0]

        for _ in range(num_iter):
            predicted_n_days_after = self.total_user_predict_repository.predict(
                model, last_n_days, self.DEVICE
            )
            reverse_scaled_predicted_n_days_after = (
                self.total_user_predict_repository.reverse_scale_data(
                    predicted_n_days_after, dataset.min_features, dataset.max_features
                ).astype("int")
            )[:, -1].tolist()

            result += reverse_scaled_predicted_n_days_after

            last_n_days = torch.tensor(predicted_n_days_after, dtype=torch.float32)

        predicted_data = result[:n_days_after]

        return predicted_data
