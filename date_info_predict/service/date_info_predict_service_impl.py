import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from date_info_predict.repository.date_info_predict_repository_impl import (
    DateInfoPredictRepositoryImpl,
)
from date_info_predict.service.date_info_predict_service import (
    DateInfoPredictService,
)


class DateInfoPredictServiceImpl(DateInfoPredictService):
    DATASET_ROOT = "assets/dataset"
    MODEL_ROOT = "assets/model"
    MODEL_NAME = "date_info_predict_model.pt"

    DEVICE = "cpu"
    WINDOW_SIZE = 30
    VAL_RATIO = 0.2
    BATCH_SIZE = 64
    EPOCHS = 1000
    IN_FEATURES = 9
    OUT_FEATURES = 9
    HIDDEN_SIZE = 32

    def __init__(self):
        self.date_info_predict_repository = DateInfoPredictRepositoryImpl()

    def train_date_info(self):
        dataset = self.date_info_predict_repository.load_data(
            self.DATASET_ROOT, window_size=self.WINDOW_SIZE
        )
        model = self.date_info_predict_repository.load_model(
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

        trainer = self.date_info_predict_repository.load_trainer(
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

        self.date_info_predict_repository.train_model(trainer)

    def predict_date_info(self, n_days_after, feature):
        label_idx = {
            "profit": -4,
            "total_user": -2,
        }
        
        result = []

        model = self.date_info_predict_repository.load_model(
            self.IN_FEATURES,
            self.OUT_FEATURES,
            self.HIDDEN_SIZE,
            os.path.join(self.MODEL_ROOT, self.MODEL_NAME),
        )
        dataset = self.date_info_predict_repository.load_data(
            self.DATASET_ROOT, window_size=self.WINDOW_SIZE
        )

        num_iter = n_days_after // self.WINDOW_SIZE
        num_iter = num_iter + 1 if n_days_after % self.WINDOW_SIZE else num_iter

        last_n_days = dataset[-1][0]

        for _ in range(num_iter):
            predicted_n_days_after = self.date_info_predict_repository.predict(
                model, last_n_days, self.DEVICE
            )
            reverse_scaled_predicted_n_days_after = (
                self.date_info_predict_repository.reverse_scale_data(
                    predicted_n_days_after, dataset.min_features, dataset.max_features
                ).astype("int")
            )[:, label_idx[feature]].tolist()

            result += reverse_scaled_predicted_n_days_after

            last_n_days = torch.tensor(predicted_n_days_after, dtype=torch.float32)

        predicted_data = result[:n_days_after]

        return predicted_data
