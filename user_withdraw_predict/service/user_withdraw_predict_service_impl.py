import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from user_withdraw_predict.repository.user_withdraw_predict_repository_impl import (
    UserWithdrawPredictRepositoryImpl,
)
from user_withdraw_predict.service.user_withdraw_predict_service import (
    UserWithdrawPredictService,
)


class UserWithdrawPredictServiceImpl(UserWithdrawPredictService):
    DATASET_ROOT = "assets/dataset"
    MODEL_ROOT = "assets/model"
    MODEL_NAME = "user_withdraw_predict_model.pt"

    DEVICE = "cpu"
    VAL_RATIO = 0.2
    BATCH_SIZE = 64
    EPOCHS = 1000
    IN_FEATURES = 11
    OUT_FEATURES = 1
    HIDDEN_SIZE = 64

    def __init__(self):
        self.user_withdraw_predict_repository = UserWithdrawPredictRepositoryImpl()

    def train_user_withdraw(self):
        dataset = self.user_withdraw_predict_repository.load_data(
            root=self.DATASET_ROOT
        )

        model = self.user_withdraw_predict_repository.load_model(
            in_features=self.IN_FEATURES,
            out_features=self.OUT_FEATURES,
            hidden_size=self.HIDDEN_SIZE,
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

        trainer = self.user_withdraw_predict_repository.load_trainer(
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

        self.user_withdraw_predict_repository.train_model(trainer)

    def predict_user_withdraw(self, *data):
        data = np.array(data)
        dataset = self.user_withdraw_predict_repository.load_data(
            root=self.DATASET_ROOT
        )

        # TODO : Implement scaler for resource efficiency
        scaled_data = self.user_withdraw_predict_repository.scale_data(
            data, dataset.min_features, dataset.max_features
        )
        scaled_data = torch.tensor(scaled_data.to_numpy()).float()
        model = self.user_withdraw_predict_repository.load_model(
            in_features=self.IN_FEATURES,
            out_features=self.OUT_FEATURES,
            hidden_size=self.HIDDEN_SIZE,
            ckpt_path=f"{self.MODEL_ROOT}/{self.MODEL_NAME}",
        )

        withdraw, withdraw_prob = self.user_withdraw_predict_repository.predict(
            model=model,
            data=scaled_data,
            device=self.DEVICE,
        )

        return withdraw, withdraw_prob
