import torch

from total_user_predict.entity.dataset import WithdrawPredictDataset
from total_user_predict.entity.model import WithdrawPredictModel
from total_user_predict.entity.trainer import WithdrawPredictTrainer
from total_user_predict.repository.total_user_predict_repository import (
    TotalUserPredictRepository,
)


class TotalUserPredictRepositoryImpl(TotalUserPredictRepository):
    def load_data(self, csv_path, window_size):
        return WithdrawPredictDataset(csv_path, normalize=True, window_size=window_size)

    def load_model(self, in_features=5, out_features=5, hidden_size=32, ckpt_path=None):
        model = WithdrawPredictModel(
            in_features=in_features, out_features=out_features, hidden_size=hidden_size
        )
        if ckpt_path:
            try:
                ckpt = torch.load(ckpt_path)
                model.load_state_dict(ckpt)
            except FileNotFoundError:
                raise FileNotFoundError("No checkpoint found. Train first.")

        return model

    def load_trainer(
        self,
        model,
        train_dataset_loader,
        val_dataset_loader,
        optimizer,
        criterion,
        epochs,
        model_path,
        model_name,
        device="cpu",
    ):
        trainer = WithdrawPredictTrainer(
            model,
            train_dataset_loader,
            val_dataset_loader,
            optimizer,
            criterion,
            epochs,
            model_path,
            model_name,
            device,
        )
        return trainer

    def train_model(self, trainer):
        trainer.train()

    def predict(self, model, data, device="cpu"):

        predicted_n_days_after = (
            model(data.unsqueeze(0).to(device)).squeeze().detach().cpu().numpy()
        )

        return predicted_n_days_after

    def reverse_scale_data(self, data, min_features, max_features):
        return WithdrawPredictDataset.reverse_scale_data(
            data, min_features, max_features
        )