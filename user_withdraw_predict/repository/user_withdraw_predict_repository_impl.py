import torch

from user_withdraw_predict.entity.dataset import UserWithdrawPredictDataset
from user_withdraw_predict.entity.model import UserWithdrawPredictModel
from user_withdraw_predict.entity.trainer import UserWithdrawPredictTrainer
from user_withdraw_predict.repository.user_withdraw_predict_repository import (
    UserWithdrawPredictRepository,
)


class UserWithdrawPredictRepositoryImpl(UserWithdrawPredictRepository):
    def load_data(self, root):
        return UserWithdrawPredictDataset(root, normalize=True)

    def load_model(
        self, in_features=11, out_features=1, hidden_size=64, ckpt_path=None
    ):
        model = UserWithdrawPredictModel(
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
        trainer = UserWithdrawPredictTrainer(
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
        data = data.unsqueeze(0).to(device)
        model.eval()
        with torch.no_grad():
            data = data.to(device)
            pred = model(data)
            withdraw_prob = pred.item() * 100
            print(f"withdraw probability: {withdraw_prob:.2f}%")
        print('pred', pred)
        withdraw = 1 if withdraw_prob >= 50 else 0

        return withdraw, withdraw_prob

    def scale_data(self, data, min_features, max_features):
        return UserWithdrawPredictDataset.scale_data(
            data, min_features[:-1], max_features[:-1]
        )

    def reverse_scale_data(self, data, min_features, max_features):
        return UserWithdrawPredictDataset.reverse_scale_data(
            data, min_features, max_features
        )
