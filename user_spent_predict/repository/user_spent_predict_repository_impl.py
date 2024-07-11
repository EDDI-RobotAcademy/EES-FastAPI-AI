import torch

from user_spent_predict.entity.dataset import UserSpentPredictDataset
from user_spent_predict.entity.model import UserSpentPredictModel
from user_spent_predict.entity.trainer import UserSpentPredictTrainer
from user_spent_predict.repository.user_spent_predict_repository import (
    UserSpentPredictRepository,
)


class UserSpentPredictRepositoryImpl(UserSpentPredictRepository):
    def load_data(self, root):
        return UserSpentPredictDataset(root, normalize=True)

    def load_model(self, in_features=8, out_features=1, hidden_size=64, ckpt_path=None):
        model = UserSpentPredictModel(
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
        trainer = UserSpentPredictTrainer(
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
            expected_spent = pred.item()

        return expected_spent

    def scale_data(self, data, min_features, max_features):
        return UserSpentPredictDataset.scale_data(
            data, min_features[:-1], max_features[:-1]
        )

    def reverse_scale_data(self, data, min_features, max_features):
        return UserSpentPredictDataset.reverse_scale_data(
            data, min_features, max_features
        )
