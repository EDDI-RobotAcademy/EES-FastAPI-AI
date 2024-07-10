import torch

from preferred_product_predict.entity.dataset import PreferredProductPredictDataset
from preferred_product_predict.entity.model import PreferredProductPredictModel
from preferred_product_predict.entity.trainer import PreferredProductPredictTrainer
from preferred_product_predict.repository.preferred_product_predict_repository import (
    PreferredProductPredictRepository,
)


class PreferredProductPredictRepositoryImpl(PreferredProductPredictRepository):
    def load_data(self, root):
        return PreferredProductPredictDataset(root, normalize=True)

    def load_model(
        self, in_features=2, out_features=47, hidden_size=64, ckpt_path=None
    ):
        model = PreferredProductPredictModel(
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
        trainer = PreferredProductPredictTrainer(
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
            predicted_product, predicted_prob = pred.argmax(dim=1).item() + 1, pred.max().item() * 100
            
            print(f'predicted product: {predicted_product}')
            print(f"predicted probability: {predicted_prob:.2f}%")

        return predicted_product, predicted_prob

    def scale_data(self, data, min_features, max_features):
        return PreferredProductPredictDataset.scale_data(
            data, min_features, max_features
        )

    def reverse_scale_data(self, data, min_features, max_features):
        return PreferredProductPredictDataset.reverse_scale_data(
            data, min_features, max_features
        )
