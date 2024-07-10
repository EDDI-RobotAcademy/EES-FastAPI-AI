from abc import ABC, abstractmethod


class PreferredProductPredictRepository(ABC):
    @abstractmethod
    def load_data(self, root):
        pass

    @abstractmethod
    def load_model(self, in_features, out_features, hidden_size, ckpt_path):
        pass

    @abstractmethod
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
        device,
    ):
        pass

    @abstractmethod
    def train_model(self, trainer, device):
        pass

    @abstractmethod
    def predict(self, model, data, device):
        pass
    
    @abstractmethod
    def scale_data(self, data, min_features, max_features):
        pass

    @abstractmethod
    def reverse_scale_data(self, data, min_features, max_features):
        pass
