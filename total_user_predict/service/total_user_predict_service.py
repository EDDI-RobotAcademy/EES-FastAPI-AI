from abc import ABC, abstractmethod


class TotalUserPredictService(ABC):
    @abstractmethod
    def train_total_user(self, trainer):
        pass

    @abstractmethod
    def predict_total_user(self, model, device, dataset):
        pass
