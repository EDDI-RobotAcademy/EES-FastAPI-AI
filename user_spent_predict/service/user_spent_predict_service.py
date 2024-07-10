from abc import ABC, abstractmethod


class UserSpentPredictService(ABC):
    @abstractmethod
    def train_user_spent(self):
        pass

    @abstractmethod
    def predict_user_spent(self, model, device, dataset):
        pass
