from abc import ABC, abstractmethod


class UserWithdrawPredictService(ABC):
    @abstractmethod
    def train_user_withdraw(self):
        pass

    @abstractmethod
    def predict_user_withdraw(self, model, device, dataset):
        pass
