from abc import ABC, abstractmethod


class DateInfoPredictService(ABC):
    @abstractmethod
    def train_date_info(self, trainer):
        pass

    @abstractmethod
    def predict_date_info(self, model, device, dataset):
        pass
