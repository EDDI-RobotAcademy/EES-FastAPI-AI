from abc import ABC, abstractmethod


class PreferredProductPredictService(ABC):
    @abstractmethod
    def train_preferred_product(self):
        pass

    @abstractmethod
    def predict_preferred_product(self, model, device, dataset):
        pass
