from abc import ABC, abstractmethod

class RegressionService:

    @abstractmethod
    def fit_model(self):
        pass

    @abstractmethod
    def predict(self, X):
        pass