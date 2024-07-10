from abc import ABC, abstractmethod


class TimeseriesService:
    @abstractmethod
    def fit_model(self):
        pass

    @abstractmethod
    def forecast(self, steps):
        pass