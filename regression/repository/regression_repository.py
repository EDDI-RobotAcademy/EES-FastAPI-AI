from abc import ABC, abstractmethod

class RegressionRepository:

    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def get_order_data(self):
        pass

