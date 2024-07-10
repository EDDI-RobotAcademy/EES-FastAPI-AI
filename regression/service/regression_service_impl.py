from sklearn.linear_model import LinearRegression

from regression.service.regression_service import RegressionService


class RegressionServiceImpl(RegressionService):
    def __init__(self, repository):
        self.repository = repository
        self.model = None

    def fit_model(self):
        try:
            data = self.repository.get_order_data()
            X = data[['gender', 'birth_year', 'date']]
            y = data['total_price']
            self.model = LinearRegression()
            self.model.fit(X, y)
            return self.model
        except Exception as e:
            print(f"Error fitting regression model: {e}")
            raise

    def predict(self, X):
        try:
            if not self.model:
                self.fit_model()
            predictions = self.model.predict(X)
            return predictions
        except Exception as e:
            print(f"Error predicting regression model: {e}")
            raise
