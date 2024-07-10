from statsmodels.tsa.holtwinters import ExponentialSmoothing

from timeseries.service.timeseries_service import TimeseriesService

class TimeseriesServiceImpl(TimeseriesService):
    def __init__(self, repository):
        self.repository = repository
        self.model_fit = None

    def fit_model(self):
        data = self.repository.get_order_data()
        model = ExponentialSmoothing(data['total_price'], seasonal='add', seasonal_periods=12)
        self.model_fit = model.fit()
        return self.model_fit

    def forecast(self, steps):
        if not self.model_fit:
            self.fit_model()
        forecast = self.model_fit.forecast(steps)
        return forecast
