import pandas as pd
from .forecasting_model import ForecastingModel
from statsmodels.tsa.arima.model import ARIMA as StatsModelsARIMA
import pmdarima as pm
from time_series import TimeSeries


class ARIMA(ForecastingModel):
    def __init__(self):
        super().__init__('ARIMA')

    def forecast(self, ts, horizon=1, order=None, seasonal_order=None):

        if order is None or seasonal_order is None:
            if ts.seasonality is not None:
                auto_arima_params = pm.auto_arima(ts.data,
                                                  error_action='ignore',
                                                  trace=False,
                                                  suppress_warnings=True,
                                                  maxiter=5,
                                                  seasonal=True,
                                                  m=ts.seasonality)
            else:
                auto_arima_params = pm.auto_arima(ts.data,
                                                  error_action='ignore',
                                                  trace=False,
                                                  suppress_warnings=True,
                                                  maxiter=5,
                                                  seasonal=False)

            order = auto_arima_params.order
            seasonal_order = auto_arima_params.seasonal_order

        known_observations = ts.data
        forecasts = pd.Series(dtype='float64')
        for i in range(horizon):
            model = StatsModelsARIMA(known_observations, order=order, seasonal_order=seasonal_order)
            forecast = model.fit().forecast()
            known_observations = pd.concat([known_observations, forecast])
            forecasts = pd.concat([forecasts, forecast])

        return forecasts

    def one_step_ahead_evaluate(self, ts, train_ratio=0.7):
        super().one_step_ahead_evaluate(ts, train_ratio)

    def multi_step_ahead_evaluate(self, ts, train_ratio=0.7):
        super().multi_step_ahead_evaluate(ts, train_ratio)
