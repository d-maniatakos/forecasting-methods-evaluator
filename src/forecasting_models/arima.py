import pandas as pd
import pmdarima as pm
from .forecasting_model import ForecastingModel
from statsmodels.tsa.arima.model import ARIMA as StatsModelsARIMA
from time_series import TimeSeries


class ARIMA(ForecastingModel):
    def __init__(self):
        super().__init__('ARIMA')

    def forecast(self, ts, horizon=1, order=None, seasonal_order=None):
        if order is None or seasonal_order is None:
            if ts.seasonality is not None:
                seasonality = ts.seasonality if ts.seasonality != 365 else 7
                auto_arima_params = pm.auto_arima(ts.data,
                                                  error_action='ignore',
                                                  trace=False,
                                                  suppress_warnings=True,
                                                  maxiter=5,
                                                  seasonal=True,
                                                  m=seasonality)
            else:
                auto_arima_params = pm.auto_arima(ts.data,
                                                  error_action='ignore',
                                                  trace=False,
                                                  suppress_warnings=True,
                                                  maxiter=5,
                                                  seasonal=False)

            order = auto_arima_params.order
            seasonal_order = auto_arima_params.seasonal_order

        model = StatsModelsARIMA(ts.data, order=order, seasonal_order=seasonal_order)
        forecasts = model.fit().forecast(steps=horizon)

        return forecasts
