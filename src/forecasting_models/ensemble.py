import pandas as pd
import numpy as np
from .forecasting_model import ForecastingModel
from .arima import ARIMA
from .holt_winters import HoltWinters

class Ensemble(ForecastingModel):
    def __init__(self):
        super().__init__('Ensemble')

    def forecast(self, ts, horizon=1):
        arima_model = ARIMA()
        holt_winters_model = HoltWinters()

        arima_forecasts = arima_model.forecast(ts, horizon)
        holt_winters_forecasts = holt_winters_model.forecast(ts, horizon)

        ensemble_forecasts = pd.Series(np.mean(np.stack((arima_forecasts, holt_winters_forecasts)), axis=0))
        ensemble_forecasts.index = arima_forecasts.index

        return ensemble_forecasts
