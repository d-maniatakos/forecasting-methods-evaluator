import pandas as pd
from .forecasting_model import ForecastingModel


class Naive(ForecastingModel):
    def __init__(self):
        super().__init__('Naive Method')

    def forecast(self, ts, horizon=1):
        forecasts_list = [ts.data[-1]] * horizon

        if ts.frequency == 'MS' or ts.frequency == 'M':
            start = ts.data.index[-1] + pd.tseries.offsets.DateOffset(months=1)
        if ts.frequency == 'D':
            start = ts.data.index[-1] + pd.tseries.offsets.DateOffset(days=1)
        dates_list = pd.date_range(start=start, periods=horizon, freq=ts.frequency)

        forecasts = pd.Series(index=dates_list, data=forecasts_list)

        return forecasts
