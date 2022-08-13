from .forecasting_model import ForecastingModel
from statsmodels.tsa.holtwinters import ExponentialSmoothing as StatsModelsHoltWinters
from statsmodels.tsa.holtwinters import Holt as StatsModelsHolt


class HoltWinters(ForecastingModel):
    def __init__(self):
        super().__init__('Holt-Winters')

    def forecast(self, ts, horizon=1):
        if ts.seasonality is not None:
            if (ts.data <= 0).values.any():
                model = StatsModelsHoltWinters(ts.data, seasonal_periods=ts.seasonality, trend='add', seasonal='add')
            else:
                model = StatsModelsHoltWinters(ts.data, seasonal_periods=ts.seasonality, trend='add', seasonal='mul')

        else:
            model = StatsModelsHolt(ts.data, damped_trend=True)
        forecasts = model.fit().forecast(steps=horizon)
        return forecasts
