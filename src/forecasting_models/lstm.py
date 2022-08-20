import pandas as pd
from .forecasting_model import ForecastingModel
from kats.models.lstm import LSTMModel, LSTMParams
from kats.consts import TimeSeriesData


class LSTM(ForecastingModel):
    def __init__(self):
        super().__init__('LSTM')

    def forecast(self, ts, horizon=1):
        df = ts.data.to_frame().reset_index()
        df.columns = ['time', 'value']

        # TODO: Generalization
        if len(ts.data) < 30:
            params = LSTMParams(hidden_size=100, time_window=10, num_epochs=50)
        elif len(ts.data) < 150:
            params = LSTMParams(hidden_size=100, time_window=30, num_epochs=30)
        elif len(ts.data) < 500:
            params = LSTMParams(hidden_size=100, time_window=50, num_epochs=20)
        else:
            params = LSTMParams(hidden_size=100, time_window=50, num_epochs=10)
        data = TimeSeriesData(df)
        model = LSTMModel(data=data, params=params)
        model.fit()

        model.predict(steps=horizon, freq=ts.frequency)

        forecasts = model.fcst_df
        forecasts.index = forecasts['time']
        forecasts = forecasts[['time', 'fcst']]
        forecasts.columns = ['time', 'value']
        forecasts.set_index('time')
        forecasts = forecasts['value']

        return forecasts
