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
        params = LSTMParams(hidden_size=100, time_window=30, num_epochs=30)
        data = TimeSeriesData(df)
        model = LSTMModel(data=data, params=params)
        model.fit()

        model.predict(steps=horizon, freq=ts.frequency)

        forecasts_df = model.fcst_df
        forecasts_df.index = forecasts_df['time']
        forecasts_df = forecasts_df[['time', 'fcst']]
        forecasts_df.columns = ['time', 'value']
        forecasts_df.set_index('time')
        forecasts_df = forecasts_df['value']

        return forecasts_df
