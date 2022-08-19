import pandas as pd
from .forecasting_model import ForecastingModel

class LSTM(ForecastingModel):
    def __init__(self):
        super().__init__('LSTM')

    def forecast(self, ts, horizon=1):
        return
