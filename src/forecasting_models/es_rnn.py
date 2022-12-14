from .forecasting_model import ForecastingModel
import pandas as pd
from ESRNN import ESRNN


class ES_RNN(ForecastingModel):
    def __init__(self):
        super().__init__('ES-RNN')

    def forecast(self, ts, horizon=1):
        y_df = ts.data.to_frame().reset_index()
        y_df.columns = ['ds', 'y']
        y_df.index.name = 'unique_id'
        y_df = y_df.reset_index()
        y_df['unique_id'] = 'Y1'
        y_df['y'] = y_df['y'].astype(float)

        x_df = y_df.copy()
        x_df.columns = ['unique_id', 'ds', 'x']
        x_df['x'] = x_df['x'].astype('str')

        # TODO: Generalization
        if len(ts.data) < 100:
            max_epochs = 100
        elif len(ts.data) < 200:
            max_epochs = 200
        elif len(ts.data) < 1000:
            max_epochs = 100
        else:
            max_epochs = 20

        if len(ts.data < 24 - horizon):
            input_size = 6
        else:
            input_size = 24

        model = ESRNN(max_epochs=max_epochs, learning_rate=1e-3,
                      per_series_lr_multip=0.8, lr_scheduler_step_size=10,
                      lr_decay=0.1, gradient_clipping_threshold=50,
                      rnn_weight_decay=0.0, level_variability_penalty=100,
                      ensemble=True, seasonality=[],
                      input_size=input_size, output_size=horizon,
                      cell_type='LSTM', state_hsize=30,
                      dilations=[[1], [6]], add_nl_layer=False,
                      random_seed=1, device='cuda')

        model.fit(x_df, y_df)
        x_test_df = x_df[:horizon].copy()

        if ts.frequency == 'MS' or ts.frequency == 'M':
            start = ts.data.index[-1] + pd.tseries.offsets.DateOffset(months=1)
        elif ts.frequency == 'D':
            start = ts.data.index[-1] + pd.tseries.offsets.DateOffset(days=1)
        elif ts.frequency == 'Y':
            start = ts.data.index[-1] + pd.tseries.offsets.DateOffset(years=1)

        x_test_df['ds'] = pd.date_range(start=start, periods=horizon, freq=ts.frequency)
        forecasts = model.predict(x_test_df)

        forecasts = forecasts.set_index('ds')
        forecasts = forecasts['y_hat']
        forecasts.rename(index='Date')

        return forecasts
