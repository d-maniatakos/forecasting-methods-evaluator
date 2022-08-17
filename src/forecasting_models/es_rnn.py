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

        model = ESRNN(max_epochs=200, learning_rate=1e-4,
                      per_series_lr_multip=0.8, lr_scheduler_step_size=10,
                      lr_decay=0.1, gradient_clipping_threshold=50,
                      rnn_weight_decay=0.0, level_variability_penalty=100,
                      ensemble=True, seasonality=[],
                      input_size=24, output_size=horizon,
                      cell_type='LSTM', state_hsize=20,
                      dilations=[[1], [6]], add_nl_layer=False,
                      random_seed=1, device='cuda')

        model.fit(x_df, y_df)
        x_test_df = x_df[:horizon].copy()

        if ts.frequency == 'MS' or ts.frequency == 'M':
            start = ts.data.index[-1] + pd.tseries.offsets.DateOffset(months=1)
            x_test_df['ds'] = pd.date_range(start=start, periods=horizon, freq=ts.frequency)

        y_hat_df = model.predict(x_test_df)

        y_hat_df = y_hat_df.set_index('ds')
        y_hat_df = y_hat_df['y_hat']
        y_hat_df.rename(index = 'Date')

        return y_hat_df
