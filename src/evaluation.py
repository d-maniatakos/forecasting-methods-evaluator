import pandas as pd

from forecasting_models.arima import ARIMA
from forecasting_models.holt_winters import HoltWinters
from forecasting_models.naive import Naive
from forecasting_models.lstm import LSTM
from forecasting_models.es_rnn import ES_RNN


class Evaluation:
    def __init__(self, time_series):
        self.time_series = time_series
        self.models = []
        self.models.append(HoltWinters())
        self.models.append(ARIMA())
        self.models.append(LSTM())
        self.models.append(ES_RNN())
        self.models.append(Naive())
        self.one_step_ahead_evaluations = pd.DataFrame(columns=[model.name for model in self.models],
                                                       index=[ts.name for ts in self.time_series])
        self.multi_step_ahead_evaluations = pd.DataFrame(columns=[model.name for model in self.models],
                                                         index=[ts.name for ts in self.time_series])

    def evaluate(self):
        for ts in self.time_series:
            for model in self.models:
                try:
                    print(ts.name)
                    print(model.name)
                    scores = model.one_step_ahead_evaluate(ts, 0.8, suppress_iterations=False)
                    self.one_step_ahead_evaluations.at[ts.name, model.name] = scores['mape']
                except:
                    pass

        for ts in self.time_series:
            print(ts.name)
            for model in self.models:
                print(model.name)
                try:
                    scores = model.multi_step_ahead_evaluate(ts, 0.8)
                    self.multi_step_ahead_evaluations.at[ts.name, model.name] = scores['mape']
                except:
                    pass

        self.one_step_ahead_evaluations.loc['Mean'] = self.one_step_ahead_evaluations.mean()
        self.multi_step_ahead_evaluations.loc['Mean'] = self.multi_step_ahead_evaluations.mean()
        print('One-Step-Ahead Evaluation (Mean Absolute Percentage Error)')
        print(self.one_step_ahead_evaluations)
        print('Multi-Step-Ahead Evaluation (Mean Absolute Percentage Error)')
        print(self.multi_step_ahead_evaluations)

        self.one_step_ahead_evaluations.to_csv('one_step_ahead_evaluations.csv')
        self.multi_step_ahead_evaluations.to_csv('multi_step_ahead_evaluations.csv')