import pandas as pd

from forecasting_models.arima import ARIMA
from forecasting_models.holt_winters import HoltWinters
from forecasting_models.naive import Naive


class Evaluation:
    def __init__(self, time_series):
        self.time_series = time_series
        self.models = []
        self.models.append(ARIMA())
        self.models.append(HoltWinters())
        self.models.append(Naive())
        self.one_step_ahead_evaluations = pd.DataFrame(columns=[model.name for model in self.models],
                                                       index=[ts.name for ts in self.time_series])
        self.multi_step_ahead_evaluations = pd.DataFrame(columns=[model.name for model in self.models],
                                                         index=[ts.name for ts in self.time_series])

    def evaluate(self):
        for ts in self.time_series:
            for model in self.models:
                scores = model.one_step_ahead_evaluate(ts, 0.8)
                self.one_step_ahead_evaluations.at[ts.name, model.name] = scores['mape']

        for ts in self.time_series:
            for model in self.models:
                scores = model.multi_step_ahead_evaluate(ts, 0.8)
                self.multi_step_ahead_evaluations.at[ts.name, model.name] = scores['mape']

        self.one_step_ahead_evaluations.loc['Mean'] = self.one_step_ahead_evaluations.mean()
        self.multi_step_ahead_evaluations.loc['Mean'] = self.multi_step_ahead_evaluations.mean()
        print(self.one_step_ahead_evaluations)
        print(self.multi_step_ahead_evaluations)
