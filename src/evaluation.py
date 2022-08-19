import pandas as pd

from forecasting_models.holt_winters import HoltWinters
from forecasting_models.arima import ARIMA
from forecasting_models.lstm import LSTM
from forecasting_models.es_rnn import ES_RNN
from forecasting_models.naive import Naive


class Evaluation:
    def __init__(self, time_series=None):
        if time_series is not None:
            self.time_series = time_series
        else:
            self.time_series = []
        self.models = []
        self.initialize_models()
        self.one_step_ahead_evaluations = None
        self.multi_step_ahead_evaluations = None
        self.initialize_results_tables()

    def add_time_series(self, ts):
        self.time_series.append(ts)
        self.initialize_results_tables()

    def initialize_results_tables(self):
        self.one_step_ahead_evaluations = pd.DataFrame(columns=[model.name for model in self.models],
                                                       index=[ts.name for ts in self.time_series])
        self.multi_step_ahead_evaluations = pd.DataFrame(columns=[model.name for model in self.models],
                                                         index=[ts.name for ts in self.time_series])

    def initialize_models(self):
        self.models.append(HoltWinters())
        # self.models.append(ARIMA())
        # self.models.append(LSTM())
        self.models.append(ES_RNN())
        self.models.append(Naive())

    def evaluate(self, one_step_ahead_evaluation=False, multi_step_ahead_evaluation=True, export_to_csv=False):

        if one_step_ahead_evaluation:
            for ts in self.time_series:
                print('Time Series: ' + ts.name)
                for model in self.models:
                    print('\tModel: ' + model.name)
                    try:
                        scores = model.one_step_ahead_evaluate(ts, 0.7, suppress_iterations_print=False)
                        self.one_step_ahead_evaluations.at[ts.name, model.name] = scores['mape']
                    except:
                        pass

        if multi_step_ahead_evaluation:
            for ts in self.time_series:
                print('Time Series: ' + ts.name)
                for model in self.models:
                    print('\tModel: ' + model.name)

                    try:
                        scores = model.multi_step_ahead_evaluate(ts, 0.7)
                        self.multi_step_ahead_evaluations.at[ts.name, model.name] = scores['mape']
                    except:
                        pass

        if one_step_ahead_evaluation:
            self.one_step_ahead_evaluations.loc['Mean'] = self.one_step_ahead_evaluations.mean()
            print('One-Step-Ahead Evaluation (Mean Absolute Percentage Error)')
            print(self.one_step_ahead_evaluations)
            if export_to_csv:
                self.one_step_ahead_evaluations.to_csv('results/one_step_ahead_evaluations' + '_' + ' ' + '.csv')

        if multi_step_ahead_evaluation:
            self.multi_step_ahead_evaluations.loc['Mean'] = self.multi_step_ahead_evaluations.mean()
            print('Multi-Step-Ahead Evaluation (Mean Absolute Percentage Error)')
            print(self.multi_step_ahead_evaluations)
            if export_to_csv:
                self.multi_step_ahead_evaluations.to_csv('multi_step_ahead_evaluations' + '_' + ' ' + '.csv')
