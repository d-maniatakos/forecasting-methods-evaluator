import pandas as pd
from forecasting_models.holt_winters import HoltWinters
from forecasting_models.arima import ARIMA
from forecasting_models.lstm import LSTM
from forecasting_models.es_rnn import ES_RNN
from forecasting_models.ensemble import Ensemble
from forecasting_models.naive import Naive


class Evaluation:
    """
    Class responsible for evaluating multiple forecasting models on multiple time series
    """

    def __init__(self, time_series=None):
        """
        Initializes time_series list field, models list field and result tables

        :param time_series: a list of TimeSeries objects
        """
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
        """
        Adds a new Time Series to the time series list field

        :param ts: A TimeSeries object to add to the time series list field
        """
        self.time_series.append(ts)
        self.initialize_results_tables()

    def initialize_results_tables(self):
        """
        Initializes dataframes for displaying results (columns correspond to models and rows correspond to time series)
        """
        self.one_step_ahead_evaluations = pd.DataFrame(columns=[model.name for model in self.models],
                                                       index=[ts.name for ts in self.time_series])
        self.multi_step_ahead_evaluations = pd.DataFrame(columns=[model.name for model in self.models],
                                                         index=[ts.name for ts in self.time_series])

    def initialize_models(self):
        """
        Add your models here by appending them to the models list field
        """

        self.models.append(HoltWinters())
        self.models.append(ARIMA())
        self.models.append(LSTM())
        self.models.append(ES_RNN())
        self.models.append(Ensemble())
        self.models.append(Naive())
        pd.set_option('display.max_columns', 10)

    def evaluate(self, one_step_ahead_evaluation=False, multi_step_ahead_evaluation=True, export_to_csv=False):
        """
        Runs all evaluations (evaluates each model on each time series)

        :param one_step_ahead_evaluation: a boolean specifying whether to evaluate models on one-step-ahead forecasts
        :param multi_step_ahead_evaluation: a boolean specifying whether to evaluate models on multi-step-ahead forecasts
        :param export_to_csv: a boolean specifying whether to export results to a csv file
        """

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
                self.one_step_ahead_evaluations.to_csv('one_step_ahead_evaluations.csv')

        if multi_step_ahead_evaluation:
            self.multi_step_ahead_evaluations.loc['Mean'] = self.multi_step_ahead_evaluations.mean()
            print('Multi-Step-Ahead Evaluation (Mean Absolute Percentage Error)')
            print(self.multi_step_ahead_evaluations)
            if export_to_csv:
                self.multi_step_ahead_evaluations.to_csv('multi_step_ahead_evaluations.csv')
