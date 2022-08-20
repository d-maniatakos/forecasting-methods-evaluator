from time_series import TimeSeries
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from matplotlib import pyplot as plt
import time
import statistics


class ForecastingModel:
    """
    Parent class for all forecasting models
    """

    def __init__(self, name):
        """
        Initializes model field

        :param name: A string for the name of the model
        """
        self.name = name

    def forecast(self, data, horizon):
        pass

    def one_step_ahead_evaluate(self, ts, train_ratio=0.7, suppress_iterations_print=True, max_iterations=20):
        """
        Method responsible for calculating evaluation metrics for one-step-ahead forecasts

        :param ts: A TimeSeries object to evaluate the model on
        :param train_ratio: A float specifying initial ratio of the time series to be used as first training set
        :param suppress_iterations_print: A boolean specifying whether to print iterations counter
        :param max_iterations: An integer for the maximum desirable number of iterations (keep it low to avoid high running times)
        :return: A dictionary with mean absolute error result ("mae"), mean absolute percentage error result ("mape) and mean time taken ("mtt")
        """
        train_len = int(len(ts.data) * train_ratio)
        test_len = len(ts.data) - train_len

        forecasted_values = []
        true_values = ts.data[train_len:].tolist()[:max_iterations]
        total_times = []

        for i in range(min(test_len, max_iterations)):
            if not suppress_iterations_print:
                print('Iteration ' + str(i + 1) + '/' + str(min(test_len, max_iterations)))
            train_ts = TimeSeries('', ts.data[:train_len + i], ts.frequency, ts.seasonality)
            start_time = time.time()
            forecasts = self.forecast(train_ts)
            end_time = time.time()
            total_times.append(end_time - start_time)
            forecasted_values.append(forecasts[0])

        mae = mean_absolute_error(true_values, forecasted_values)
        mape = mean_absolute_percentage_error(true_values, forecasted_values)
        mtt = statistics.mean(total_times)

        scores_dict = {'mae': mae, 'mape': mape, 'mtt': mtt}

        # print(scores_dict)
        #
        # plt.plot(true_values)
        # plt.plot(forecasted_values)
        # plt.show()

        return scores_dict

    def multi_step_ahead_evaluate(self, ts, train_ratio=0.7):
        """
        Method responsible for calculating evaluation metrics for multi-step-ahead forecasts

        :param ts: A TimeSeries object to evaluate the model on
        :param train_ratio: A float specifying ratio of the time series to be used as training set
        :return: A dictionary with mean absolute error result ("mae"), mean absolute percentage error result ("mape) and total time taken ("tt")
        """
        train_len = int(len(ts.data) * train_ratio)
        test_len = len(ts.data) - train_len

        true_values = ts.data[train_len:].tolist()

        train_ts = TimeSeries('', ts.data[:train_len], ts.frequency, ts.seasonality)

        start_time = time.time()
        forecasted_values = self.forecast(train_ts, test_len).tolist()
        end_time = time.time()

        # plt.plot(true_values)
        # plt.plot(forecasted_values)
        # plt.show()

        mae = mean_absolute_error(true_values, forecasted_values)
        mape = mean_absolute_percentage_error(true_values, forecasted_values)

        scores_dict = {'mae': mae, 'mape': mape, 'tt': end_time - start_time}

        # print(scores_dict)

        return scores_dict

    def __str__(self):
        return self.name
