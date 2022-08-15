from time_series import TimeSeries
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from matplotlib import pyplot as plt
import time
import statistics


class ForecastingModel:
    def __init__(self, name):
        self.name = name

    def forecast(self, data, horizon):
        pass

    def one_step_ahead_evaluate(self, ts, train_ratio=0.7, suppress_iterations=True):
        train_len = int(len(ts.data) * train_ratio)
        test_len = len(ts.data) - train_len

        forecasted_values = []
        true_values = ts.data[train_len:].tolist()
        total_times = []

        for i in range(test_len):
            if not suppress_iterations:
                print('Iteration ' + str(i+1) + '/' + str(test_len))
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

        scores_dict = {'mae': mae, 'mape': mape, 'tt': end_time-start_time}

        # print(scores_dict)

        return scores_dict

    def __str__(self):
        return self.name
