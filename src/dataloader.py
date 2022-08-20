import pandas as pd
from time_series import TimeSeries


class DataLoader:
    """
    Class responsible for loading time series data (from csv files) to be used for the evaluation(s)
    """

    def __init__(self):
        """
        Initializes the time_series_list field (as an empty list)
        """

        self.time_series_list = []

    def load_ts(self, file_name, ts_name, value_column, starting_date, freq, seasonality):
        """
        Creates a TimeSeries instance for the loaded time series and appends it to the time series list. Use it to add
        more time series!

        :param file_name: The name of the csv file containing the time series data
        :param ts_name: The desirable name/title for this time series (which will be shown on evaluation results)
        :param value_column: The name of the csv column containing the time series' values
        :param starting_date: The first date of the time series (should be in YYYY-MM-DD format)
        :param freq: The sampling frequency of the time series. Valid frequencies so far:
                    'M' for Month End
                    'MS' for Month Start
                    'D' for Day
                    'Y' for Year
        :param seasonality: An integer specifying the seasonality of the time series
        """

        try:
            df = pd.read_csv(file_name)
            df.index = pd.date_range(start=starting_date, periods=len(df), freq=freq)
            series = df[value_column]
            ts = TimeSeries(ts_name, series, freq, seasonality)
            self.time_series_list.append(ts)
        except:
            pass
