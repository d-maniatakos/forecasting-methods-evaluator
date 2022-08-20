import pandas as pd
from time_series import TimeSeries


class DataLoader:
    def __init__(self):
        self.time_series_list = []
        self.load_ts('datasets/AirPassengers.csv', 'Air Passengers', '#Passengers', '1949-01', 'MS', 12)
        self.load_ts('datasets/monthly-car-sales.csv', 'Monthly Car Sales', 'Sales', '1960-01', 'MS', 12)
        self.load_ts('datasets/monthly-beer-production-in-australia.csv', 'Monthly Beer Production',
                     'Monthly beer production', '1956-01', 'MS', 12)
        self.load_ts('datasets/perrin-freres-monthly-champagne.csv', 'Monthly Champagne', 'Value',
                     '1964-01', 'MS', 12)
        self.load_ts('datasets/electricity_consumption.csv', 'Electricity Consumption', 'Usage Charge', '2016-01', 'MS',
                     12)
        self.load_ts('datasets/daily-min-temperatures.csv', 'Daily Min Temperatures', 'Temp', '1981-01-01', 'D', 365)
        self.load_ts('datasets/daily-total-female-births.csv', 'Daily Total Female Births', 'Births', '1959-01-01','D', None)
        self.load_ts('datasets/yearly-water-usage.csv', 'Yearly Water Usage', 'Water', '1885', 'Y', None)
        self.load_ts('datasets/longley.csv', 'Longley', 'Employed', '1947', 'Y', None)
        self.load_ts('datasets/monthly_sailing_traffic.csv', 'Monthly Sailing Traffic', 'passengercount', '2018-01-01', 'MS', 12)

    def load_ts(self, file_name, ts_name, value_column, starting_date, freq, seasonality):
        try:
            df = pd.read_csv(file_name)
            df.index = pd.date_range(start=starting_date, periods=len(df), freq=freq)
            series = df[value_column]
            ts = TimeSeries(ts_name, series, freq, seasonality)
            self.time_series_list.append(ts)
        except:
            pass

    def get_time_series_list(self):
        return self.time_series_list
