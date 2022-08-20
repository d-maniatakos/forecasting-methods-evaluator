from dataloader import DataLoader
from evaluation import Evaluation


if __name__ == "__main__":
    data_loader = DataLoader()
    data_loader.load_ts('datasets/AirPassengers.csv', 'Air Passengers', '#Passengers', '1949-01', 'MS', 12)
    data_loader.load_ts('datasets/monthly-car-sales.csv', 'Monthly Car Sales', 'Sales', '1960-01', 'MS', 12)
    data_loader.load_ts('datasets/monthly-beer-production-in-australia.csv', 'Monthly Beer Production',
                 'Monthly beer production', '1956-01', 'MS', 12)
    data_loader.load_ts('datasets/perrin-freres-monthly-champagne.csv', 'Monthly Champagne', 'Value',
                 '1964-01', 'MS', 12)
    data_loader.load_ts('datasets/electricity_consumption.csv', 'Electricity Consumption', 'Usage Charge', '2016-01', 'MS',
                 12)
    data_loader.load_ts('datasets/daily-min-temperatures.csv', 'Daily Min Temperatures', 'Temp', '1981-01-01', 'D', 365)
    data_loader.load_ts('datasets/daily-total-female-births.csv', 'Daily Total Female Births', 'Births', '1959-01-01', 'D',
                 None)
    data_loader.load_ts('datasets/yearly-water-usage.csv', 'Yearly Water Usage', 'Water', '1885', 'Y', None)
    data_loader.load_ts('datasets/longley.csv', 'Longley', 'Employed', '1947', 'Y', None)
    data_loader.load_ts('datasets/monthly_sailing_traffic.csv', 'Monthly Sailing Traffic', 'passengercount', '2018-01-01',
                 'MS', 12)

    time_series_list = data_loader.time_series_list
    Evaluation(time_series_list).evaluate()
