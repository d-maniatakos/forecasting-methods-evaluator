from evaluation import Evaluation
from time_series import TimeSeries
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('datasets/AirPassengers.csv')
df.index = pd.date_range(start='1949-01', end='1960-12', freq='MS')
df = df['#Passengers']

df2 = pd.read_csv('datasets/monthly-car-sales.csv')
df2.index = pd.date_range(start='1960-01', end='1968-12', freq='MS')
df2 = df2['Sales']

df4 = pd.read_csv('datasets/monthly-beer-production-in-australia.csv', index_col='Month')
df4.index = pd.date_range(start='1956-01', periods=len(df4), freq='MS')
df4 = df4['Monthly beer production']

df5 = pd.read_csv('datasets/perrin-freres-monthly-champagne.csv', index_col='Month')
df5.index = pd.date_range(start='1964-01', periods=len(df5), freq='MS')
df5 = df5['Value']

df6 = pd.read_csv('datasets/electricity_consumption.csv', index_col='Bill_Date')
df6.index = pd.date_range(start='2016-01', periods=len(df6), freq='MS')
df6 = df6['Usage_charge']

df7 = pd.read_csv('datasets/daily-min-temperatures.csv', index_col='Date')
df7.index = pd.date_range(start='1981-01-01', periods=len(df7), freq='D')
df7 = df7['Temp']

df8 = pd.read_csv('datasets/daily-total-female-births.csv', index_col='Date')
df8.index = pd.date_range(start='1959-01-01', periods=len(df8), freq='D')
df8 = df8['Births']

df9 = pd.read_csv('datasets/yearly-water-usage.csv')
df9.index = pd.date_range(start='1885', periods=len(df9), freq='Y')
df9 = df9['Water']

df10 = pd.read_csv('datasets/longley.csv')
df10.index = pd.date_range(start='1947', periods=len(df10), freq='Y')
df10 = df10['Employed']

df11 = pd.read_csv('datasets/monthly_sailing_traffic.csv')
df11.index = pd.date_range(start='2018-01-01', periods=len(df11), freq='M')
df11 = df11['passengercount']

ts1 = TimeSeries('Air Passengers', df, 'MS', 12)
ts2 = TimeSeries('Car Sales', df2, 'MS', 12)
ts3 = TimeSeries('Monthly Beer Production', df4[:100], 'MS', 12)
ts4 = TimeSeries('Monthly Champagne', df5, 'MS', 12)
ts5 = TimeSeries('Monthly Electricity Consumption', df6, 'MS', 12)
ts6 = TimeSeries('Daily Min Temperatures', df7, 'D', 365)
ts7 = TimeSeries('Daily Total Female Births', df8, 'D', 7)
ts8 = TimeSeries('Yearly Water Usage', df9, 'Y', None)
ts9 = TimeSeries('Longley', df10, 'Y', None)
ts10 = TimeSeries('Sailing Traffic', df11, 'M', 12)

Evaluation([ts1, ts2, ts3, ts4, ts5, ts6, ts7, ts8, ts9, ts10]).evaluate()
