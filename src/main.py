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

ts = TimeSeries('Air Passengers', df, 'MS', 12)
ts2 = TimeSeries('Car Sales', df2, 'MS', 12)
ts3 = TimeSeries('Monthly Beer Production', df4[:100], 'MS', 12)
ts4 = TimeSeries('Monthly Champagne', df5, 'MS', 12)
ts5 = TimeSeries('Monthly Electricity Consumption', df6, 'MS', 12)

Evaluation([ts,ts2,ts3,ts4,ts5]).evaluate()
