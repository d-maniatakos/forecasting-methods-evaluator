from evaluation import Evaluation
from time_series import TimeSeries
import pandas as pd

df = pd.read_csv('datasets/AirPassengers.csv')
df.index = pd.date_range(start='1949-01', end='1960-12', freq='MS')
df = df['#Passengers']

df2 = pd.read_csv('datasets/monthly-car-sales.csv')
df2.index = pd.date_range(start='1960-01', end='1968-12', freq='MS')
df2 = df2['Sales']

df3 = pd.read_csv('datasets/shampoo_sales.csv')
df3.index = pd.date_range(start='1990-01', end='1992-12', freq='MS')
df3 = df3['Sales']

df4 = pd.read_csv('datasets/UBER.csv')
df4.index = df4['Date']
df4 = df4['Adj Close']
df4.index = pd.to_datetime(df4.index)
df4 = df4.resample(rule='M').sum()
df4.index = pd.date_range(start='2019-05', end='2022-03', freq='MS')
#
# df = pd.read_csv('datasets/daily-min-temperatures.csv')
# df.index = df['Date']
# df = df['Temp']
# df.index = pd.date_range(start='1981-01-01', end='1990-12-31', freq='D').to_period('D')
# df = df[:1000]


ts = TimeSeries('Air Passengers', df, 'MS', 12)
ts2 = TimeSeries('Car Sales', df2, 'MS', 12)
ts3 = TimeSeries('Shampoo Sales', df3, 'MS', 12)
ts4 = TimeSeries('UBER Stock', df4, 'MS', None)

Evaluation([ts4, ts, ts2, ts3]).evaluate()



