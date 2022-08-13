from forecasting_models.arima import ARIMA
from time_series import TimeSeries
import pandas as pd

df = pd.read_csv('datasets/AirPassengers.csv')
df.index = pd.date_range(start='1949-01', end='1961-01', freq='M').to_period('M')
df = df['#Passengers']
#
# df = pd.read_csv('datasets/monthly-car-sales.csv')
# df.index = pd.date_range(start='1960-01', end='1969-01', freq='M').to_period('M')
# df = df['Sales']
# df

# df = pd.read_csv('datasets/shampoo_sales.csv')
# df.index = pd.date_range(start='1990-01', end='1993-01', freq='M').to_period('M')
# df = df['Sales']
# df

# df = pd.read_csv('datasets/UBER.csv')
# df.index = df['Date']
# df = df['Adj Close']
# df.index = pd.to_datetime(df.index)
# df = df.resample(rule='M').sum()
# df.index = pd.date_range(start='2019-05', end='2022-04', freq='M').to_period('M')

# df = pd.read_csv('datasets/daily-min-temperatures.csv')
# df.index = df['Date']
# df = df['Temp']
# df.index = pd.date_range(start='1981-01-01', end='1990-12-31', freq='D').to_period('D')
# df = df[:len(df)-2500]

ts = TimeSeries('Air Passengers', df, 12)
model = ARIMA()

model.multi_step_ahead_evaluate(ts, 0.7)

