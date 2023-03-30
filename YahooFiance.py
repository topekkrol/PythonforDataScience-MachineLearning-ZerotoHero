import pandas as pd
import yfinance as yf

df = yf.download('AAPL', start='2022-01-01', end='2022-01-31')
print(type(df))
x = df.loc['2022-01-21']
df.index = pd.to_datetime(df.index)
print(x)
print(type(df.index))

