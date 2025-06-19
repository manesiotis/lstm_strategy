import yfinance as yf
import pandas as pd

def load_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    data = data[['Close']]
    data.rename(columns={'Close': 'Price'}, inplace=True)
    data.dropna(inplace=True)
    return data
