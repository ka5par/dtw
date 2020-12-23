
# Download data from Yahoo/Investing.
# Put tickers into data/stocks_cfs
# Compiles the data together saves the adjusted closes.
# Calculate returns

# Links:
# https://stackoverflow.com/questions/54854276/no-data-fetched-web-datareader-panda
# https://medium.com/wealthy-bytes/5-lines-of-python-to-automate-getting-the-s-p-500-95a632e5e567
# https://pythonprogramming.net/combining-stock-prices-into-one-dataframe-python-programming-for-finance/
# https://pythonprogramming.net/sp500-company-price-data-python-programming-for-finance/

import numpy as np
import pandas as pd
import pandas_datareader.data as web
import investpy

import datetime as dt
import os
import shutil
import pickle

from tqdm import trange


def get_data_from_yahoo(tickers, years=15):

    if not os.path.exists('data/stock_dfs'):
        os.makedirs('data/stock_dfs')

    start = dt.datetime.now() - dt.timedelta(days=years*365)
    end = dt.datetime.now()
    print("Downloading from yahoo.")

    for i in trange(len(tickers)):
        ticker = tickers[i]
        if not os.path.exists('data/stock_dfs/{}.csv'.format(ticker)):
            df = web.get_data_yahoo(str(ticker), start, end)
            df.reset_index(inplace=True)
            df.set_index("Date", inplace=True)
            df.to_csv('data/stock_dfs/{}.csv'.format(ticker))


def get_data_from_investing(tickers, countries, years=15, t_type="index"):

    if t_type != "index":
        print("Currently, only work with indexes from investing.com.")
        return

    if not os.path.exists('data/stock_dfs'):
        os.makedirs('data/stock_dfs')

    start = dt.datetime.now() - dt.timedelta(days=years * 365)
    end = dt.datetime.now()
    print("Downloading from investing.com")

    for i in trange(len(tickers)):
        ticker = tickers[i]
        if not os.path.exists('data/stock_dfs/{}.csv'.format(ticker)):
            df = investpy.get_index_historical_data(index=ticker, from_date=str(start.strftime("%d/%m/%Y")),
                                                    to_date=str(end.strftime("%d/%m/%Y")),
                                                    country=countries[i])
            df.reset_index(inplace=True)
            df.set_index("Date", inplace=True)
            df.to_csv('data/stock_dfs/{}.csv'.format(ticker))


def compile_data(tickers):

    df = pd.DataFrame()
    print("Compiling Adjusted Close.", "/n")

    for i in trange(len(tickers)):
        ticker = tickers[i]
        try:
            temp = pd.read_csv('data/stock_dfs/{}.csv'.format(ticker))
        except FileNotFoundError:
            continue

        temp.set_index('Date', inplace=True)

        temp.rename(columns={'Adj Close': ticker}, inplace=True)
        temp.drop(['Open', 'High', 'Low', 'Close', 'Volume'], 1, inplace=True)

        if df.empty:
            df = temp
        else:
            df = df.join(temp, how='outer')

    df["monthID"] = pd.to_datetime(df.index.values).year*100 + pd.to_datetime(df.index.values).month
    df.to_csv("data/combined_data.csv")

    return df


def calculate_returns(file):

    print("Calculating returns.")
    df = pd.read_csv(file, index_col=0)
    df.index = pd.to_datetime(df.index)
    returns = df.loc[df.groupby(df.index.to_period('M')).apply(lambda x: x.index.max())]  # https://stackoverflow.com/questions/48288059/how-to-get-last-day-of-each-month-in-pandas-dataframe-index-using-timegrouper
    returns = returns.pct_change()
    returns = returns.shift(-1)  # Shift them up by 1 because, they are to be used as predictions per id.
    returns["monthID"] = pd.to_datetime(returns.index.values).year * 100 + pd.to_datetime(returns.index.values).month
    returns.to_csv("data/next_month_returns.csv")


tickers_yahoo = ["^GSPC", "^DJI", "^GDAXI", "^FCHI", "^N225", "^VIX"]

# tickers_investing = ["S&P 500", "FTSE 100", "DAX", "CAC 40", "TOPIX"]
# countries_investing = ["United States", "United Kingdom", "Germany", "France", "Japan"]

# get_data_from_investing(tickers_investing, countries_investing, years=40)
# compile_data(tickers_investing)


get_data_from_yahoo(tickers_yahoo, years=40)
compile_data(tickers_yahoo)

calculate_returns("data/combined_data.csv")
