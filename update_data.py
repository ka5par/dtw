# Get list of sp500 companies
# Download last 365 calendar day daily info (Date,High,Low,Open,Close,Volume,Adj Close)
# Put tickers into data/stocks_cfs
# Put ticker data it into a folder name data/stock_cfs.
# Compiles the data together saves the adjusted closes.
# Get DtD diffs and normalizes the data between -1 and 1.

# Links:
# https://stackoverflow.com/questions/54854276/no-data-fetched-web-datareader-panda
# https://medium.com/wealthy-bytes/5-lines-of-python-to-automate-getting-the-s-p-500-95a632e5e567
# https://pythonprogramming.net/combining-stock-prices-into-one-dataframe-python-programming-for-finance/
# https://pythonprogramming.net/sp500-company-price-data-python-programming-for-finance/

import numpy as np
import pandas as pd
import pandas_datareader.data as web

import datetime as dt
import os
import shutil
import pickle

from tqdm import trange


def save_sp500_tickers(purge=False):

    if purge:
        if os.path.exists("data/stock_dfs"):
            shutil.rmtree("data/stock_dfs")
        if os.path.exists("data/sp500.csv"):
            os.remove("data/sp500.csv")

    table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    df = table[0]

    if not os.path.exists('data/stock_dfs'):
        os.makedirs('data/stock_dfs')

    df.to_csv("data/sp500.csv", index=False)
    tickers = list(df['Symbol'])

    with open("data/sp500tickers.pickle", "wb") as f:
        pickle.dump(tickers, f)

    return tickers


def get_data_from_yahoo(reload_sp500=False, companies=True, years=15):

    if companies:
        if reload_sp500:
            tickers = save_sp500_tickers(reload_sp500)
        else:
            with open("data/sp500tickers.pickle", "rb") as f:
                tickers = pickle.load(f)

    start = dt.datetime.now() - dt.timedelta(days=years*365)
    end = dt.datetime.now()
    errors = []
    print("Starting update.")

    if companies:
        for i in trange(len(tickers)):
            ticker = tickers[i]
            if not os.path.exists('data/stock_dfs/{}.csv'.format(ticker)):
                try:
                    df = web.get_data_yahoo(str(ticker), start, end)
                    df.reset_index(inplace=True)
                    df.set_index("Date", inplace=True)
                    df.to_csv('data/stock_dfs/{}.csv'.format(ticker))
                except KeyError:
                    errors.append(ticker)
                    continue
        print("Errors:", errors)
    else:
        df = web.get_data_yahoo("^GSPC", start, end)
        df.reset_index(inplace=True)
        df.set_index("Date", inplace=True)
        df.to_csv('data/stock_dfs/sp500.csv')


def compile_data(companies=True):
    if companies:
        with open("data/sp500tickers.pickle", "rb") as f:
            tickers = pickle.load(f)
    else:
        tickers = ["sp500"]

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
    df.to_csv("data/combined_sp500.csv")
    return df


def normalize_data(companies=True):
    df = compile_data(companies)
    print("Percentage change and normalize.")
    df_diff = df.pct_change().iloc[1:].fillna(0)  # Drop first row and nan is 0
    df_norm = df_diff.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))

    df_norm["monthID"] = pd.to_datetime(df_norm.index.values).year * 100 + pd.to_datetime(df_norm.index.values).month
    df_norm.to_csv("data/normalized_sp500.csv")


def calculate_returns(stocks):
    print("Calculate returns.")
    df = pd.read_csv(stocks, index_col=0)
    df.index = pd.to_datetime(df.index)
    returns = df.loc[df.groupby(df.index.to_period('M')).apply(lambda x: x.index.max())]  # https://stackoverflow.com/questions/48288059/how-to-get-last-day-of-each-month-in-pandas-dataframe-index-using-timegrouper
    returns = returns.pct_change()
    returns = returns.shift(-1)  # Shift them up by 1 because, they are to be used as predictions per id.
    returns["monthID"] = pd.to_datetime(returns.index.values).year * 100 + pd.to_datetime(returns.index.values).month
    returns.to_csv("data/next_month_returns.csv")


get_data_from_yahoo(companies=False, years=40)
normalize_data(False)
calculate_returns("data/combined_sp500.csv")
